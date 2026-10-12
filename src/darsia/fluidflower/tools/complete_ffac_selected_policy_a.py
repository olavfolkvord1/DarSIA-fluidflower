"""Operational continuation from real Policy A completion to review-only Studio.

No inference, fitting, scoring or acceptance rules are implemented here. Every
stage calls an immutable producer/review CLI and preserves its actual receipt.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import socket,subprocess,time


def now():return datetime.now(timezone.utc).isoformat()
def asset(path):
    p=Path(path);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def bound(item):
    raw=Path(item['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('Bound operational artifact changed: '+item['path'])
    return json.loads(raw)
def write_new(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def phase_completed(record,plan_descriptor,runs):
    rows=record.get('results',[])
    if (record.get('complete') is not True or record.get('failure') is not None
            or record.get('plan')!=plan_descriptor or record.get('all_runs')!=len(runs)
            or len(rows)!=len(runs) or {r.get('run') for r in rows}!=set(runs)
            or any(r.get('complete') is not True for r in rows)):
        raise ValueError('Actual successful phase run receipts for all 39 runs are required')
    return True


class Controller:
    def __init__(self,config_path):
        self.descriptor=asset(config_path);self.config=bound(self.descriptor)
        self.plan=bound(self.config['phase_plan']);self.ops=Path(self.config['operations'])
        if self.ops.exists():raise FileExistsError('Use a new completion operation')
        self.ops.mkdir(parents=True)
        if socket.gethostname().casefold()!=self.config['approved_main_host'].casefold():
            raise ValueError('Completion controller belongs to Main')
        self.stop=Path(self.plan['phase_campaign_binding']['shared_stop_path'])
        self.records=[]
        write_new(self.ops/'process.json',{'contract':'ffac-policy-a-review-controller-process-v1',
            'pid':os.getpid(),'host':socket.gethostname(),'started_utc':now(),'config':self.descriptor,
            'phase_plan':self.config['phase_plan'],'phase_inference':False,'scorer_pending':True,'round3_allowed':False})

    def wait_file(self,path,hours=48):
        deadline=time.monotonic()+hours*3600
        while not Path(path).is_file():
            if self.stop.exists():raise ValueError('Shared stop; no downstream publication')
            if time.monotonic()>deadline:raise TimeoutError('Waiting for '+str(path))
            time.sleep(self.config.get('poll_seconds',30))
        if self.stop.exists():raise ValueError('Shared stop; no downstream publication')
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))

    def start(self,name,module,arguments,runtime):
        if self.stop.exists():raise ValueError('Shared stop before '+name)
        folder=self.ops/name;folder.mkdir()
        env=os.environ.copy();env['PYTHONPATH']=runtime
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        command=[self.config['python'],'-m',module,*map(str,arguments)]
        out=(folder/'stdout.log').open('wb');err=(folder/'stderr.log').open('wb')
        child=subprocess.Popen(command,env=env,stdout=out,stderr=err)
        write_new(folder/'process.json',{'pid':child.pid,'started_utc':now(),'command':command,
            'runtime_src':runtime,'phase_plan':self.config['phase_plan'],'phase_inference':False})
        return child,out,err,folder

    def finish(self,handle):
        child,out,err,folder=handle
        while child.poll() is None:time.sleep(10)
        out.close();err.close();row={'stage':folder.name,'exit_code':child.returncode,'finished_utc':now()}
        write_new(folder/'done.json',row);self.records.append(row)
        if child.returncode!=0:raise RuntimeError('Stage failed without retry: '+folder.name+'; exit '+str(child.returncode))

    def call(self,name,module,args,runtime):self.finish(self.start(name,module,args,runtime))

    def execute(self):
        config=self.config;root=config['candidate'];plan=config['phase_plan']['path'];identity=self.plan['identity_gate']['path']
        phase_completed(self.wait_file(config['phase_supervisor_done']),config['phase_plan'],self.plan['runs'])
        immutable=config['producer_runtime_src'];review=config['review_runtime_src']
        main_display=self.start('main_display','darsia.fluidflower.tools.render_ffac_selected_display_sources',
            ['--manifest',config['main_display_manifest']['path'],'--destination',config['main_display_stage'],
             '--workers',str(config['main_display_workers'])],review)
        completion=Path(root)/'policy_a_completion_v1.json'
        report=Path(root)/'policy_a_saved_report_v1'
        export=Path(root)/'claude_prediction_export_v1'
        finisher='darsia.fluidflower.tools.finish_ffac_selected_policy_a_baseline_inventory'
        common=['--plan',plan,'--root',root,'--identity-gate',identity]
        self.call('collect',finisher,[*common,'--stage','collect','--destination',completion],immutable)
        self.call('spatial_report',finisher,[*common,'--stage','report','--destination',report],immutable)
        self.call('claude_export',finisher,[*common,'--stage','export','--report',report/'report.json','--destination',export],immutable)
        self.finish(main_display)
        remote=self.wait_file(config['olav_display_stage'])
        main=json.loads(Path(config['main_display_stage']).read_text())
        for name,saved,manifest in [('main',main,config['main_display_manifest']),('olav',remote,config['olav_display_manifest'])]:
            declaration=bound(manifest)
            if (saved.get('complete') is not True or saved.get('manifest')!=manifest
                    or saved.get('phase_plan')!=config['phase_plan'] or saved.get('phase_inference') is not False
                    or saved.get('expected_frames')!=sum(map(len,declaration['runs'].values()))
                    or len(saved.get('results',[]))!=len(declaration['runs'])
                    or any(r.get('complete') is not True for r in saved.get('results',[]))):
                raise ValueError('Display shard incomplete or changed: '+name)
        if main['expected_frames']+remote['expected_frames']!=1603:raise ValueError('Honest timeline requires 1603 photographic frames')
        gate=Path(root)/'selected_policy_a_review_completion_v1.json'
        self.call('review_seal','darsia.fluidflower.tools.seal_ffac_selected_policy_a_review',
            ['--plan',plan,'--candidate',root,'--completion',completion,'--report',report/'report.json','--destination',gate],review)
        self.call('studio_package','darsia.fluidflower.tools.build_ffac_selected_policy_a_review',
            ['--candidate',root,'--phase-plan',plan,'--accepted-gate',gate],review)
        self.call('studio_register','darsia.fluidflower.tools.register_ffac_selected_policy_a_studio_candidate',
            ['--candidate',root,'--results',config['results_root']],review)
        return {'complete':True,'phase_completion':asset(completion),'report':asset(report/'report.json'),
            'claude_export':asset(export/'export_receipt.json'),'review_seal':asset(gate),
            'studio_package':asset(Path(root)/'selected_policy_a_studio_receipt.json'),
            'registered_nondefault':True,'claude_scorer_gate_pending':True,'round3_allowed':False,
            'mass_admission':False,'bottom_mass_allowed':False,'scorer_executed':False,'default_changed':False}

    def run(self):
        try:result=self.execute()
        except Exception as error:
            result={'complete':False,'error':repr(error),'registered_nondefault':False,
                'claude_scorer_gate_pending':True,'round3_allowed':False,'scorer_executed':False}
        result.update(contract='ffac-policy-a-complete-to-review-controller-v1',finished_utc=now(),
            config=self.descriptor,phase_plan=self.config['phase_plan'],stages=self.records,code=asset(__file__))
        write_new(self.ops/'controller_done.json',result)
        print(json.dumps(result),flush=True)
        return 0 if result['complete'] else 2


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    return Controller(parser.parse_args(argv).config).run()
if __name__=='__main__':raise SystemExit(main())
