"""Durable operational controller for one source-bound dual-host T2 job.

This standalone helper uses only the standard library. Its config declares
external commands and receipts; the scientific snapshot remains unchanged.
No stage is retried automatically after a missing/failed terminal receipt.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

CONTRACT='ffac-t2-dual-host-controller-v1'
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write(path,value,*,mutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not mutable and path.exists():
        if read(path)!=value:raise ValueError(f'Refusing to replace an immutable controller record: {path}')
        return
    temp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
    with temp.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    try:
        if mutable:os.replace(temp,path)
        elif os.name=='nt':temp.rename(path)
        else:os.link(temp,path);temp.unlink()
    finally:temp.unlink(missing_ok=True)


def field(value,key):
    for part in key.split('.'):
        value=value[int(part)] if isinstance(value,list) else value[part]
    return value


def receipt(spec):
    path=Path(spec['path'])
    if not path.exists():return None
    value=read(path)
    for key,expected in spec.get('require',{}).items():
        actual=field(value,key)
        if actual!=expected or (type(expected) is bool and type(actual) is not bool):
            raise ValueError(f'Terminal receipt violates {key}: {path}')
    for match in spec.get('matches',[]):
        reference=read(match['path'])
        if field(value,match['field'])!=field(reference,match['reference_field']):
            raise ValueError(f'Terminal receipt lost source binding {match["field"]}: {path}')
    return dict(path=str(path),sha256=digest(path))


def detect_failure(step):
    for raw in step.get('failure_paths',[]):
        if Path(raw).exists():raise RuntimeError(f'Declared worker/stage failure: {raw}')
    for spec in step.get('terminal_receipts',[]):
        path=Path(spec['path'])
        if path.exists():
            code=field(read(path),spec.get('exit_code_field','exit_code'))
            if code!=spec.get('success_code',0):raise RuntimeError(f'Remote worker exited with {code}: {path}')


def complete_receipts(step):
    values=[]
    for spec in step.get('success_receipts',[]):
        value=receipt(spec)
        if value is None:return None
        values.append(value)
    return values


def verify_expected_assets(config):
    for asset in config.get('expected_assets',[]):
        if digest(asset['path'])!=asset['sha256']:raise ValueError(f'Controller asset changed: {asset["path"]}')


def allocation_policy(config):
    """Keep legacy CPU12 configs; explicit host budgets may use more resources.

    Scientific workers still apply actual CPU and available-memory admission.
    These declared budgets do not silently expand an existing configuration.
    """
    if 'owner_workers' not in config:
        if type(config.get('workers_per_host')) is not int or config['workers_per_host']!=12:
            raise ValueError('Legacy controller requires explicit CPU12/host')
        return dict(workers_per_host=12, affinity_mask=config.get('affinity_mask',4095))
    owners=config['owner_workers']
    if (not isinstance(owners,dict) or len(owners)!=2
            or any(not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,48}',name)
                   for name in owners)
            or any(type(count) is not int or count<1 for count in owners.values())):
        raise ValueError('Explicit dual-host owner_workers requires two portable owners with positive integer budgets')
    if config.get('workers_per_host') is not None:
        raise ValueError('Use owner_workers without a conflicting workers_per_host limit')
    return dict(owner_workers=owners,affinity_mask=config.get('affinity_mask'))


def set_allocation(mask):
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        process=kernel.GetCurrentProcess()
        if not kernel.SetPriorityClass(process,0x4000):raise ctypes.WinError(ctypes.get_last_error())
        if mask is not None:
            kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
            if not kernel.SetProcessAffinityMask(process,int(mask)):raise ctypes.WinError(ctypes.get_last_error())
    elif mask is not None and hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{i for i in range(int(mask).bit_length()) if int(mask)&(1<<i)})
    for key in THREAD_ENV:os.environ[key]='1'


@contextmanager
def lease(work_root,config_sha256):
    path=Path(work_root)/'controller.pidlock'
    owner=dict(pid=os.getpid(),hostname=socket.gethostname(),token=uuid.uuid4().hex,
        config_sha256=config_sha256,acquired_utc=now())
    path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('x',encoding='utf-8') as stream:
            json.dump(owner,stream);stream.flush();os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(f'Controller already claimed; inspect {path}, no automatic PID recovery') from error
    try:yield
    finally:
        if path.exists() and read(path).get('token')==owner['token']:path.unlink()


def terminate_owned(process):
    if process.poll() is not None:return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],check=False,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        import signal
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:process.kill()


def execute_stop_commands(step,root):
    records=[]
    for index,command in enumerate(step.get('on_failure_commands',[])):
        try:
            result=subprocess.run(command['argv'],cwd=command.get('cwd'),timeout=command.get('timeout_s',60),
                capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            log=Path(root)/f'stop_{index:02d}.log'
            log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            records.append(dict(name=command.get('name',f'stop_{index}'),exit_code=result.returncode,log=str(log)))
        except Exception as error:records.append(dict(name=command.get('name'),error=repr(error)))
    return records


def _alive(pid):
    if os.name=='nt':
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        try:
            code=ctypes.c_uint32()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code)) and code.value==259)
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False


class ProcessWitness:
    """Retain a Windows process handle and verify birth before waiting."""
    def __init__(self,spec):
        self.spec=spec;self.handle=None;self.kernel=None
        if os.name!='nt':
            if not _alive(spec['pid']):self.ended=True
            else:self.ended=False
            return
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
        kernel.OpenProcess.restype=ctypes.c_void_p
        kernel.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.c_void_p]*4
        kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint32)]
        kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        handle=kernel.OpenProcess(0x1000|0x100000,False,int(spec['pid']))
        self.kernel=kernel;self.handle=handle
        if not handle:self.ended=True;return
        creation,exit_time,kernel_time,user_time=[ctypes.c_uint64() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,ctypes.byref(creation),ctypes.byref(exit_time),ctypes.byref(kernel_time),ctypes.byref(user_time)):
            self.close();raise ctypes.WinError(ctypes.get_last_error())
        expected=spec.get('creation_filetime')
        if expected is None and spec.get('birth_utc'):
            raw=spec['birth_utc'];stamp=datetime.fromisoformat(raw.replace('Z','+00:00'))
            delta=stamp-datetime(1601,1,1,tzinfo=timezone.utc)
            expected=(delta.days*86400+delta.seconds)*10_000_000+delta.microseconds*10
            fraction=re.search(r'\.(\d+)',raw)
            if fraction and len(fraction[1])>=7:expected+=int(fraction[1][6])
        if expected is None or int(expected)!=creation.value:
            self.close();raise RuntimeError(f'Declared PID birth changed: {spec["pid"]}')
        self.ended=False

    def alive(self):
        if self.ended:return False
        if os.name!='nt':return _alive(self.spec['pid'])
        import ctypes
        code=ctypes.c_uint32()
        if not self.kernel.GetExitCodeProcess(self.handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
        return code.value==259

    def close(self):
        if self.handle:self.kernel.CloseHandle(self.handle);self.handle=None


def run_step(config,config_sha,index,step,*,resume=False,poll_s=20):
    root=Path(config['work_root'])/'steps'/f'{index:02d}_{step["name"]}'
    root.mkdir(parents=True,exist_ok=True)
    binding=dict(config_sha256=config_sha,step_sha256=canonical(step),controller_sha256=digest(__file__))
    done=root/'complete.json';started=root/'started.json'
    if done.exists():
        saved=read(done)
        if saved['binding']!=binding:raise ValueError('Completed step changed its immutable command/source policy')
        for asset in saved['receipts']:
            if digest(asset['path'])!=asset['sha256']:raise ValueError('Completed stage receipt changed')
        detect_failure(step)
        return saved
    kind=step['kind']
    if kind not in ('wait','command','parallel'):raise ValueError('Unknown controller step kind')
    processes=[];streams=[];launches=[];witnesses=[]
    recovery=False
    if started.exists():
        previous=read(started)
        if not resume or previous['binding']!=binding:raise ValueError('A started stage requires exact --resume')
        if kind=='wait':recovery=True
        elif kind=='parallel':
            # Never relaunch a scheduled remote worker after controller loss.
            # The declared worker terminal and completion receipts settle it.
            recovery=True
        elif complete_receipts(step) is not None and step.get('success_receipts'):
            recovery=True
        else:raise RuntimeError('Command stage was interrupted without its declared terminal receipt; inspect, do not retry blindly')
    elif kind=='wait':
        write(started,dict(binding=binding,started_utc=now(),launches=[]))
    else:
        commands=step.get('commands') or ([step['command']] if 'command' in step else [])
        if (kind=='command' and len(commands)!=1) or not commands:raise ValueError('Stage command count is invalid')
        # This no-clobber intent must precede every launch. A crash between
        # launches is recorded and requires inspection, never blind duplication.
        write(started,dict(binding=binding,started_utc=now(),launches_intended=commands))
        try:
            for number,command in enumerate(commands):
                stdout=root/f'{number:02d}_{command.get("name","command")}.stdout.log'
                stderr=root/f'{number:02d}_{command.get("name","command")}.stderr.log'
                out,err=stdout.open('ab'),stderr.open('ab');streams.extend((out,err))
                env=os.environ.copy();env.update(command.get('env',{}))
                for key in THREAD_ENV:env[key]='1'
                process=subprocess.Popen(command['argv'],cwd=command.get('cwd'),env=env,stdin=subprocess.DEVNULL,
                    stdout=out,stderr=err,creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS) if os.name=='nt' else 0,
                    start_new_session=os.name!='nt')
                processes.append(process)
                item=dict(name=command.get('name'),argv=command['argv'],pid=process.pid,hostname=socket.gethostname(),
                    stdout=str(stdout),stderr=str(stderr),kind=command.get('kind','process'),started_utc=now())
                item['success_receipts']=command.get('success_receipts',[])
                launches.append(item);write(root/f'launch_{number:02d}.json',dict(binding=binding,**item))
        except BaseException as error:
            for process in processes:terminate_owned(process)
            for stream in streams:stream.close()
            stops=execute_stop_commands(step,root)
            write(root/'failure.json',dict(binding=binding,error=repr(error),failed_utc=now(),remote_stop_results=stops,no_automatic_retry=True))
            raise
    started_at=time.monotonic()
    try:
        witnesses=[ProcessWitness(spec) for spec in step.get('watched_processes',[])]
        while True:
            detect_failure(step)
            for witness in witnesses:
                if not witness.alive():
                    required=witness.spec['success_receipt']
                    if receipt(required) is None:raise RuntimeError(f'Declared worker PID {witness.spec["pid"]} ended without its completion receipt')
            exit_codes=[process.poll() for process in processes]
            if any(code is not None and code!=0 for code in exit_codes):raise RuntimeError(f'Local stage command failed: {exit_codes}')
            if recovery and kind=='parallel':
                previous_launches=list(root.glob('launch_*.json'))
                intended=read(started).get('launches_intended',[])
                if len(previous_launches)!=len(intended):raise RuntimeError('Controller stopped between dual-host launches; inspect declared task state')
                for path in previous_launches:
                    old=read(path)
                    if old.get('kind','process')=='process' and old['hostname']==socket.gethostname() and not _alive(old['pid']):
                        own=old.get('success_receipts',[])
                        if not own or any(receipt(spec) is None for spec in own):
                            raise RuntimeError('Previously launched local worker ended without its own completion receipt')
            receipts=complete_receipts(step)
            all_exited=all(code is not None for code in exit_codes)
            if receipts is not None and (kind=='wait' or recovery or all_exited):
                value=dict(binding=binding,complete=True,completed_utc=now(),receipts=receipts,exit_codes=exit_codes,
                    no_automatic_retry=True,cpu_only=True,mass_eligible=False,bottom_mass_allowed=False)
                write(done,value);return value
            if kind=='command' and all_exited and receipts is None:
                raise RuntimeError('Stage exited successfully without its required completion receipt')
            if time.monotonic()-started_at>step.get('timeout_s',24*3600):raise TimeoutError('Declared stage timeout; no automatic retry')
            write(Path(config['work_root'])/'progress.json',dict(contract=CONTRACT,config_sha256=config_sha,
                pid=os.getpid(),hostname=socket.gethostname(),stage=step['name'],stage_index=index,
                updated_utc=now(),complete=False,local_processes=[dict(pid=p.pid,exit_code=p.poll()) for p in processes],
                recovery_wait=recovery),mutable=True)
            time.sleep(poll_s)
    except BaseException as error:
        for process in processes:terminate_owned(process)
        stops=execute_stop_commands(step,root)
        write(root/'failure.json',dict(binding=binding,error=repr(error),failed_utc=now(),
            remote_stop_results=stops,no_automatic_retry=True))
        raise
    finally:
        for stream in streams:stream.close()
        for witness in witnesses:witness.close()


def run(config_path,*,resume=False,poll_s=20):
    config=read(config_path);sha=digest(config_path)
    if (config.get('contract')!=CONTRACT or config.get('cpu_only') is not True
            or config.get('mass_eligible') is not False or config.get('bottom_mass_allowed') is not False
            or set(config.get('blind_hold_runs',[]))!={'ac30','ac49','ac54'}):
        raise ValueError('Controller requires explicit CPU-only, blind and mass holds')
    allocation=allocation_policy(config)
    if not 1<=poll_s<=60:raise ValueError('Progress interval must be 1–60 seconds')
    names=[step['name'] for step in config['steps']]
    if not names or len(set(names))!=len(names) or any(not name.replace('_','').replace('-','').isalnum() for name in names):
        raise ValueError('Stage names must be unique portable names')
    verify_expected_assets(config);set_allocation(allocation['affinity_mask'])
    root=Path(config['work_root'])
    with lease(root,sha):
        binding=dict(contract=CONTRACT,config_sha256=sha,controller_sha256=digest(__file__))
        frozen=root/'controller_binding.json'
        write(frozen,binding)
        results=[]
        try:
            for index,step in enumerate(config['steps']):results.append(run_step(config,sha,index,step,resume=resume,poll_s=poll_s))
        except BaseException as error:
            write(root/'failure_receipt.json',dict(binding=binding,error=repr(error),failed_utc=now(),automatic_retry=False),mutable=True)
            write(root/'progress.json',dict(contract=CONTRACT,config_sha256=sha,pid=os.getpid(),updated_utc=now(),
                complete=False,failed=True,error=repr(error)),mutable=True)
            raise
        value=dict(binding=binding,complete=True,completed_utc=now(),stages=[r['binding']['step_sha256'] for r in results],
            scientific_acceptance=False,mass_eligible=False,bottom_mass_allowed=False,automatic_activation=False)
        if (root/'exit_receipt.json').exists():
            previous=read(root/'exit_receipt.json')
            value['completed_utc']=previous['completed_utc']
        write(root/'exit_receipt.json',value)
        write(root/'progress.json',dict(contract=CONTRACT,config_sha256=sha,pid=os.getpid(),updated_utc=now(),
            complete=True,stage='complete',stages_done=len(results)),mutable=True)
        return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--poll-seconds',type=float,default=20)
    a=p.parse_args(argv);run(a.config,resume=a.resume,poll_s=a.poll_seconds)


if __name__=='__main__':main()
