"""Root's Main servicing lane for one locked Olav round-four phase.

serve is an explicit Root action. It does not launch numerical work. One SCP
lane services the existing port's exact input/publication requests; a separate
ten-second control lane forwards original Main status bytes. All publication
uses the port's original LocalProductionJob/NasIO entrypoint.
"""
from __future__ import annotations

import base64
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

SPEC = "bf69e427d39e4b351dbdeb2f4e3067a6724f51f6b4cc84587ba41e959e3d9143"
AMENDMENT = "f1567b3f8724b41c6401e8d72d139c45910cd4a6aeba09dba75436da2a0882d0"
HOST = "olav_@10.10.10.146"
OPAQUE = "4491b8ea239853d441ece789a371e92f6516874b6ea62308eeafb057981d938c"
CAP = 32 * 1024**2


def _bound(value):
    if (not isinstance(value, dict) or set(value) != {"path","sha256","size_bytes"}
            or not ntpath.isabs(value.get("path", ""))
            or re.fullmatch("[0-9a-f]{64}", value.get("sha256", "")) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0):
        raise ValueError("Exact full absolute content descriptor required")
    return value


def _strict(raw):
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value: raise ValueError("Duplicate transport metadata key")
            value[key] = item
        return value
    def invalid(value): raise ValueError("Nonfinite transport metadata")
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs, parse_constant=invalid)


def _relative(value):
    path = Path(value)
    if (path.is_absolute() or ntpath.isabs(value) or value.startswith(("/","\\"))
            or ntpath.splitdrive(value)[0] or not path.parts
            or any(p in {".",".."} or ".partial" in p for p in path.parts)):
        raise ValueError("Exact ordinary relative immutable file name required")
    return path


def _stamp(path):
    stat = Path(path).stat()
    return [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]


def _metadata(io, descriptor):
    descriptor=_bound(descriptor)
    if descriptor["sha256"]==OPAQUE or descriptor["size_bytes"]>64*1024**2:
        raise ValueError("Opaque/non-bounded metadata must never be JSON parsed")
    return io.read_json(descriptor,max_bytes=descriptor["size_bytes"])


# Only bounded metadata, content checks, and exact owned cleanup run remotely.
# Scientific files are streamed for SHA verification, never decoded.
_REMOTE = r'''
import base64,hashlib,importlib.util,json,os,pathlib,sys,time
import psutil
p=P;op=p['op'];a=p['args'];root=pathlib.Path(p['job_root']).resolve()
t=p['transport'];cap=32*1024**2
def own(path,extra=False):
 q=pathlib.Path(path).resolve()
 roots=[root,pathlib.Path(t['incoming_root']).resolve(),pathlib.Path(t['publication_spool']).resolve()]
 controls={pathlib.Path(t[k]).resolve() for k in ('incoming_mapping','root_status_control')}
 if not q.is_relative_to(pathlib.Path('C:/FFAC_cache').resolve()) or not (q in controls or any(q.is_relative_to(r) for r in roots)):raise ValueError('Remote path escaped its actual phase ownership')
 if q.is_symlink() or q.is_junction():raise ValueError('Remote path substituted')
 return q
def stamp(q):
 s=q.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def asset(path,expected=None):
 q=own(path);before=stamp(q);h=hashlib.sha256();n=0
 with q.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b);n+=len(b)
 if stamp(q)!=before:raise ValueError('Remote immutable file changed during transfer verification')
 v=dict(path=str(q),sha256=h.hexdigest(),size_bytes=n)
 if expected is not None and (v['sha256'],n)!=(expected['sha256'],expected['size_bytes']):raise ValueError('Remote content binding differs')
 return v
def meta(path):
 q=own(path)
 if not q.exists():return None
 with q.open('rb') as f:b=f.read(cap+1)
 if len(b)>cap:raise ValueError('Remote metadata exceeds32MiB')
 def pairs(rows):
  v={}
  for key,item in rows:
   if key in v:raise ValueError('Duplicate remote metadata key')
   v[key]=item
  return v
 def bad(value):raise ValueError('Nonfinite remote metadata')
 v=json.loads(b.decode('utf-8-sig'),object_pairs_hook=pairs,parse_constant=bad)
 return dict(descriptor=dict(path=str(q),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b)),value=v)
def facts(identity):
 try:
  proc=psutil.Process(identity['pid']);actual=dict(pid=proc.pid,process_create_time=str(proc.create_time()))
  return dict(alive=actual==identity,identity=actual,argv=proc.cmdline(),parents=[dict(pid=x.pid,process_create_time=str(x.create_time())) for x in proc.parents()[:4]])
 except psutil.NoSuchProcess:return dict(alive=False)
def getio():
 d=p['io_source'];q=pathlib.Path(d['path']);b=q.read_bytes()
 if len(b)!=d['size_bytes'] or hashlib.sha256(b).hexdigest()!=d['sha256']:raise ValueError('Actual locked remote IO source differs')
 spec=importlib.util.spec_from_file_location('_r4_transport_io',q);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m
 exec(compile(b,str(q),'exec'),m.__dict__);return m.NasIO()
if op=='poll':
 out=dict(ready=meta(own(t['publication_spool'])/'phase_ready.json'),packet_ready=meta(own(t['publication_spool'])/'phase_packet_ready.json'),finished=meta(own(t['publication_spool'])/'phase_finished.json'),failed=meta(root/'operations/failed.json'),requests=[],natives=[],completed_runs=[])
 if out['ready']:out['phase_facts']=facts(out['ready']['value']['actual_phase_owner'])
 total=0
 for q in sorted(own(t['publication_spool']).glob('**/*.request.json')):
  if q.with_name(q.name.replace('.request.json','.ack.json')).exists():continue
  row=meta(q);total+=row['descriptor']['size_bytes']
  if total>8*1024**2:break
  out['requests'].append(row)
  if len(out['requests'])>=32:break
 if (root/'operations').exists():
  for q in sorted((root/'operations').glob('*/completed.json')):
   v=meta(q)['value']
   if v.get('complete') is True and v.get('worker_references_closed') is True:out['completed_runs'].append(q.parent.name)
  for q in sorted((root/'operations').glob('*/native_ready.json')):
   row=meta(q);row['facts']=facts(row['value']['actual_native_worker']);row['worker_packet']=meta(q.parent/'worker_packet.json');out['natives'].append(row)
 result=out
elif op=='prepare_put':
 q=own(a['path']);q.parent.mkdir(parents=True,exist_ok=True)
 if q.exists():raise FileExistsError('Fresh SCP temporary already exists')
 result=dict(path=str(q))
elif op=='install_file':
 io=getio();source=own(a['temporary']);target=own(a['target']);target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  asset(target,a['expected']);raise FileExistsError('Immutable incoming target already exists; preserve new temporary')
 result=io.publish_file(source,target,a['expected'])
 result=dict(result,path=str(target),identity=stamp(target))
elif op=='install_control':
 io=getio();q=own(a['path']);q.parent.mkdir(parents=True,exist_ok=True)
 result=io.replace_json(q,lambda:a['document'],max_bytes=cap)
elif op=='verify':result=asset(a['descriptor']['path'],a['descriptor'])
elif op=='retired':
 if not a['identities'] or any(facts(v)['alive'] for v in a['identities']):raise ValueError('Actual complete worker union has not retired')
 result=dict(actual_worker_retirement_verified=True)
elif op=='cleanup':
 e=a['extra'];owned=own(e['owned_run_directory']);inputs=own(e['owned_inputs_directory']);results=own(e['owned_results_directory'])
 if inputs.parent!=owned or results.parent!=owned or inputs.name!='inputs' or results.name!='results' or owned.name!=a['run']:raise ValueError('Exact owned run input/result directories required')
 if e.get('worker_references_closed') is not True or not e.get('actual_retired_identities'):raise ValueError('Actual closed references and exact retired worker union required')
 if any(facts(v)['alive'] for v in e['actual_retired_identities']):raise ValueError('An actual worker incarnation is still alive')
 if stamp(owned)[:2]!=e['folder_id']:raise ValueError('Owned remote folder identity changed')
 snapshots=[]
 for folder,expected,identities in [(inputs,e['original_inputs'],e['input_identities']),(results,a['files'],e['result_identities'])]:
  paths=list(folder.rglob('*'))
  if any(q.is_symlink() or q.is_junction() or '.partial' in q.name for q in paths) or {q.relative_to(folder).as_posix() for q in paths if q.is_file()}!=set(expected):raise ValueError('Owned remote inventory changed or contains preserved partials')
  if set(identities)!=set(expected):raise ValueError('Exact owned file identity table required')
  for q in paths:
   if q.is_file() and stamp(q)!=identities[q.relative_to(folder).as_posix()]:raise ValueError('Owned remote file changed after verified copy/publication')
  snapshots.append((folder,paths))
 if any(facts(v)['alive'] for v in e['actual_retired_identities']):raise ValueError('Retirement changed before owned cleanup')
 for folder,paths in snapshots:
  for q in paths:
   if q.is_file():q.unlink()
  for q in sorted((q for q in paths if q.is_dir()),key=lambda x:len(x.parts),reverse=True):q.rmdir()
  folder.rmdir()
 owned.rmdir();result=dict(removed=True,run=a['run'],owned_run_directory=str(owned),actual_worker_retirement_verified=True,exact_local_identities_verified=True)
elif op=='forget_input':
 q=own(a['descriptor']['path'])
 if not q.is_relative_to(pathlib.Path(t['incoming_root']).resolve()) or stamp(q)!=a['identity']:raise ValueError('Incoming owned copy identity changed')
 if any(facts(v)['alive'] for v in a['retired']):raise ValueError('Actual retired consumer lives')
 q.unlink();result=dict(removed=True)
else:raise ValueError('Unknown fixed remote operation')
print(json.dumps(result,allow_nan=False,separators=(',',':')))
'''


class _SSH:
    def __init__(self, args, guard):
        self.args,self.guard = args,guard
        self.python = _bound(args["capacity"]["python"])
        self.transport = args["capacity"]["transport"]
        self.scp_lane = threading.Lock()

    def call(self, op, args, *, timeout=90, check_guard=True):
        if check_guard: self.guard()
        payload=dict(source=base64.b64encode(_REMOTE.encode()).decode(),
            source_sha256=hashlib.sha256(_REMOTE.encode()).hexdigest(),
            operation=dict(op=op,args=args,job_root=self.args["job_root"],transport=self.transport,
                           io_source=self.args["sourcepins"]["production_io"]))
        code="import sys,json,base64,hashlib;d=json.load(sys.stdin);s=base64.b64decode(d['source'],validate=True);assert hashlib.sha256(s).hexdigest()==d['source_sha256'];P=d['operation'];exec(compile(s,'<locked-round4-Main-transport>','exec'))"
        quote=lambda s:"'"+str(s).replace("'","''")+"'"
        py=self.python
        script=("$ErrorActionPreference='Stop';$p="+quote(py["path"])+";"
            "if((Get-Item -LiteralPath $p).Length -ne "+str(py["size_bytes"])+") {throw 'Python size changed'};"
            "if((Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower() -ne "+quote(py["sha256"])+") {throw 'Python SHA changed'};"
            "$d=[Console]::In.ReadToEnd();$d | & $p -I -B -c "+quote(code)+";if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}")
        argv=["ssh","-o","BatchMode=yes","-o","StrictHostKeyChecking=yes","-o","ConnectTimeout=10",HOST,
              "powershell.exe","-NoProfile","-NonInteractive","-EncodedCommand",
              base64.b64encode(script.encode("utf-16le")).decode()]
        process=subprocess.run(argv,input=json.dumps(payload,ensure_ascii=True).encode(),capture_output=True,timeout=timeout)
        if process.returncode:raise RuntimeError("Bounded round4 SSH operation failed: "+process.stderr[-2000:].decode(errors="replace"))
        if len(process.stdout)>16*1024**2:raise ValueError("Remote metadata exceeds its finite bound")
        value=_strict(process.stdout)
        if check_guard:self.guard()
        return value

    def put(self, local, remote, expected):
        temporary=remote+"."+uuid.uuid4().hex+".scp.partial"
        with self.scp_lane:
            self.call("prepare_put",dict(path=temporary))
            self.guard()
            result=subprocess.run(["scp","-q","-o","BatchMode=yes","-o","StrictHostKeyChecking=yes","-o","ConnectTimeout=10",
                                   str(local),HOST+":"+temporary.replace("\\","/")],capture_output=True,timeout=600)
            if result.returncode:raise RuntimeError("Exact input SCP failed: "+result.stderr[-1500:].decode(errors="replace"))
            self.guard()
            return self.call("install_file",dict(temporary=temporary,target=remote,expected=expected))

    def get(self, descriptor, destination, io):
        descriptor=_bound(descriptor);destination=Path(destination)
        root=Path(self.args["job_root"]).resolve()
        if not destination.resolve().is_relative_to(root) or not Path(descriptor["path"]).resolve().is_relative_to(root):
            raise ValueError("Exact output/producer metadata must remain inside this fresh phase root")
        if destination.exists():io.verify(dict(descriptor,path=str(destination)));return
        temporary=destination.with_name(destination.name+"."+uuid.uuid4().hex+".scp.partial")
        destination.parent.mkdir(parents=True,exist_ok=True)
        with self.scp_lane:
            self.call("verify",dict(descriptor=descriptor))
            self.guard()
            result=subprocess.run(["scp","-q","-o","BatchMode=yes","-o","StrictHostKeyChecking=yes","-o","ConnectTimeout=10",
                HOST+":"+descriptor["path"].replace("\\","/"),str(temporary)],capture_output=True,timeout=600)
            if result.returncode:raise RuntimeError("Exact output SCP failed: "+result.stderr[-1500:].decode(errors="replace"))
            self.guard();io.publish_file(temporary,destination,descriptor,before_attempt=self.guard)


def _source_scope(args):
    return dict(host="Olav",stage=args["stage"],code_lock=args["lock_descriptor"],
                assignment=args["assignment_descriptor"],specs=args["specs"])


def _live_status(args):
    rows={}
    for name,path in args["capacity"]["status_paths"].items():
        with Path(path).open("rb") as stream:raw=stream.read(2*1024**2+1)
        if len(raw)>2*1024**2:raise ValueError("Original status DATA exceeds2MiB")
        value=_strict(raw);rows[name]=dict(path=path,raw_utf8_base64=base64.b64encode(raw).decode(),
                                        sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))
        if name=="claude" and (value["label_check"]["state"]!="PASS"
                or value["label_check"]["manifest_sha256"]!=args["capacity"]["phase_pass_sha256"]
                or any(any(row.get(k)=="CONCERN" for k in ("review","verdict","state"))
                       for row in value.get("technical_fix_review",[]))):
            raise PermissionError("Current phase PASS or CONCERN guard denies round4")
        for field,expected in args["capacity"]["live_expectations"].get(name,{}).items():
            actual=value
            for key in field.split("."):actual=actual[key]
            if isinstance(expected,list):
                if not expected or any(v in {"STOP","CONCERN","FAIL","FAILED","DENIED"} for v in expected) or actual not in expected:raise PermissionError("Current actual Root state denies phase")
            elif actual!=expected:raise PermissionError("Actual Root authority changed")
    if set(rows)!={"claude","codex"}:raise ValueError("Both original actual status DATA documents are required")
    return rows


def serve(root_request_descriptor, *, main_service_root, io, jobs, olav_operations, before_action):
    """One explicit Root servicing call; it never launches a native process.

    Start this alongside the port's phase(root_request_descriptor). The request
    is the ONE actual Root phase request, not another approval or review chain.
    Actual source/PASS/resource/decision checks remain in before_action and in
    the original port. New post05 Claude decisions must be denied by Root.
    """
    if not callable(before_action):raise PermissionError("Actual fresh Root guard required")
    bound=_bound(root_request_descriptor);before_action()
    request=_metadata(io,bound);args=request["build_arguments"]
    if request.get("contract")!="ffac-round4-olav-phase-request-v1" or request.get("complete") is not True or request.get("host")!="Olav":raise ValueError("Actual Root Olav round4 request required")
    scope=_source_scope(args);cap=args["capacity"];remote_root=Path(args["job_root"]).resolve()
    if scope["specs"]["v4"]["sha256"]!=SPEC or scope["specs"]["urest_amendment"]["sha256"]!=AMENDMENT:raise ValueError("Exactv4/amendment1 required")
    if args["stage"] not in {"extraction","heldout","loro","production","provisional"} or cap["backend"]!="cpu":raise ValueError("Only actual locked CPU round4 phase allowed")
    for name in ("incoming_root","incoming_mapping","root_status_control","publication_spool"):
        if Path(cap["transport"][name]).resolve().is_relative_to(remote_root):
            raise ValueError("Transport controls must be separate from the builder's strict fresh job root")
    lock=_metadata(io,args["lock_descriptor"])
    qualified=lock.get("qualified_resources",{}).get("Olav/cpu/float64",{})
    if lock.get("complete") is not True or lock.get("precision")!="float64" or qualified.get("passed") is not True:raise ValueError("Actual completed CPU64 PartA qualification/ONElock required")
    io.verify(_bound(qualified["equivalence_result"]))
    if jobs.code_identity(cap["Main_worktree"])["commit"]!=lock["commit"] or jobs.environment_lock()!=lock["environments_by_host"]["Main"]:raise ValueError("Actual Main source/environment differs from ONElock")
    assignment=_metadata(io,args["assignment_descriptor"])
    if assignment.get("format")!="darsia-production-static-whole-run-assignment-v1" or assignment.get("assignment_only") is not True or assignment.get("production_permission_granted") is not False:raise ValueError("ONE global non-GO whole-run assignment required")
    owners={row["run"]:row for row in assignment["assignments"]}
    if len(owners)!=39 or len(owners)!=len(assignment["assignments"]):raise ValueError("Exactly39 distinct whole-run owners required")
    current=io.asset(olav_operations.__file__)
    if (current["sha256"],current["size_bytes"])!=(args["sourcepins"]["operations"]["sha256"],args["sourcepins"]["operations"]["size_bytes"]):raise ValueError("Actual Main publication port source differs from locked remote port")
    normalize=__import__(io.__class__.__module__,fromlist=["normalize_locator"]).normalize_locator
    self_asset=io.asset(__file__)
    if not any(normalize(d.get("path",""))==normalize(self_asset["path"]) and d.get("sha256")==self_asset["sha256"] and d.get("size_bytes")==self_asset["size_bytes"] for d in lock["scientific_sources_by_host"]["Main"].values()):
        raise ValueError("Actual Main servicing source must be part of the ONElock")
    service=Path(main_service_root).resolve()
    if not service.is_relative_to(Path("C:/FFAC_cache").resolve()) or service.exists():raise ValueError("Fresh owned Main service root required")
    if not remote_root.is_relative_to(Path("C:/FFAC_cache").resolve()) or remote_root.exists():raise ValueError("Fresh same-locator Main mirror of the actual Olav job required")
    service.mkdir(parents=True);remote_root.mkdir(parents=True)
    identity=jobs.worker_identity();io.write_json(service/"owner.json",dict(root_request=bound,actual_Main_owner=identity))
    errors=[]
    def guard():
        before_action()
        if jobs.worker_identity()!=identity:raise RuntimeError("Actual Main service incarnation changed")
        if errors:raise errors[0]
    transport=_SSH(args,guard);mapping=dict(contract="ffac-round4-olav-SCP-input-map-v1",complete=True,scope=scope,arrays_decoded=False,inputs={})
    transport.call("install_control",dict(path=cap["transport"]["incoming_mapping"],document=mapping))
    inventory=_metadata(io,args["selected_inventory"])
    allowed={};run_assets={};common=set()
    def permit(asset,runs=()):
        d=_bound(asset);key=normalize(d["path"])
        if key in allowed and (allowed[key]["sha256"],allowed[key]["size_bytes"])!=(d["sha256"],d["size_bytes"]):raise ValueError("Conflicting original flat content identity")
        allowed[key]=d;run_assets.setdefault(key,set()).update(runs)
    for i,row in enumerate(inventory["assets"]):
        permit(row["descriptor"],row["working_runs"])
        if i in inventory["common_roles"]:common.add(normalize(row["descriptor"]["path"]))
    for d in inventory["logical_aliases"].values():permit(d);common.add(normalize(d["path"]))
    for d in [args["lock_descriptor"],args["selected_inventory"],args["template_packet"],args["assignment_descriptor"],*args["specs"].values()]:permit(d);common.add(normalize(d["path"]))
    # Explicit phase metadata (freeze/folds/statistics) only; no inferred SHA alias.
    def declared(value):
        if isinstance(value,dict):
            if set(value)=={"path","sha256","size_bytes"}:permit(value);return
            for v in value.values():declared(v)
        elif isinstance(value,list):
            for v in value:declared(v)
    for key in ("sourcepins","freeze","folds","statistics","physical_correction","computed_checks","provisional_delegation"):
        declared(args.get(key))
    old=_metadata(io,args["template_packet"])
    permit(old["boxes"]);common.add(normalize(old["boxes"]["path"]))
    if args.get("freeze"):
        freeze=_metadata(io,args["freeze"])
        declared(freeze["calibration_files"])
    ready=None;packet=None;stop=threading.Event();done=set();mapping_lock=threading.RLock();state_lock=threading.Lock()
    observed={};incoming={}
    def accept_ready(row,facts,finished=None):
        value=row["value"]
        if (value.get("contract")!="ffac-round4-olav-phase-ready-v1" or value.get("complete") is not True
                or value.get("root_request")!=bound or value.get("scope")!=scope or value.get("sources")!=args["sourcepins"]
                or value.get("scientific_arrays_read") is not False or value.get("numerical_imports_started") is not False):raise ValueError("Actual phase identity/source handshake differs")
        if facts.get("alive") is not True:
            if (ready!=value or finished is None or finished.get("complete") is not True
                    or finished.get("root_request")!=bound or finished.get("actual_phase_owner")!=value["actual_phase_owner"]
                    or finished.get("result",{}).get("complete") is not True):raise ValueError("Actual phase retired without exact completion")
            return value
        if facts["identity"]!=value["actual_phase_owner"]:raise ValueError("Actual phase incarnation differs")
        argv=facts["argv"];expected=[cap["python"]["path"],"-I","-B",args["sourcepins"]["operations"]["path"],"phase","--packet",bound["path"],"--sha256",bound["sha256"],"--size",str(bound["size_bytes"])]
        if len(argv)!=len(expected) or any(normalize(a)!=normalize(b) if i in (0,3,6) else a!=b for i,(a,b) in enumerate(zip(argv,expected))):raise ValueError("Actual phase command differs")
        return value
    def snapshot(continuing=True):
        guard()
        with state_lock:phase=ready;actual_packet=packet
        if phase is None:return
        poll=transport.call("poll",{})
        if poll.get("finished"):return
        accept_ready(poll["ready"],poll["phase_facts"])
        doc=dict(contract="ffac-round4-olav-Root-status-snapshot-v1",complete=True,Root_authorized=True,continue_=continuing,
            scope=scope,root_request=bound,phase_owner=phase["actual_phase_owner"],issued_epoch=time.time(),
            coordinator_packet_sha256=None if actual_packet is None else actual_packet["descriptor"]["sha256"],documents=_live_status(args))
        doc["continue"]=doc.pop("continue_")
        transport.call("install_control",dict(path=cap["transport"]["root_status_control"],document=doc))
    def controls():
        try:
            while not stop.wait(10):snapshot()
        except BaseException as error:errors.append(error);stop.set()
    thread=threading.Thread(target=controls,name="actual-Root-status-byte-forwarder",daemon=False);thread.start()
    try:
        while True:
            guard()
            if errors:raise errors[0]
            poll=transport.call("poll",{})
            if poll.get("failed"):raise RuntimeError("Actual Olav phase failed: "+str(poll["failed"]["value"].get("error")))
            if poll["ready"]:
                value=accept_ready(poll["ready"],poll["phase_facts"],None if not poll.get("finished") else poll["finished"]["value"])
                if ready is None:
                    with state_lock:ready=value
                    snapshot()
                elif value!=ready:raise ValueError("Actual phase ready record changed")
            elif ready is not None:raise ValueError("Actual phase ready disappeared")
            if poll["packet_ready"]:
                value=poll["packet_ready"]["value"]
                if value.get("root_request")!=bound or value.get("scope")!=scope or value.get("actual_phase_owner")!=ready["actual_phase_owner"]:raise ValueError("Actual coordinator packet binding differs")
                descriptor=_bound(value["packet"])
                if packet is None:
                    transport.get(descriptor,descriptor["path"],io);body=_metadata(io,descriptor)
                    if body.get("code_lock")!=args["lock_descriptor"] or body.get("original_assignment")!=args["assignment_descriptor"] or body.get("sources")!=args["sourcepins"]:raise ValueError("Actual coordinator source/assignment differs")
                    transport.get(_bound(body["assignment"]),body["assignment"]["path"],io)
                    with state_lock:packet=dict(descriptor=descriptor,value=body)
                    snapshot()
            for native in poll["natives"]:
                value=native["value"];facts=native["facts"]
                if facts.get("alive"):
                    if (value.get("contract")!="ffac-round4-native-ready-v1" or value.get("scientific_arrays_read") is not False
                            or value.get("numerical_imports_started") is not False or value.get("sources")!=args["sourcepins"]
                            or ready["actual_phase_owner"] not in facts["parents"]
                            or facts.get("identity")!=value["actual_native_worker"]):raise ValueError("Actual native source/ancestry handshake differs")
                    wp=native["worker_packet"]
                    if wp is None or value.get("packet")!=wp["descriptor"] or wp["value"].get("sources")!=args["sourcepins"] or wp["value"].get("code_lock")!=args["lock_descriptor"]:
                        raise ValueError("Actual native worker packet/source lock differs")
                    expected=[cap["python"]["path"],"-I","-B",args["sourcepins"]["operations"]["path"],"worker","--packet",wp["descriptor"]["path"],"--sha256",wp["descriptor"]["sha256"],"--size",str(wp["descriptor"]["size_bytes"]),"--run",value["run"]]
                    if len(facts["argv"])!=len(expected) or any(normalize(a)!=normalize(b) if i in (0,3,6) else a!=b for i,(a,b) in enumerate(zip(facts["argv"],expected))):raise ValueError("Actual native worker command differs")
                    if owners[value["run"]]["host"]!="Olav":raise ValueError("Native worker differs from global whole-run owner")
                    observed[value["run"]]=value["actual_native_worker"]
            # Output first, preserving the original one-deep publication bound.
            rows=sorted(poll["requests"],key=lambda r:r["value"].get("contract")=="ffac-round4-olav-SCP-input-request-v1")
            current_next=set()
            if packet is not None:
                order=_metadata(io,packet["value"]["assignment"])
                # The actual port packet's phase assignment is itself mirrored
                # below when first received; no allocation is recomputed here.
                for slot in {r["worker"] for r in order["assignments"] if r["host"]=="Olav"}:
                    remaining=[r["run"] for r in order["assignments"] if r["host"]=="Olav" and r["worker"]==slot and r["run"] not in poll["completed_runs"]]
                    current_next.update(remaining[:2])
            for row in rows:
                d=row["descriptor"];value=row["value"];key=d["path"]
                if key in done:continue
                if value.get("scope")!=scope:raise ValueError("Request escaped its actual phase scope")
                if value.get("contract")=="ffac-round4-olav-SCP-input-request-v1":
                    logical=_bound(value["logical"]);k=normalize(logical["path"])
                    if value.get("root_request")!=bound or value.get("arrays_decoded") is not False or ready is None or value["requester"] not in [ready["actual_phase_owner"],*observed.values()]:raise ValueError("No arbitrary requester or numerical input request accepted")
                    if k not in allowed or (logical["sha256"],logical["size_bytes"])!=(allowed[k]["sha256"],allowed[k]["size_bytes"]):raise ValueError("Input is outside the original flat declarations")
                    if k not in common and run_assets[k] and not any(owners[r]["host"]=="Olav" for r in run_assets[k]):raise ValueError("Another host's run cannot be staged")
                    if k not in common and run_assets[k] and not run_assets[k].intersection(current_next):raise ValueError("Only current and one-next whole-run working inputs may be copied")
                    before=io.retry.run("stat-Main-original",logical["path"],lambda:_stamp(logical["path"]),before_attempt=guard)
                    local=service/"incoming"/logical["sha256"]/hashlib.sha256(k.encode()).hexdigest()[:16]/Path(logical["path"]).name
                    copied=io.copy_file(logical,local,before_attempt=guard)
                    after=io.retry.run("stat-Main-original-after-copy",logical["path"],lambda:_stamp(logical["path"]),before_attempt=guard)
                    if before!=after:raise ValueError("Original frozen byte source changed across its one verified copy")
                    target=str(Path(cap["transport"]["incoming_root"])/logical["sha256"]/hashlib.sha256(k.encode()).hexdigest()[:16]/Path(logical["path"]).name)
                    actual=transport.put(copied["path"],target,logical)
                    with mapping_lock:
                        mapping["inputs"][k]=dict(logical=logical,local={n:actual[n] for n in ("path","sha256","size_bytes")},original_source_identity=after)
                        transport.call("install_control",dict(path=cap["transport"]["incoming_mapping"],document=mapping))
                    incoming[k]=dict(local=copied,local_identity=_stamp(copied["path"]),original=logical,identity=after,remote=actual)
                    transport.call("install_control",dict(path=str(Path(key).with_name(Path(key).name.replace(".request.json",".ack.json"))),
                        document=dict(complete=True,request=d,scope=scope,logical=logical,local=mapping["inputs"][k]["local"])))
                    done.add(key)
                elif value.get("contract")=="ffac-round4-olav-publication-request-v1":
                    run=value["run"]
                    if ready is None or owners.get(run,{}).get("host")!="Olav":raise ValueError("Another whole-run owner cannot publish")
                    worker_packet=_bound(value["packet"]);transport.get(worker_packet,worker_packet["path"],io)
                    body=_metadata(io,worker_packet);task=body["tasks"][run]
                    if value["manifest"]!=task["manifest"] or value["packet_sha256"]!=worker_packet["sha256"]:raise ValueError("Original worker manifest/packet differs")
                    for metadata in (value["manifest"],):transport.get(metadata,metadata["path"],io)
                    if value["kind"]=="frame":
                        registration=_bound(value["extra"]["registration"]);transport.get(registration,registration["path"],io)
                    local_directory=Path(value["local_directory"]).resolve()
                    if not local_directory.is_relative_to(remote_root):raise ValueError("Output mirror escaped its fresh actual producer root")
                    for relative,expected in value["files"].items():
                        target=local_directory/_relative(relative);descriptor=dict(expected,path=str(target));transport.get(descriptor,target,io)
                    request_path=Path(d["path"]);transport.get(d,request_path,io)
                    guard();ack=olav_operations.service_publication_on_Main(d,worker_packet,io=io,jobs=jobs,boundary_guard=guard)
                    if value["kind"]=="archive":
                        extra=value["extra"]
                        for relative,original in extra["original_inputs"].items():
                            actual=io.retry.run("stat-original-before-retired-cleanup",original["path"],lambda p=original["path"]:_stamp(p),before_attempt=guard)
                            if actual!=extra["original_input_identities"][relative]:raise ValueError("Original source changed after SHA-verified staging")
                        guard();ack["owned_cleanup"]=transport.call("cleanup",dict(extra=extra,files=value["files"],run=run))
                        # Only this run's exclusive fresh incoming copies can go;
                        # common/model assets and another current/next run remain.
                        for logical_key,item in list(incoming.items()):
                            if logical_key in common or run_assets.get(logical_key)!={run}:continue
                            guard()
                            if _stamp(item["local"]["path"])!=item["local_identity"]:raise ValueError("Main owned incoming copy changed before cleanup")
                            transport.call("forget_input",dict(descriptor=item["remote"],identity=item["remote"]["identity"],retired=extra["actual_retired_identities"]))
                            Path(item["local"]["path"]).unlink()
                            with mapping_lock:
                                mapping["inputs"].pop(logical_key);transport.call("install_control",dict(path=cap["transport"]["incoming_mapping"],document=mapping))
                            incoming.pop(logical_key)
                    ack_path=str(request_path.with_name(request_path.name.replace(".request.json",".ack.json")))
                    transport.call("install_control",dict(path=ack_path,document=ack));done.add(key)
                    print(json.dumps(dict(round4_Olav_publication=value["kind"],run=run,frame=value["case_id"],actual_NAS_verified=True)),flush=True)
                else:raise ValueError("No new or inferred transport request kind accepted")
            if poll["finished"]:
                value=poll["finished"]["value"]
                if (value.get("complete") is not True or value.get("root_request")!=bound or value["result"].get("complete") is not True
                        or packet is None or value.get("packet")!=packet["descriptor"]
                        or value.get("actual_phase_owner")!=ready["actual_phase_owner"]
                        or value["result"].get("owner")!=ready["actual_phase_owner"]):raise ValueError("Actual completed phase differs")
                retired=value["result"]["workers_retired"]
                if any(v not in retired for v in observed.values()):raise ValueError("An observed actual native was omitted from the retirement union")
                transport.call("retired",dict(identities=retired))
                result=dict(complete=True,root_request=bound,actual_Main_owner=identity,actual_Olav_phase_owner=ready["actual_phase_owner"],result=value["result"],actual_verified_publications=len(done),no_numerical_work_on_Main_service=True)
                io.write_json(service/"completed.json",result);return result
            time.sleep(.5)
    except BaseException as error:
        io.write_json(service/"failed.json",dict(complete=False,root_request=bound,error_type=type(error).__name__,error=str(error),original_stop_utc=jobs.utc_now(),preserved_originals_common_and_partials=True))
        raise
    finally:
        stop.set();thread.join()
