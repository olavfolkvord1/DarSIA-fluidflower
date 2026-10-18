"""Memory-only pre-import application, never a host cache admission/profile."""
from types import SimpleNamespace
import json
import tempfile

import darsia_cache.bootstrap as bootstrap
from darsia_cache.environment import derive_cache_environment, WRITER_ENVIRONMENT_KEYS


def test_bootstrap_applies_all_child_writer_keys_and_preserves_parent(monkeypatch):
    token = "b" * 32
    source = dict(path="C:/TESTONLY/stores.toml", sha256="a" * 64, size_bytes=1)
    document = {"cache": dict(root="C:/TESTONLY_CACHE", quota_bytes=1,
                              profiles={"jobs": {"TESTONLY": "not a real profile"}})}
    parent = dict(CUPY_CACHE_DIR="old-cupy", TEMP="old-temp", HOME="durable-home",
                  LOCALAPPDATA="durable-state", OMP_NUM_THREADS="7",
                  UNRELATED_APPLICATION_KEY="preserved")
    original_parent = dict(parent)
    child = dict(parent)
    calls = []

    class Lease_TESTONLY:
        def __init__(self, actual_document, actual_source, package_file, *,
                     entry_kind, role_identifier):
            assert actual_document is document and actual_source is source
            assert package_file == "TESTONLY_PACKAGE"
            assert entry_kind == "worker" and role_identifier is None
            self.purpose = "jobs"
            self.reservation = dict(token=token)
            self.root = "C:/TESTONLY_CACHE/jobs/" + token
            self.receipt = dict(TESTONLY="not a real host/service receipt")
        def start(self):
            calls.append("start")
        def exit_intent(self):
            raise AssertionError("TESTONLY atexit registration must not execute")
        def close_pins(self):
            calls.append("close_pins")

    child_sys = SimpleNamespace(modules={}, pycache_prefix="old-prefix")
    monkeypatch.setattr(bootstrap, "_LEASE", None)
    monkeypatch.setattr(bootstrap, "os", SimpleNamespace(environ=child))
    monkeypatch.setattr(bootstrap, "sys", child_sys)
    monkeypatch.setattr(bootstrap, "_worker_arguments", lambda: ["TESTONLY"])
    monkeypatch.setattr(bootstrap, "_stores_argument", lambda: source["path"])
    monkeypatch.setattr(bootstrap, "_explicit_document", lambda *args: (document, source))
    monkeypatch.setattr(bootstrap, "_entry_kind", lambda: "worker")
    monkeypatch.setattr(bootstrap, "ImportLease", Lease_TESTONLY)
    monkeypatch.setattr(bootstrap, "atexit", SimpleNamespace(register=lambda fn: calls.append(fn)))
    monkeypatch.setattr(tempfile, "tempdir", "old-tempfile-choice")

    lease = bootstrap.before_darsia_import(package_file="TESTONLY_PACKAGE")
    expected = derive_cache_environment(document, purpose="jobs", token=token,
        scope_root=lease.root, parent_environment=parent)
    assert all(child[key] == expected[key] for key in WRITER_ENVIRONMENT_KEYS)
    assert child["CUPY_CACHE_DIR"].endswith(token + "\\cupy")
    assert child_sys.pycache_prefix == expected["PYTHONPYCACHEPREFIX"]
    assert json.loads(child[bootstrap.RECEIPT_ENV]) == lease.receipt
    assert all(child[key] == parent[key] for key in
               ("HOME", "LOCALAPPDATA", "OMP_NUM_THREADS", "UNRELATED_APPLICATION_KEY"))
    assert parent == original_parent
    assert tempfile.tempdir is None
    assert calls == ["start", lease.exit_intent]
    assert bootstrap.current_import_lease() is lease


def test_first_only_redirects_all_inherited_cache_application_paths_inside_owned_scope():
    from pathlib import PureWindowsPath
    from darsia_cache.environment import first_qualification_environment, FIRST_WRITER_ENVIRONMENT_KEYS
    token="b"*32;root="Q:/TESTONLY_CACHE/jobs/"+token
    document={"cache":dict(root="Q:/TESTONLY_CACHE",quota_bytes=1,profiles={"jobs":{}})}
    parent=dict(HOME="C:/old-home",LOCALAPPDATA="C:/old-local",XDG_CACHE_HOME="C:/old-xdg",
                USERPROFILE="C:/old-user",APPDATA="C:/old-app",UNRELATED="preserved")
    original=dict(parent)
    ordinary=derive_cache_environment(document,purpose="jobs",token=token,scope_root=root,parent_environment=parent)
    assert ordinary["HOME"]==parent["HOME"] and ordinary["LOCALAPPDATA"]==parent["LOCALAPPDATA"]
    actual=first_qualification_environment(ordinary,scope_root=root)
    assert parent==original and actual["UNRELATED"]=="preserved"
    for key in FIRST_WRITER_ENVIRONMENT_KEYS:
        if key!="NUMBA_CACHE_LOCATOR_CLASSES":
            assert PureWindowsPath(actual[key]).is_relative_to(PureWindowsPath(root))
    assert actual["NUMBA_CACHE_LOCATOR_CLASSES"]=="UserProvidedCacheLocator"



def explicit_stores_path_TESTONLY(monkeypatch,*,changed=None,declared_size=None,raw=None):
    data=b'[machine]\nid="TESTONLY"\n' if raw is None else raw
    fields=dict(st_dev=123,st_ino=456,st_size=len(data),st_mtime_ns=789,
        st_atime=1.0,st_atime_ns=1_000_000_000)
    if declared_size is not None:fields['st_size']=declared_size
    before=SimpleNamespace(**fields);after=SimpleNamespace(**dict(fields,**(changed or {})))
    states=iter([before,after]);reads=[]
    class Path_TESTONLY:
        def stat(self):return next(states)
        def read_bytes(self):reads.append('read');return data
        def __str__(self):return 'C:/TESTONLY/stores.toml'
    path=Path_TESTONLY()
    monkeypatch.setattr(bootstrap,'_outside',lambda selected,package:path)
    return data,reads


def test_explicit_stores_read_ignores_access_time_only_with_exact_raw_descriptor(monkeypatch):
    import hashlib
    raw,reads=explicit_stores_path_TESTONLY(monkeypatch,changed=dict(
        st_atime=9.0,st_atime_ns=9_000_000_000))
    document,source=bootstrap._explicit_document('C:/TESTONLY/stores.toml','TESTONLY_PACKAGE')
    assert document=={'machine':{'id':'TESTONLY'}} and reads==['read']
    assert source==dict(path='C:/TESTONLY/stores.toml',sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


def test_explicit_stores_read_still_refuses_every_identity_or_content_version_change(monkeypatch):
    import pytest
    from darsia_cache.policy import CachePolicyError
    for field,value in [('st_dev',124),('st_ino',457),('st_size',999),('st_mtime_ns',790)]:
        explicit_stores_path_TESTONLY(monkeypatch,changed={field:value})
        with pytest.raises(CachePolicyError,match='changed while reading'):
            bootstrap._explicit_document('C:/TESTONLY/stores.toml','TESTONLY_PACKAGE')
    raw,_=explicit_stores_path_TESTONLY(monkeypatch)
    explicit_stores_path_TESTONLY(monkeypatch,declared_size=len(raw)+1)
    with pytest.raises(CachePolicyError,match='changed while reading'):
        bootstrap._explicit_document('C:/TESTONLY/stores.toml','TESTONLY_PACKAGE')


def test_explicit_stores_read_preserves_positive_one_mib_before_read_bound(monkeypatch):
    import pytest
    from darsia_cache.policy import CachePolicyError
    for size in [0,1024**2+1]:
        _,reads=explicit_stores_path_TESTONLY(monkeypatch,declared_size=size)
        with pytest.raises(CachePolicyError,match='one MiB'):
            bootstrap._explicit_document('C:/TESTONLY/stores.toml','TESTONLY_PACKAGE')
        assert reads==[]
