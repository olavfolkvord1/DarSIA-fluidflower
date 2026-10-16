"""Operational startup with injected transport facts only; no Git/job/claim/run."""
import pytest
from darsia.fluidflower.pipeline import worker
from darsia.fluidflower import job_filesystem as filesystem


@pytest.mark.parametrize("facts", [
    filesystem.FilesystemFacts("unknown", "unknown fixture"),
    filesystem.FilesystemFacts("smb", "offline fixture", offline_cache=True),
    filesystem.FilesystemFacts("smb", "synchronized fixture", synchronized=True),
])
def test_startup_refuses_transport_before_reading_job_or_calling_git(monkeypatch,tmp_path,facts):
    seen=[]
    monkeypatch.setattr(worker,"require_unc",lambda path:seen.append("original UNC spelling check"))
    monkeypatch.setattr(filesystem,"inspect_shared_folder",lambda path:filesystem.classify_shared_folder(path,facts))
    monkeypatch.setattr(worker,"read_json",lambda *_:pytest.fail("Job must not be read on transport refusal"))
    monkeypatch.setattr(worker.subprocess,"check_output",lambda *_a,**_k:pytest.fail("Git must not be queried on transport refusal"))
    with pytest.raises(ValueError,match="Shared queue transport refused"):
        worker.verify_startup(tmp_path)
    assert seen==["original UNC spelling check"]


def test_positive_explicit_transport_continues_original_startup_without_claim_or_new_api(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(worker,"require_unc",lambda path:seen.append("unc"))
    def transport(path):
        seen.append("actual transport admission")
        return filesystem.classify_shared_folder(path,filesystem.FilesystemFacts("nfs","positive injected OS facts"))
    monkeypatch.setattr(filesystem,"inspect_shared_folder",transport)
    def original_job_reader(path):
        seen.append("original job read")
        raise RuntimeError("original reader sentinel; no actual job")
    monkeypatch.setattr(worker,"read_json",original_job_reader)
    with pytest.raises(RuntimeError,match="original reader sentinel"):
        worker.verify_startup(tmp_path)
    assert seen==["unc","actual transport admission","original job read"]
