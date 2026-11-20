"""Done-marker measurements are optional and belong to the accepted attempt."""
import pytest

from darsia.jobs import queue


@pytest.mark.parametrize("failed", [False, True])
@pytest.mark.parametrize("memory", [None, {"peak_commit_bytes": 123456}])
def test_terminal_marker_keeps_measurements_only_for_success(tmp_path, failed, memory):
    for state in queue.STATES:
        (tmp_path / state).mkdir()
    claim = queue.Claim(tmp_path, "step.run", "attempt-token", "machine")
    claim.path.write_text("{}\n")
    queue.finish(claim, ("adopted-token",), "failed" if failed else None, memory=memory)
    state = queue.FAILED if failed else queue.DONE
    marker = queue.read_marker(tmp_path / state / claim.path.name)
    assert marker["token"] == claim.token
    assert marker["machine"] == claim.machine
    assert not claim.path.exists()
    if failed:
        assert marker["traceback"] == "failed"
        assert "accepted" not in marker
        assert "memory" not in marker
    else:
        assert set(marker["accepted"]) == {"adopted-token", claim.token}
        if memory:
            assert marker["memory"] == memory
        else:
            assert "memory" not in marker


def test_revoked_attempt_cannot_record_memory(tmp_path):
    for state in queue.STATES:
        (tmp_path / state).mkdir()
    claim = queue.Claim(tmp_path, "step.run", "revoked-token", "machine")
    with pytest.raises(queue.Revoked):
        queue.finish(claim, memory={"peak_commit_bytes": 123456})
    assert not list((tmp_path / queue.DONE).iterdir())
