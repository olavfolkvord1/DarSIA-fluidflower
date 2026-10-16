"""Fault injection for the two original cleanup sites; no real series or jobs.

Compile-only delivery. Default False must retain the original deletion behavior;
True preserves only the failed Store staging/unindexed artifact directory.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import geometry_facies as subject


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("site", ["canonical", "materialized_validation"])
def test_failure_cleanup_is_opt_in_and_never_publishes_current(tmp_path, monkeypatch, preserve, site):
    project = SimpleNamespace(root=tmp_path / "project", manifest={})
    depth = tmp_path / "original-depth.npz"
    depth.write_bytes(b"source bytes, copied without decode in this fault test")
    config = subject.default_geometry_config(project)
    config.update(depth_path=str(depth), labels_path=str(tmp_path / "labels.npz"))
    validation_calls = []

    def validate(value):
        validation_calls.append(value)
        errors = int(site == "materialized_validation" and len(validation_calls) == 2)
        return {"config_sha256": subject.geometry_config_digest(value),
                "summary": {"error_count": errors}}

    def canonical(_value):
        if site == "canonical":
            raise RuntimeError("exact canonical fault")
        return np.array([[0]], dtype=np.int32), []

    monkeypatch.setattr(subject, "validate_geometry_config", validate)
    monkeypatch.setattr(subject, "_canonical_facies", canonical)
    monkeypatch.setattr(subject, "_darsia_metadata", lambda _path: {})
    store = subject.GeometryRevisionStore(project)
    expected = "exact canonical fault" if site == "canonical" else "Materialized geometry revision failed validation"
    error_type = RuntimeError if site == "canonical" else ValueError
    with pytest.raises(error_type, match=expected):
        store.save_revision(config, actor="synthetic", reason="fault test",
                            preserve_failed_artifacts=preserve)
    children = list(store.artifact_root.iterdir())
    assert len(children) == int(preserve)
    if preserve:
        partial = children[0]
        assert (partial / "depth.npz").read_bytes() == depth.read_bytes()
        if site == "canonical":
            assert partial.name.startswith(".") and partial.name.endswith(".tmp")
        else:
            assert len(partial.name) == 32 and (partial / "facies.npz").is_file()
        assert not (partial / "revision.json").exists()
    assert not store.current_path.exists() and not store.revisions_path.exists()
    assert project.manifest == {} and depth.read_bytes().startswith(b"source bytes")


def test_omitted_option_keeps_original_failure_cleanup(tmp_path, monkeypatch):
    project = SimpleNamespace(root=tmp_path / "project", manifest={})
    depth = tmp_path / "depth.npz"
    depth.write_bytes(b"original")
    config = subject.default_geometry_config(project)
    config["depth_path"] = str(depth)
    monkeypatch.setattr(subject, "validate_geometry_config", lambda value: {
        "config_sha256": subject.geometry_config_digest(value), "summary": {"error_count": 0}})

    def fail(_value):
        raise RuntimeError("original default fault")

    monkeypatch.setattr(subject, "_canonical_facies", fail)
    store = subject.GeometryRevisionStore(project)
    with pytest.raises(RuntimeError, match="original default fault"):
        store.save_revision(config, actor="synthetic", reason="default")
    assert list(store.artifact_root.iterdir()) == []
    assert not store.current_path.exists() and not store.revisions_path.exists()
