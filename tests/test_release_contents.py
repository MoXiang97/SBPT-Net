from pathlib import Path

from tools.verify_release import verify_release_tree


def test_release_tree_rejects_private_or_large_artifacts(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"checkpoint")
    errors = verify_release_tree(tmp_path)
    assert any("model.pt" in error for error in errors)


def test_repository_release_tree_is_clean():
    assert verify_release_tree(Path(".")) == []


def test_release_contains_only_final_model_code():
    assert not Path("baselines").exists()
    assert not Path("results").exists()
