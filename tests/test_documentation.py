from pathlib import Path


def test_documentation_describes_current_architecture_only():
    paths = [
        Path("README.md"),
        Path("README_zh-CN.md"),
        Path("docs/MODEL_ARCHITECTURE.md"),
        Path("docs/REPRODUCTION.md"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    for stale in (
        "context refiner",
        "contextual refinement",
        "ordered point transformer",
        "ordered attention",
        "16d/19d/23d",
        "23d input",
    ):
        assert stale not in text
    assert "pointmlp" in text
    assert "12d" in text
