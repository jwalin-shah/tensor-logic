from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_index_json_is_gitignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    assert "tools/index.json" in gitignore
