"""Create a compact deliverable and verify credentials are excluded."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    members = set()
    for folder in ["food_delivery","scripts","tests",".github","input","notebooks","monitoring","evidence","artifacts"]:
        for path in (ROOT/folder).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith((".pyc",".db-wal",".db-shm")):
                if "/prefect/" in path.as_posix():
                    continue
                members.add(path)
    for path in (ROOT/"artifacts-tabpfn-gpu").glob("*"):
        if path.is_file() and path.suffix not in {".db",".safetensors"}:
            members.add(path)
    for path in ROOT.iterdir():
        if path.is_file() and (path.suffix in {".md",".txt",".toml",".yaml",".bundle"} or path.name in {"Dockerfile",".gitignore",".dockerignore"}):
            members.add(path)
    token_file = ROOT/".cache-tabpfn/token.local"
    secret = token_file.read_bytes().strip() if token_file.exists() else b""
    checksums = []
    target = ROOT/"Food_Delivery_CP413008.zip"
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(members):
            name = path.relative_to(ROOT).as_posix()
            payload = path.read_bytes()
            if secret and secret in payload:
                raise ValueError(f"Credential detected in {name}; packaging stopped")
            if any(part in {".cache-tabpfn",".venv-tabpfn-gpu","runtime_deps",".tmp"} for part in path.parts):
                raise ValueError("Private cache/runtime included")
            checksums.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
            archive.writestr(name,payload)
        content = "\n".join(checksums)+"\n"
        archive.writestr("CHECKSUMS.sha256",content)
        (ROOT/"CHECKSUMS.sha256").write_text(content,encoding="utf8")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        count = len(archive.namelist())
    print(json.dumps({"archive":str(target),"bytes":target.stat().st_size,"files":count,"integrity":"PASSED","credential_exclusion":"PASSED"}))


if __name__ == "__main__":
    main()
