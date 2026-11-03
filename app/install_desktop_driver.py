"""Fetch the pinned native desktop driver from its npm registry tarball."""
import base64
import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://registry.npmjs.org/agent-desktop/-/agent-desktop-1.0.2.tgz"
SHA512 = "k2Qi3f0y1U60YaJOw1dob3EO55P/DuH4HK3WKdI9sMZLjVxpGZAzuG50rIWuIqS0JIVy07g+0pp9Of2JDjTX0w=="
MEMBER = "package/bin/agent-desktop-win32-x64.exe"


def main():
    print("Downloading pinned agent-desktop 1.0.2...", flush=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = response.read()
    if base64.b64encode(hashlib.sha512(archive).digest()).decode() != SHA512:
        raise RuntimeError("npm package integrity check failed")
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        member = tar.getmember(MEMBER)
        if not member.isfile():
            raise RuntimeError("Expected Windows x64 executable missing from npm package")
        dest = ROOT / "bin" / "agent-desktop.exe"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tar.extractfile(member) as src, dest.open("wb") as out:
            while block := src.read(1024 * 1024):
                out.write(block)
    print(f"Installed {dest}", flush=True)


if __name__ == "__main__":
    main()
