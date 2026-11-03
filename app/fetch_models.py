"""Download the pinned base model and private round-2 checkpoint into this repo."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
BASE_ID = "Qwen/Qwen3.5-0.8B"
BASE_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CHECKPOINT_ID = "Harisundar/lays-r5ft-ckpt"
CHECKPOINT_REVISION = "30758a7cd5772ccb65de34ecb013166bb5007d6d"
RUN = "molab-round2-fast"
WEIGHT_SHA256 = "ffff0b07cb6d5bc17ea3f356c611dc0c257d1e3da3684e09bd7b533ac58776a1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    MODELS.mkdir(exist_ok=True)
    print(f"Downloading public base model {BASE_ID} (pinned revision)...", flush=True)
    snapshot_download(repo_id=BASE_ID, revision=BASE_REVISION, local_dir=MODELS / "base")

    print(f"Checking access to private checkpoint {CHECKPOINT_ID}...", flush=True)
    HfApi().model_info(CHECKPOINT_ID, revision=CHECKPOINT_REVISION)
    remote_dir = f"round2/{RUN}/"
    files = ["trainable.safetensors", "state.json", "MANIFEST.json", "TRAINING_MANIFEST.json", "README.md"]
    snapshot_download(
        repo_id=CHECKPOINT_ID,
        repo_type="model",
        revision=CHECKPOINT_REVISION,
        local_dir=MODELS,
        allow_patterns=[remote_dir + name for name in files],
    )

    checkpoint = MODELS / remote_dir
    actual = sha256(checkpoint / "trainable.safetensors")
    if actual != WEIGHT_SHA256:
        raise RuntimeError(f"Checkpoint hash mismatch: {actual}")
    print(f"Verified round-2 checkpoint SHA-256 {actual}", flush=True)
    print(json.dumps({"base_model": str(MODELS / "base"), "checkpoint": str(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
