# LAYS Companion

Windows source repository for a local LAYS desktop assistant. The Git repo intentionally stays small: `Setup.cmd` downloads the pinned Qwen base model, the private round-2 checkpoint, Python packages, the desktop driver and Hex. Model weights and installed binaries are excluded from Git.

## One-command setup

1. Clone `https://github.com/HARISUNDARRAJENDRAN/lays_companion.git`.
2. Double-click **Setup.cmd** (or run `powershell -ExecutionPolicy Bypass -File .\setup.ps1`).
3. Setup installs Python 3.12 and pinned packages with CUDA 13.0 PyTorch, downloads the model assets, installs the native Windows driver and Hex, then checks CUDA and audio.
4. The round-2 weights are hosted in the private Hugging Face model `Harisundar/lays-r5ft-ckpt`. Sign in with an account that has access when Setup prompts; request repository access from its owner if needed. Setup verifies the downloaded weight SHA-256. The Qwen base model is public and pinned to the revision in `app/fetch_models.py`.
5. Complete Hex's microphone and speech-model onboarding when its window opens, then double-click **Start.cmd**.

Use **Start Dry Run.cmd** to see predictions without sending input. **Start Iris Xe.cmd** uses Intel Iris Xe Vulkan for speech; LAYS inference uses CUDA. Press Ctrl+Alt+Space to speak and F8 to stop. **Stop.cmd** stops the LAYS companion.

Setup needs internet and several gigabytes of free space. It installs Python in the project if needed, creates `.venv`, and downloads data into `models/`, `bin/`, and the user's Hex install. All generated and private data directories are ignored by Git.

## Repository contents

- `app/`: companion loop, Hex stream client, model downloader, desktop driver downloader, diagnostics.
- `harness/`: frozen C6 runtime and Computah grounding/safety wrapper.
- `setup.ps1`: dependency and asset installation.
- `requirements.lock.txt`: pinned Python dependency versions, including CUDA PyTorch.
- `launch.ps1`, `stop.ps1`, `Start*.cmd`: Windows controls.

The pipeline takes a hotkey command through local Hex speech recognition, then the resident round-2 LAYS model and C6 harness observe, ground, guard and execute desktop actions. C6's stop, freshness, delivery and repetition checks remain active. Hex uses local energy-based voice activity detection; this adapts Computah's turn flow without Deepgram Flux.

Physical input runs only in **Start.cmd**. Begin by focusing a harmless target app. A model completion prediction does not prove the task succeeded. See [VALIDATION.md](VALIDATION.md) for what was verified locally and what still needs a live microphone test.

## Git identity

The repository is configured locally with `user.name=HARISUNDARRAJENDRAN`. No email is stored in this repo so it cannot inherit an unrelated global Git identity. Add your GitHub email or noreply address before committing. Git LFS is not needed because large assets download during setup.
