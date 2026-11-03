# Local pre-release verification

The full local build was verified on Windows x64 with an RTX 3050 and Intel Iris Xe. Round-2 inference completed a C6 dry-run; no input was sent. Hex transcribed a deterministic test WAV on NVIDIA and Iris Vulkan. Lifecycle regressions passed, and managed start/stop reached readiness.

The GitHub repository omits the large downloaded runtime/model assets. `Setup.cmd` pins the Qwen base revision, downloads the private round-2 checkpoint from Hugging Face with a SHA-256 check, installs the CUDA-enabled Python environment and native driver, and installs Hex. A clean clone still needs a live Setup run and the user must complete Hex's microphone/model onboarding. This machine has not run Setup from an empty clone, and live microphone-to-desktop task success remains unverified.
