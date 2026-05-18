# Auto Whisper Docker Batch Transcriber

Batch transcribe supported audio and video files with OpenAI Whisper from Docker. The default runtime is CPU-only and works across Windows, Linux, and macOS. NVIDIA CUDA is available as an optional Compose profile for Linux and Windows WSL2 users.

The GitHub repo should contain only code and documentation. Local media, transcripts, model files, state, and `.env` stay private.

## Platform Support

- Windows CPU: supported with Docker Desktop.
- Windows NVIDIA CUDA: supported with Docker Desktop, WSL2, and working Docker GPU access.
- Linux CPU: supported with Docker Engine or Docker Desktop.
- Linux NVIDIA CUDA: supported with NVIDIA Container Toolkit.
- macOS CPU: supported with Docker Desktop.
- macOS GPU through Docker/CUDA: not supported.

## Prerequisites

- Docker with Compose v2
- Enough disk space for Whisper model files and generated transcripts
- Optional for CUDA: NVIDIA driver plus Docker GPU support
- Optional for setup scripts: PowerShell on Windows, POSIX shell on Linux/macOS

## Fresh Clone Setup

Start Docker first, then clone the repo and enter the project folder.

Generate `.env` on Windows:

```powershell
.\scripts\init-env.ps1 -SourceDir "D:\path\to\audio-folder"
```

Generate `.env` on Linux or macOS:

```sh
sh ./scripts/init-env.sh --source-dir /path/to/audio-folder
```

The source folder should normally live outside this repo. The transcriber writes sidecar transcript files beside the media files, so keeping media outside the repo helps prevent accidental commits.

You can also create `.env` manually:

```sh
cp .env.example.cpu .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example.cpu .env
```

Set `SOURCE_DIR` for your OS:

```env
SOURCE_DIR=D:/path/to/audio-folder
```

```env
SOURCE_DIR=/home/user/audio-folder
```

```env
SOURCE_DIR=/Users/user/audio-folder
```

## CPU Workflow

CPU mode is the default and is the recommended first run on every OS.

Validate Compose:

```sh
docker compose config
```

Build the CPU image:

```sh
docker compose build whisper
```

Download the configured Whisper model:

```sh
docker compose run --rm whisper python /app/download_model.py
```

Run transcription:

```sh
docker compose up
```

To rebuild and run:

```sh
docker compose up --build
```

Confirm the container is CPU-only:

```sh
docker compose run --rm whisper python -c "import torch; print(torch.cuda.is_available())"
```

Expected output is `False`.

## CUDA Workflow

CUDA is only for NVIDIA GPU users on Linux or Windows WSL2. Use CPU mode on macOS.

Create a CUDA `.env` from the example:

```sh
cp .env.example.cuda .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example.cuda .env
```

Or generate it with a helper:

```powershell
.\scripts\init-env.ps1 -SourceDir "D:\path\to\audio-folder" -Cuda
```

```sh
sh ./scripts/init-env.sh --source-dir /path/to/audio-folder --cuda
```

Check host GPU access:

```sh
nvidia-smi
```

Check Docker GPU access:

```sh
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Validate the CUDA Compose profile:

```sh
docker compose --profile cuda config
```

Build the CUDA image:

```sh
docker compose --profile cuda build whisper-cuda
```

Download the configured model:

```sh
docker compose --profile cuda run --rm whisper-cuda python /app/download_model.py
```

Run transcription with CUDA:

```sh
docker compose --profile cuda up whisper-cuda
```

To rebuild and run:

```sh
docker compose --profile cuda up --build whisper-cuda
```

## Output

For each source file, `WHISPER_OUTPUT_FORMAT=all` writes:

- `.txt`
- `.json`
- `.tsv`
- `.srt`
- `.vtt`

Example sidecars beside the source file:

```text
song.mp3
song_created-20260517143000_modified-20260517154512.txt
song_created-20260517143000_modified-20260517154512.json
song_created-20260517143000_modified-20260517154512.tsv
song_created-20260517143000_modified-20260517154512.srt
song_created-20260517143000_modified-20260517154512.vtt
```

The same transcript files are copied into:

```text
output/<source_folder_name>_<YYYYMMDDHHMMSS>/
```

Relative subfolders are preserved in the project output.

## Resume Behavior

Progress is stored in `state/progress.json`, which is ignored by Git.

If a run is interrupted, run Compose again. Completed files are skipped when the source fingerprint and expected outputs still match. Failed files are recorded and retried on the next run. The active timestamped output folder is reused while a run is incomplete.

## Configuration

Important `.env` values:

- `SOURCE_DIR`: host folder containing audio/video files
- `WHISPER_MODEL`: Whisper model such as `base`, `small`, `medium`, or `large`
- `WHISPER_OUTPUT_FORMAT`: `txt`, `json`, `tsv`, `srt`, `vtt`, or `all`
- `WHISPER_LANGUAGE`: optional language code; leave blank for auto-detect
- `WHISPER_TASK`: `transcribe` or `translate`
- `WHISPER_DEVICE`: `cpu`, `auto`, or `cuda`
- `WHISPER_DOWNLOAD_DEVICE`: model download verification device, usually `cpu`
- `WHISPER_FP16`: `false` for CPU, `auto` for CUDA, or explicit `true`/`false`
- `WHISPER_CONDITION_ON_PREVIOUS_TEXT`: `true` or `false`
- `WHISPER_VERBOSE`: `true` or `false`
- `SUPPORTED_EXTENSIONS`: comma-separated file extensions to scan

Recommended defaults:

- CPU: `WHISPER_DEVICE=cpu`, `WHISPER_FP16=false`, `NVIDIA_VISIBLE_DEVICES=void`
- CUDA: `WHISPER_DEVICE=auto`, `WHISPER_FP16=auto`, `NVIDIA_VISIBLE_DEVICES=all`

## Resource Checks

All platforms:

```sh
docker info
docker compose config
uname -m
```

Windows WSL resource audit:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_resources.ps1
```

If the image is not built yet, or you only want host checks:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_resources.ps1 -SkipContainerChecks
```

Recommended WSL baseline for longer Windows runs:

```ini
[wsl2]
memory=12GB
processors=6
swap=24GB
localhostForwarding=true
```

A matching template is included at `.wslconfig.example`. If you copy it to `$env:USERPROFILE\.wslconfig`, restart WSL and Docker Desktop:

```powershell
wsl --shutdown
```

## Troubleshooting

If Docker cannot access the GPU, use CPU mode:

```env
WHISPER_DEVICE=cpu
WHISPER_FP16=false
NVIDIA_VISIBLE_DEVICES=void
```

If CUDA runs out of memory during transcription, first try:

```env
WHISPER_MODEL=base
```

If it still fails, use CPU mode. CPU mode is slower but has the broadest platform compatibility.

On Apple Silicon Macs, Docker runs Linux ARM containers. Use the CPU image and avoid the CUDA profile.

## GitHub Safety

Audio, video, transcripts, and progress state may contain confidential speech, names, file paths, or project details. Treat them as private local artifacts.

Safe to commit:

- `Dockerfile`
- `Dockerfile.cpu`
- `Dockerfile.cuda`
- `docker-compose.yml`
- `scripts/`
- `.env.example`
- `.env.example.cpu`
- `.env.example.cuda`
- `.gitignore`
- `.dockerignore`
- `README.md`
- `.github/workflows/ci.yml`
- `.wslconfig.example`
- `models/.gitkeep`

Do not commit:

- `.env`
- `output/`
- `state/`
- `test_input/`
- model caches such as `models/*.pt`
- source media files
- generated transcript sidecars

Before pushing to GitHub, run:

```sh
git status --short
```

Confirm that no private files are staged. In particular, `.env`, `output/`, `state/`, `test_input/`, `models/*.pt`, source media, and generated transcripts must not appear.

To verify ignore rules for specific paths:

```sh
git check-ignore .env output/ state/ test_input/ models/small.pt
```

If a confidential file was already committed in a previous repository history, removing it from the working tree is not enough. Rotate any exposed credentials and purge the file from Git history before publishing.
