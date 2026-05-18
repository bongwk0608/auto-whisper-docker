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
.\scripts\init-env.ps1 -SourceDirs "D:\path\to\audio-folder"
```

Generate `.env` on Linux or macOS:

```sh
sh ./scripts/init-env.sh --source-dir /path/to/audio-folder
```

The helper writes both `.env` and `docker-compose.override.yml`. The override file contains local folder bind mounts and is ignored by Git.

## Windows WSL Docker Engine Setup

Use this path when Docker Engine is installed inside your WSL distro and you are not using Docker Desktop for Windows.

Run Docker commands from the WSL shell. For best performance during long transcriptions, keep this repo under the WSL filesystem, such as `~/projects/auto_whisper`, instead of running it from `/mnt/c` or `/mnt/d`.

Confirm Docker, Compose, and GPU access inside WSL:

```sh
docker info
docker compose version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Generate WSL-native `.env` and `docker-compose.override.yml` files from inside WSL. If your source media is still on Windows drive `D:`, use the `/mnt/d/...` path:

```sh
sh ./scripts/init-env.sh --source-dir /mnt/d/auto_whisper/test_input --cuda --model medium --output-format all
```

Validate and run the CUDA service:

```sh
docker compose --profile cuda config
docker compose --profile cuda build whisper-cuda
docker compose --profile cuda run --rm whisper-cuda python /app/download_model.py
docker compose --profile cuda up whisper-cuda
```

For CPU fallback, regenerate without `--cuda`, or set `WHISPER_DEVICE=cpu`, `WHISPER_FP16=false`, and `NVIDIA_VISIBLE_DEVICES=void`, then run:

```sh
docker compose up whisper
```

Keep `models/`, `output/`, and `state/` local to the WSL repo when possible. If you already downloaded models on Windows, you can copy the `.pt` files into the WSL repo's `models/` directory.

To configure multiple input folders, pass each source folder. All transcripts are written under this project folder's ignored `output/` directory:

```powershell
.\scripts\init-env.ps1 -SourceDirs "D:\audio-a,D:\audio-b"
```

```sh
sh ./scripts/init-env.sh --source-dir /audio-a --source-dir /audio-b
```

The source folders should normally live outside this repo. Transcripts are written only under this repo's `output/` folder by default. Advanced users can still pass `-OutputDirs` or `--output-dir` values if they want custom output roots.

You can also create `.env` manually:

```sh
cp .env.example.cpu .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example.cpu .env
```

Manual setup requires both `.env` and a Compose override that mounts each host source/output folder to the container paths named in `INPUT_OUTPUT_PAIRS`. The setup scripts are recommended because they generate both files.

```env
SOURCE_DIRS=D:/path/to/audio-folder
OUTPUT_DIRS=D:/auto_whisper/output
INPUT_OUTPUT_PAIRS=[{"input":"/inputs/input-001","output":"/outputs/output-001"}]
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
.\scripts\init-env.ps1 -SourceDirs "D:\path\to\audio-folder" -Cuda
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

For CUDA runs, keep the `whisper-cuda` service name in the command. Plain `docker compose up` starts the default CPU service and may build the CPU image.

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

Example files in the paired output folder:

```text
output/audio-folder_created-20260517143000/song_created-20260517143000_modified-20260517154512.txt
output/audio-folder_created-20260517143000/song_created-20260517143000_modified-20260517154512.json
output/audio-folder_created-20260517143000/song_created-20260517143000_modified-20260517154512.tsv
output/audio-folder_created-20260517143000/song_created-20260517143000_modified-20260517154512.srt
output/audio-folder_created-20260517143000/song_created-20260517143000_modified-20260517154512.vtt
```

For each source folder, transcript files are written into:

```text
output/<source_folder_name>_created-<YYYYMMDDHHMMSS>/
```

Relative subfolders are preserved in that output folder. Transcript sidecars are not written beside the original media files.

## Resume Behavior

Progress is stored in `state/progress.json`, which is ignored by Git.

If a run is interrupted, run Compose again. Completed files are skipped when the source fingerprint and expected outputs still match. Failed files are recorded and retried on the next run. Each input/output pair reuses its active timestamped output folder while a run is incomplete.

## Configuration

Important `.env` values:

- `SOURCE_DIRS`: semicolon-separated host folders containing audio/video files; generated by setup scripts
- `OUTPUT_DIRS`: semicolon-separated host transcript output folders; defaults to this project's `output/` directory for every source
- `INPUT_OUTPUT_PAIRS`: JSON array of container-side input/output pairs; generated by setup scripts
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

A matching template is included at `.wslconfig.example`. If you copy it to `$env:USERPROFILE\.wslconfig`, restart WSL. If you use Docker Desktop, restart Docker Desktop too:

```powershell
wsl --shutdown
```

When you use Docker Engine inside WSL without Docker Desktop, restart WSL and then start your WSL Docker service again:

```powershell
wsl --shutdown
```

```sh
sudo service docker start
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
- `docker-compose.override.yml`
- `output/`
- `state/`
- `test_input/`
- model caches such as `models/*.pt`
- source media files
- generated transcripts

Before pushing to GitHub, run:

```sh
git status --short
```

Confirm that no private files are staged. In particular, `.env`, `docker-compose.override.yml`, `output/`, `state/`, `test_input/`, `models/*.pt`, source media, and generated transcripts must not appear.

To verify ignore rules for specific paths:

```sh
git check-ignore .env docker-compose.override.yml output/ state/ test_input/ models/small.pt
```

If a confidential file was already committed in a previous repository history, removing it from the working tree is not enough. Rotate any exposed credentials and purge the file from Git history before publishing.
