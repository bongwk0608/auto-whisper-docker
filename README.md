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

Create an editable input folder list:

```powershell
Copy-Item input-folders.example.txt input-folders.txt
notepad input-folders.txt
```

Generate `.env` on Windows:

```powershell
.\scripts\init-env.ps1 -SourceListFile ".\input-folders.txt"
```

Generate `.env` on Linux or macOS:

```sh
cp input-folders.example.txt input-folders.txt
${EDITOR:-vi} input-folders.txt
sh ./scripts/init-env.sh --source-list-file ./input-folders.txt
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

If your input folders are on mapped Windows network drives such as `X:`, `Y:`, or `Z:`, mount them inside WSL before generating `.env`:

```sh
sh ./scripts/mount-wsl-network-drives.sh
```

The helper reads the current Windows mappings from `net.exe use` and mounts connected drives at `/mnt/x`, `/mnt/y`, and `/mnt/z`. If a drive is only available while connected to VPN, connect the VPN first, then rerun the helper.

To mount those drives automatically when WSL starts and retry after VPN or internet reconnects, install the systemd timer from WSL:

```sh
sh ./scripts/install-wsl-network-drive-automount.sh
```

The timer runs once shortly after WSL starts and then retries every minute. It remounts stale drives when `/mnt/x`, `/mnt/y`, or `/mnt/z` exists but the network share is no longer reachable. If systemd is not enabled in WSL, add this to `/etc/wsl.conf`, then run `wsl --shutdown` from Windows PowerShell and start WSL again:

```ini
[boot]
systemd=true
```

For selected drives only:

```sh
sh ./scripts/install-wsl-network-drive-automount.sh "Y Z"
```

Generate WSL-native `.env` and `docker-compose.override.yml` files from inside WSL. If your source media is still on Windows drive `D:`, use the `/mnt/d/...` path:

```sh
cp input-folders.example.txt input-folders.txt
printf '%s\n' '/mnt/d/auto_whisper/test_input' > input-folders.txt
sh ./scripts/init-env.sh --source-list-file ./input-folders.txt --cuda --model medium --output-format all
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

## Adding Or Changing Input Folders

Add or change folders by editing `input-folders.txt`, then rerunning the setup helper. The helper updates both `.env` and `docker-compose.override.yml`; keep those files in sync because `.env` names the input/output pairs and the override file creates the Docker bind mounts.

The list file uses one folder path per line. Blank lines and lines starting with `#` are ignored. `input-folders.txt` is ignored by Git so local private paths stay out of commits.

From Windows PowerShell with Docker Desktop, use Windows paths:

```powershell
Copy-Item input-folders.example.txt input-folders.txt
notepad input-folders.txt
.\scripts\init-env.ps1 -SourceListFile ".\input-folders.txt"
```

From WSL, use WSL paths. Windows drive `D:\...` becomes `/mnt/d/...`, and `C:\Users\USER\Downloads` becomes `/mnt/c/Users/USER/Downloads`:

```sh
sh ./scripts/init-env.sh \
  --source-list-file ./input-folders.txt \
  --model medium \
  --output-format all \
  --cuda
```

The source folders should normally live outside this repo. Transcripts are written only under this repo's `output/` folder by default. Advanced users can still pass `-SourceDirs` / `--source-dir` directly, and can pass `-OutputDirs` or `--output-dir` values if they want custom output roots.

Validate the generated Compose configuration after changing folders:

```sh
docker compose --profile cuda config
```

For CPU-only setup, omit `--cuda` and validate with:

```sh
docker compose config
```

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

### CUDA Command Cheat Sheet

You do not need to run every CUDA command every time.

Run this after Dockerfile or dependency changes:

```sh
docker compose --profile cuda build whisper-cuda
```

Run this the first time you use a model, or after changing `WHISPER_MODEL`:

```sh
docker compose --profile cuda run --rm whisper-cuda python /app/download_model.py
```

Run this for normal transcription launches:

```sh
docker compose --profile cuda up whisper-cuda
```

Changing input folders in `.env` and `docker-compose.override.yml` does not require a rebuild. Run the setup helper again, validate with `docker compose --profile cuda config`, then run `docker compose --profile cuda up whisper-cuda`.

## Output

For each source file, `WHISPER_OUTPUT_FORMAT=all` writes:

- `.txt`
- `.json`
- `.tsv`
- `.srt`
- `.vtt`

Example files in the paired output folder:

```text
output/audio-folder_created-20260517143000_modified-20260517154512/song_created-20260517143000_modified-20260517154512.txt
output/audio-folder_created-20260517143000_modified-20260517154512/song_created-20260517143000_modified-20260517154512.json
output/audio-folder_created-20260517143000_modified-20260517154512/song_created-20260517143000_modified-20260517154512.tsv
output/audio-folder_created-20260517143000_modified-20260517154512/song_created-20260517143000_modified-20260517154512.srt
output/audio-folder_created-20260517143000_modified-20260517154512/song_created-20260517143000_modified-20260517154512.vtt
```

For each source folder, transcript files are written into:

```text
output/<source_folder_name>_created-<YYYYMMDDHHMMSS>_modified-<YYYYMMDDHHMMSS>/
```

Relative subfolders are preserved in that output folder. For example, a source file at `course/week1/audio.mp3` writes transcript files under `course/week1/` inside that source folder's timestamped output folder. Transcript sidecars are not written beside the original media files.

The output root also contains mapping manifests that show which input folder maps to which timestamped output folder:

```text
output/input-output-mapping.json
output/input-output-mapping.csv
```

The mapping includes each pair id, host input path when available, container input path, host output root when available, container output root, timestamped run output folder, input folder created/modified timestamps, output formats, recursive scan status, and the number of supported files found under that input folder.

When multiple input folders share the same output root, each input still gets its own timestamped subfolder. For example:

```text
/mnt/c/Users/USER/Downloads -> /mnt/d/auto_whisper/output/Downloads_created-<YYYYMMDDHHMMSS>_modified-<YYYYMMDDHHMMSS>/
/mnt/y/Class Recording/UM CS -> /mnt/d/auto_whisper/output/UM_CS_created-<YYYYMMDDHHMMSS>_modified-<YYYYMMDDHHMMSS>/
```

## Resume Behavior

Progress is stored in `state/progress.json`, which is ignored by Git.

If a run is interrupted, run Compose again. Completed files are skipped when the source fingerprint and expected outputs still match. Failed files are recorded and retried on the next run. Each input/output pair reuses its active timestamped output folder while a run is incomplete.

By default, `FINGERPRINT_MODE=metadata` uses file size and modified timestamp for fast skip checks, which is much faster for large network-drive folders. Set `FINGERPRINT_MODE=sha256` if you want the older maximum-safety behavior that reads each supported file fully before deciding whether to skip it.

`LOCAL_STAGING=false` by default, so pending media files are transcribed directly from their mounted input path. If network reads are unstable during transcription, set `LOCAL_STAGING=true`; Auto Whisper will copy only the current pending file to `LOCAL_STAGING_DIR`, transcribe that local temporary copy, and clean it up afterward. It does not cache or copy the whole input folder.

## Configuration

Important `.env` values:

- `input-folders.txt`: editable local source folder list used by `-SourceListFile` or `--source-list-file`; ignored by Git
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
- `FINGERPRINT_MODE`: `metadata` for fast network-drive skip checks, or `sha256` for full-file hashing
- `LOCAL_STAGING`: `true` to copy only pending files to local temporary storage before transcription
- `LOCAL_STAGING_DIR`: staging directory used when `LOCAL_STAGING=true`
- `SUPPORTED_EXTENSIONS`: comma-separated file extensions to scan

Recommended defaults:

- CPU: `WHISPER_DEVICE=cpu`, `WHISPER_FP16=false`, `FINGERPRINT_MODE=metadata`, `LOCAL_STAGING=false`, `NVIDIA_VISIBLE_DEVICES=void`
- CUDA: `WHISPER_DEVICE=auto`, `WHISPER_FP16=auto`, `FINGERPRINT_MODE=metadata`, `LOCAL_STAGING=false`, `NVIDIA_VISIBLE_DEVICES=all`

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
- `input-folders.txt`
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

Confirm that no private files are staged. In particular, `.env`, `input-folders.txt`, `docker-compose.override.yml`, `output/`, `state/`, `test_input/`, `models/*.pt`, source media, and generated transcripts must not appear.

To verify ignore rules for specific paths:

```sh
git check-ignore .env input-folders.txt docker-compose.override.yml output/ state/ test_input/ models/small.pt
```

If a confidential file was already committed in a previous repository history, removing it from the working tree is not enough. Rotate any exposed credentials and purge the file from Git history before publishing.
