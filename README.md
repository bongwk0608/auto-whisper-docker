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

## Windows Network Drive Auto-Mount

Use this path when Docker Desktop or other Windows apps need mapped network drive letters such as `X:`, `Y:`, and `Z:`.

Mount the default Auto Whisper network drives manually:

```powershell
.\scripts\mount-windows-network-drives.ps1
```

By default, the helper maps `X:` to `\\10.0.0.3\smb_backup`, `Y:` to `\\10.0.0.3\smb_documents`, and `Z:` to `\\10.0.0.3\smb_media`. To override the mappings:

```powershell
.\scripts\mount-windows-network-drives.ps1 -Mappings @{
  X = "\\10.0.0.3\smb_backup"
  Y = "\\10.0.0.3\smb_documents"
  Z = "\\10.0.0.3\smb_media"
}
```

The helper uses your existing Windows credentials. Sign in with a domain account that can access the shares, save credentials in Windows Credential Manager, or connect once through File Explorer before relying on unattended startup.

To inspect the current mapping state without changing anything:

```powershell
cd D:\auto_whisper
.\scripts\mount-windows-network-drives.ps1 -Diagnose
```

The diagnosis reports the current Windows user context, whether PowerShell is elevated, the Scheduled Task action, and each configured drive as `Ready`, `Missing`, `Unreachable`, or `Conflict`. If normal and Administrator PowerShell disagree, run `-Diagnose` in both sessions; Windows can keep separate mapped-drive state for those contexts.

If a configured drive letter is already mapped to the same server by hostname instead of IP, run a one-time repair to remove that conflicting mapping and recreate it with the configured IP path:

```powershell
cd D:\auto_whisper
.\scripts\mount-windows-network-drives.ps1 -RepairConflicts -RetryCount 1
net use Y:
```

For `Y:`, the remote name should be `\\10.0.0.3\smb_documents`.

If Windows still reports that the drive letter has a remembered connection, rerun the repair command from PowerShell as Administrator.

Install automatic mounting for Windows startup, user logon, and reconnect recovery:

```powershell
.\scripts\mount-windows-network-drives.ps1 -InstallTask
```

The installed task also runs every 1 minute with a quick check so mapped drives can recover after later Wi-Fi, LAN, VPN, or SMB reconnects. Healthy drives are skipped without remapping.

The scheduled task starts PowerShell hidden so the recovery check runs quietly in the background.

To use a different recovery interval:

```powershell
.\scripts\mount-windows-network-drives.ps1 -InstallTask -RepeatMinutes 5
```

If Windows refuses to register the startup trigger, rerun the install command from an elevated PowerShell session.

Remove the automatic mounting task:

```powershell
.\scripts\mount-windows-network-drives.ps1 -UninstallTask
```

Windows drive letters are scoped to user sessions. The logon trigger is what makes mapped drives visible to Explorer, Docker Desktop, and normal PowerShell sessions; the startup trigger is best-effort coverage while Windows is booting.

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

When setup scripts provide `SOURCE_DIRS`, `<source_folder_name>` comes from the real host input folder name, not the Docker mount alias. For example, `/mnt/c/Users/USER/Downloads` writes under `Downloads_created-..._modified-...`, and `/mnt/y/Class Recording/UM CS` writes under `UM_CS_created-..._modified-...`. If host paths are not available, Auto Whisper falls back to the container input folder name.

Relative subfolders are preserved in that output folder. For example, a source file at `course/week1/audio.mp3` writes transcript files under `course/week1/` inside that source folder's timestamped output folder. Transcript sidecars are not written beside the original media files.

Auto Whisper also writes transcript copies to a merged overall view:

```text
output_overall/pair-001/
output_overall/pair-002/
```

This view does not include the timestamped run folder layer. It accumulates transcript outputs by input pair id, preserving the same relative nested source paths. For example, `pair-002` source file `Dissertation Discussion/STEREO/FOLDER01/ZOOM0001.WAV` also writes copies under:

```text
output_overall/pair-002/Dissertation Discussion/STEREO/FOLDER01/ZOOM0001_created-<YYYYMMDDHHMMSS>_modified-<YYYYMMDDHHMMSS>.txt
```

If the same overall destination already exists, the newest generated transcript overwrites it. Original media files are not copied into `output_overall/`.

The output root also contains mapping manifests that show which input folder maps to which timestamped output folder:

```text
output/input-output-mapping.json
output/input-output-mapping.csv
```

The mapping includes each pair id, host input path when available, container input path, host output root when available, container output root, timestamped run output folder, overall output folder, input folder created/modified timestamps, output formats, recursive scan status, and the number of supported files found under that input folder.

## Optional Speaker Diarization

Speaker diarization is implemented as a separate optional post-processing service. The Whisper service owns transcription, and the diarization service owns speaker labeling. They communicate only through files, not shared runtime state.

```text
Existing Whisper service:
Audio -> Whisper transcription -> output/ -> output_overall/

New diarization service:
Whisper JSON + original audio -> pyannote diarization -> speaker merge/export -> output_pyannote/ -> output_pyannote_overall/
```

Whisper outputs in `output/` and `output_overall/` are immutable raw transcript artifacts. Diarization never edits those files. It reads Whisper `.json` files as the transcript source of truth and writes derived speaker artifacts only under:

```text
output_pyannote/
output_pyannote_overall/
state/diarization-progress.json
state/diarization-cache/
```

For each Whisper JSON transcript, diarization can generate:

```text
example.speaker.json
example.speaker.txt
example.speaker.srt
example.speaker.tsv
example.speaker.vtt
example.diarization.json
```

Set a Hugging Face token that can access the pyannote model:

```env
PYANNOTE_AUTH_TOKEN=hf_...
DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
DIARIZATION_VERBOSE=false
DIARIZATION_PROGRESS=
DIARIZATION_TF32=false
DIARIZATION_OOM_FALLBACK=cpu
DIARIZATION_CUDA_QUARANTINE_AFTER_OOM=false
DIARIZATION_CUDA_DEBUG_ERRORS=false
DIARIZATION_WORKER_MODE=always
DIARIZATION_GPU_MEMORY_LOG=false
DIARIZATION_WORKER_TIMEOUT_SECONDS=7200
DIARIZATION_GPU_MEMORY_WAIT_SECONDS=0
DIARIZATION_AUDIO_PREPROCESS=always
DIARIZATION_AUDIO_PREPROCESS_DIR=/tmp/auto-whisper-diarization
SAFE_OUTPUT_FILENAMES=auto
PYANNOTE_METRICS_ENABLED=0
```

`DIARIZATION_TF32` controls PyTorch TensorFloat-32 behavior for pyannote on CUDA:

- `false` disables TF32 and is the default because pyannote does this for more reproducible speaker labels.
- `true` enables TF32 and may improve speed on Ampere-class and newer GPUs, but speaker segmentation/assignment can differ between runs or CUDA stacks.
- `auto` leaves the current PyTorch/pyannote setting unchanged.

Pyannote's TF32 reproducibility warning is not a crash. If you enable TF32 intentionally, expect a speed/reproducibility tradeoff. `DIARIZATION_VERBOSE=true` prints project-level progress and timing such as path resolution, cache hits, inference timing, merge, and export steps; it does not enable internal pyannote debug logs.

`DIARIZATION_PROGRESS` controls live pyannote stage progress:

- Empty/default follows `DIARIZATION_VERBOSE`, so verbose runs show progress automatically.
- `true` enables file-level pyannote progress, percentage when pyannote exposes totals, elapsed time, and ETA when enough data exists.
- `false` disables live progress lines.

Pyannote does not diarize Whisper transcript lines one by one. During inference, progress is file-level and pyannote-stage-level, such as segmentation, embedding, and clustering. After inference finishes, the project logs the merge back into Whisper segments.

`DIARIZATION_OOM_FALLBACK` controls what happens when CUDA runs out of memory during diarization:

- `cpu` is the default and retries the same file once on CPU after CUDA cleanup. This is slower, but gives the best chance of finishing the full batch.
- `skip` records the file as failed and continues to the next file.
- `fail` stops the run immediately.

`DIARIZATION_CUDA_QUARANTINE_AFTER_OOM=false` is the default and retries CUDA on each new uncached file after a hard cleanup, while still retrying the OOM file itself on CPU. Set it to `true` for conservative survival mode, where remaining uncached files run on CPU after the first hard OOM. `DIARIZATION_CUDA_DEBUG_ERRORS=false` keeps CUDA cleanup logs concise; set it to `true` only when you need the full CUDA diagnostic text.

`DIARIZATION_WORKER_MODE` controls whether Pyannote inference runs inside the main backfill process or a short-lived worker process:

- `always` is the default and runs every uncached Pyannote inference in a worker process for the strongest practical VRAM/RAM release.
- `on_oom` is the faster hybrid mode: start in-process, then use worker isolation for later CUDA attempts after the first CUDA OOM.
- `false` keeps all inference in the main Python process.

In WSL + Docker, `torch.cuda.empty_cache()` can release PyTorch's cache, but only process exit reliably destroys that process's CUDA context. Worker mode uses that process boundary to release VRAM more aggressively. Small NVIDIA driver/context reservations may still remain visible. Set `DIARIZATION_GPU_MEMORY_LOG=true` to log `nvidia-smi` memory around worker starts/exits when available. `DIARIZATION_WORKER_TIMEOUT_SECONDS=7200` kills a stuck worker after two hours; set `DIARIZATION_GPU_MEMORY_WAIT_SECONDS` above `0` only when you want diagnostics to wait briefly after worker exit for GPU memory accounting to settle.

`DIARIZATION_AUDIO_PREPROCESS` controls whether Pyannote receives the original media file or a temporary 16 kHz mono PCM WAV:

- `always` is the default and preprocesses every audio/video file before Pyannote for the most consistent Pyannote input.
- `auto` preprocesses only compressed/risky formats such as `.m4a`, `.aac`, `.wma`, `.mp4`, `.mov`, `.mkv`, and `.webm`.
- `false` keeps the previous direct-to-Pyannote behavior.

This is useful for Pyannote crop errors such as `resulted in 440989 samples instead of the expected 441000 samples`. The source file is not renamed or modified; temporary WAV files are written under `DIARIZATION_AUDIO_PREPROCESS_DIR` and removed after each file.

`SAFE_OUTPUT_FILENAMES` controls generated output names:

- `auto` is the default and keeps readable Unicode names unless a filename is risky for Windows/Docker/path handling.
- `true` always normalizes generated output paths to ASCII-safe names.
- `false` keeps original generated output names, except path traversal and separators are still sanitized.

Original source filenames are not renamed; they are preserved in state and JSON metadata so legacy outputs remain traceable.

Build and run the separate CUDA diarization service manually:

```sh
docker compose --profile diarization build diarization-cuda
docker compose --profile diarization run --rm diarization-cuda
```

Python files are bind-mounted into `/app` for `diarization-cuda`, so script-only edits do not require rebuilding the image. Rebuild only after dependency or Dockerfile changes.

The service runs `scripts/backfill_diarization.py` by default:

```sh
python scripts/backfill_diarization.py \
  --transcripts-dir ./output \
  --output-dir ./output_pyannote \
  --overall-transcripts-dir ./output_overall \
  --overall-output-dir ./output_pyannote_overall
```

Useful options:

```sh
python scripts/backfill_diarization.py --dry-run
python scripts/backfill_diarization.py --force
python scripts/backfill_diarization.py --verbose --progress --tf32 false
python scripts/backfill_diarization.py --no-progress
python scripts/backfill_diarization.py --audio-preprocess always
python scripts/backfill_diarization.py --min-overlap-ratio 0.3 --min-speakers 1 --max-speakers 5
```

For a single transcript:

```sh
python scripts/run_diarization.py \
  --audio path/to/audio.mp3 \
  --whisper-json path/to/example.json \
  --output-dir ./output_pyannote
```

Raw pyannote output is cached per source audio, file size, modified time, backend, model, speaker-count parameters, and audio preprocessing mode. Existing legacy caches are still recognized. If `output/` and `output_overall/` reference the same audio, the second pass reuses the cached speaker timeline and reruns only merge/export.

For RTX 3050Ti 4GB and similar small-GPU systems, run Whisper and pyannote sequentially:

```sh
docker compose run --rm whisper-cuda
docker compose --profile diarization run --rm diarization-cuda
```

Do not keep Whisper and pyannote running on the GPU at the same time. To check WSL and Docker GPU visibility:

You can also run the sequential pipeline with one command. On Windows PowerShell:

```powershell
.\scripts\run_pipeline.ps1 -Cuda -Diarization
```

On Linux, macOS, or WSL:

```sh
sh ./scripts/run_pipeline.sh --cuda --diarization
```

For a diarization dry run after Whisper:

```powershell
.\scripts\run_pipeline.ps1 -Cuda -Diarization -DiarizationDryRun
```

```sh
sh ./scripts/run_pipeline.sh --cuda --diarization-dry-run
```

```sh
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

When multiple input folders share the same output root, each input still gets its own timestamped subfolder. For example:

```text
/mnt/c/Users/USER/Downloads -> /mnt/d/auto_whisper/output/Downloads_created-<YYYYMMDDHHMMSS>_modified-<YYYYMMDDHHMMSS>/
/mnt/y/Class Recording/UM CS -> /mnt/d/auto_whisper/output/UM_CS_created-<YYYYMMDDHHMMSS>_modified-<YYYYMMDDHHMMSS>/
```

## Resume Behavior

Progress is stored in `state/progress.json`, which is ignored by Git.

If a run is interrupted, run Compose again. Completed files are skipped when the source fingerprint and expected outputs still match. Video-only files with no usable audio stream are recorded and skipped on later runs unless the source file changes. Failed files are recorded and retried on the next run. Each input/output pair reuses its active timestamped output folder while a run is incomplete.

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
- `DIARIZATION_VERBOSE`: `true` for project-level pyannote progress/timing logs, or `false`
- `DIARIZATION_PROGRESS`: empty to follow `DIARIZATION_VERBOSE`, `true` for live pyannote stage progress with percentage/ETA when available, or `false`
- `DIARIZATION_TF32`: `false` for reproducible pyannote CUDA output, `true` for faster TF32 inference, or `auto` to leave PyTorch defaults unchanged
- `DIARIZATION_OOM_FALLBACK`: `cpu` to retry CUDA OOM files on CPU, `skip` to continue without retry, or `fail` to stop immediately
- `DIARIZATION_CUDA_QUARANTINE_AFTER_OOM`: `false` to retry CUDA on each new uncached file after cleanup, or `true` to switch remaining uncached files to CPU after the first CUDA OOM
- `DIARIZATION_CUDA_DEBUG_ERRORS`: `false` for concise CUDA cleanup warnings, or `true` for full CUDA exception details
- `DIARIZATION_WORKER_MODE`: `always` to isolate every uncached inference, `on_oom` to enable worker isolation after the first CUDA OOM, or `false` for in-process inference
- `DIARIZATION_GPU_MEMORY_LOG`: `true` to log `nvidia-smi` memory around worker processes, or `false`
- `DIARIZATION_WORKER_TIMEOUT_SECONDS`: max seconds before a stuck Pyannote worker is killed; `7200` by default, `0` disables the timeout
- `DIARIZATION_GPU_MEMORY_WAIT_SECONDS`: optional diagnostic wait after worker exit when GPU memory logging is enabled; `0` disables the wait
- `DIARIZATION_AUDIO_PREPROCESS`: `always` to convert all files to temporary PCM WAV before pyannote, `auto` to convert only risky compressed formats, or `false` for direct input
- `DIARIZATION_AUDIO_PREPROCESS_DIR`: temporary directory for pyannote preprocessing WAV files
- `SAFE_OUTPUT_FILENAMES`: `auto` to keep readable names when safe, `true` to always normalize, or `false` to keep original generated names
- `FINGERPRINT_MODE`: `metadata` for fast network-drive skip checks, or `sha256` for full-file hashing
- `LOCAL_STAGING`: `true` to copy only pending files to local temporary storage before transcription
- `LOCAL_STAGING_DIR`: staging directory used when `LOCAL_STAGING=true`
- `OVERALL_OUTPUT_ENABLED`: `true` to copy transcript outputs into merged `output_overall/pair-###/` folders
- `OVERALL_OUTPUT_DIR`: container path for the merged overall output, normally `/overall-output`
- `SUPPORTED_EXTENSIONS`: comma-separated file extensions to scan; defaults to `.mp3,.wav,.m4a,.flac,.ogg,.aac,.wma,.mp4,.m4v,.mov,.mkv,.webm,.avi,.wmv,.flv,.ts,.mts,.m2ts,.3gp,.3g2,.mpg,.mpeg,.vob,.ogv`

Recommended defaults:

- CPU: `WHISPER_DEVICE=cpu`, `WHISPER_FP16=false`, `FINGERPRINT_MODE=metadata`, `LOCAL_STAGING=false`, `OVERALL_OUTPUT_ENABLED=true`, `NVIDIA_VISIBLE_DEVICES=void`
- CUDA: `WHISPER_DEVICE=auto`, `WHISPER_FP16=auto`, `FINGERPRINT_MODE=metadata`, `LOCAL_STAGING=false`, `OVERALL_OUTPUT_ENABLED=true`, `NVIDIA_VISIBLE_DEVICES=all`

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

Video-only files are skipped when FFmpeg reports no usable audio stream. If a file should contain audio but is skipped, verify it with a media player or `ffprobe` and check whether the source file has a valid audio track.

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
