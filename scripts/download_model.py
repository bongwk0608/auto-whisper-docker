from __future__ import annotations

import os
from pathlib import Path

import torch
import whisper


MODEL_CACHE_DIR = Path("/root/.cache/whisper")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def choose_device(requested: str) -> str:
    requested = requested.lower()
    if not requested:
        requested = "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA, but torch cannot see a GPU. Downloading on CPU.")
        return "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError("WHISPER_DEVICE must be auto, cuda, or cpu")
    return requested


def main() -> int:
    model_name = env("WHISPER_MODEL", "small")
    device = choose_device(env("WHISPER_DOWNLOAD_DEVICE", "cpu"))
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading/loading Whisper model '{model_name}' on {device}.", flush=True)
    print(f"Model cache: {MODEL_CACHE_DIR}", flush=True)
    try:
        whisper.load_model(model_name, device=device, download_root=str(MODEL_CACHE_DIR))
    except torch.OutOfMemoryError:
        if device != "cuda":
            raise
        print("CUDA ran out of memory while verifying the model. Retrying on CPU.", flush=True)
        torch.cuda.empty_cache()
        whisper.load_model(model_name, device="cpu", download_root=str(MODEL_CACHE_DIR))

    cached_files = sorted(path.name for path in MODEL_CACHE_DIR.iterdir() if path.is_file())
    print("Model ready.", flush=True)
    if cached_files:
        print("Cached files:", flush=True)
        for file_name in cached_files:
            print(f"- {file_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
