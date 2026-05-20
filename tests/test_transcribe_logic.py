from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path


def load_transcribe_module():
    sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    whisper_module = types.ModuleType("whisper")
    whisper_utils = types.ModuleType("whisper.utils")
    whisper_utils.get_writer = lambda *_args, **_kwargs: None
    whisper_module.utils = whisper_utils
    sys.modules.setdefault("whisper", whisper_module)
    sys.modules.setdefault("whisper.utils", whisper_utils)

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "transcribe.py"
    spec = importlib.util.spec_from_file_location("transcribe_under_test", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


transcribe = load_transcribe_module()


class TranscribeLogicTests(unittest.TestCase):
    def test_scan_files_recurses_and_sorts_by_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "course" / "week1"
            nested.mkdir(parents=True)
            (nested / "audio.mp3").write_bytes(b"nested")
            (root / "z.wav").write_bytes(b"root")
            (nested / "notes.txt").write_text("ignore", encoding="utf-8")

            files = transcribe.scan_files(root, [".mp3", ".wav"])

            self.assertEqual([path.relative_to(root).as_posix() for path in files], ["course/week1/audio.mp3", "z.wav"])

    def test_run_id_includes_input_created_and_modified_timestamps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="Audio Folder ") as temp_dir:
            run_id = transcribe.make_run_id(Path(temp_dir))

            self.assertRegex(run_id, r"^Audio_Folder_[^/\\]*_created-\d{14}_modified-\d{14}$")

    def test_prepare_pair_preserves_nested_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            media = input_dir / "course" / "week1" / "audio.mp3"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"audio")

            pair = transcribe.InputOutputPair("pair-001", input_dir, output_dir)
            state = {"version": 1, "runs": {}, "files": {}}
            _skipped, pending = transcribe.prepare_pair(pair, state, ["txt"], [".mp3"], {"model": "base"})

            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].display_key, "course/week1/audio.mp3")
            self.assertEqual(pending[0].output_media_path.relative_to(output_dir / state["active_run_ids"]["pair-001"]).as_posix(), "course/week1/audio.mp3")

    def test_prepare_pair_reuses_existing_active_run_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            active_run_dir = output_dir / "previous_active_run"
            input_dir.mkdir()
            active_run_dir.mkdir(parents=True)
            (input_dir / "audio.mp3").write_bytes(b"audio")

            pair = transcribe.InputOutputPair("pair-001", input_dir, output_dir)
            state = {"version": 1, "runs": {}, "files": {}, "active_run_ids": {"pair-001": "previous_active_run"}}
            transcribe.prepare_pair(pair, state, ["txt"], [".mp3"], {"model": "base"})

            self.assertEqual(state["active_run_ids"]["pair-001"], "previous_active_run")


if __name__ == "__main__":
    unittest.main()
