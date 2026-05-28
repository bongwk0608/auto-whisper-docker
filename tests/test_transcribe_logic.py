from __future__ import annotations

import importlib.util
import json
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

    def test_run_id_can_use_host_display_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="input-001") as temp_dir:
            run_id = transcribe.make_run_id(Path(temp_dir), "Downloads")

            self.assertRegex(run_id, r"^Downloads_created-\d{14}_modified-\d{14}$")

    def test_source_display_name_prefers_host_input_basename(self) -> None:
        with tempfile.TemporaryDirectory(prefix="input-001") as temp_dir:
            display_name = transcribe.source_display_name(Path(temp_dir), "/mnt/y/Class Recording/UM CS")

            self.assertEqual(display_name, "UM CS")

    def test_source_display_name_falls_back_to_container_folder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="input-001") as temp_dir:
            display_name = transcribe.source_display_name(Path(temp_dir), "")

            self.assertTrue(display_name.startswith("input-001"))

    def test_safe_folder_name_sanitizes_host_display_name(self) -> None:
        self.assertEqual(transcribe.safe_folder_name("UM CS"), "UM_CS")
        self.assertEqual(transcribe.safe_folder_name("Class Recording: UM/CS"), "Class_Recording__UM_CS")

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
            prepared = transcribe.prepare_pair(pair, 0, state, ["txt"], [".mp3"], {"model": "base"}, [], [], "metadata", False, True, root / "overall")
            pending = prepared.pending

            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].display_key, "course/week1/audio.mp3")
            self.assertEqual(pending[0].state_key, "pair-001:course/week1/audio.mp3")
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
            prepared = transcribe.prepare_pair(pair, 0, state, ["txt"], [".mp3"], {"model": "base"}, [], [], "metadata", False, True, root / "overall")

            self.assertEqual(state["active_run_ids"]["pair-001"], "previous_active_run")
            self.assertIsNotNone(prepared.mapping)
            self.assertEqual(prepared.mapping["run_id"], "previous_active_run")

    def test_prepare_pair_uses_host_folder_name_for_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input-001"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "audio.mp3").write_bytes(b"audio")

            pair = transcribe.InputOutputPair("pair-001", input_dir, output_dir)
            state = {"version": 1, "runs": {}, "files": {}}
            prepared = transcribe.prepare_pair(
                pair,
                0,
                state,
                ["txt"],
                [".mp3"],
                {"model": "base"},
                ["/mnt/c/Users/USER/Downloads"],
                ["/mnt/d/auto_whisper/output"],
                "metadata",
                False,
                True,
                root / "overall",
            )

            run_id = state["active_run_ids"]["pair-001"]
            self.assertRegex(run_id, r"^Downloads_created-\d{14}_modified-\d{14}$")
            self.assertEqual(prepared.mapping["source_folder_name"], "Downloads")
            self.assertEqual(prepared.mapping["run_id"], run_id)
            self.assertEqual(Path(prepared.mapping["run_output_dir"]).name, run_id)

    def test_scan_files_recurses_multiple_levels_and_ignores_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            deep = root / "course" / "week1" / "day1"
            deep.mkdir(parents=True)
            (deep / "lecture.MP3").write_bytes(b"deep")
            (root / "course" / "week1" / "slides.pdf").write_bytes(b"ignore")

            files = transcribe.scan_files(root, [".mp3"])

            self.assertEqual([path.relative_to(root).as_posix() for path in files], ["course/week1/day1/lecture.MP3"])

    def test_mapping_manifest_includes_pair_details_and_file_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_a = root / "Downloads"
            input_b = root / "UM CS"
            output_root = root / "output"
            (input_a / "nested").mkdir(parents=True)
            (input_b / "course" / "week1").mkdir(parents=True)
            (input_a / "nested" / "a.mp3").write_bytes(b"a")
            (input_b / "course" / "week1" / "b.wav").write_bytes(b"b")

            state = {"version": 1, "runs": {}, "files": {}}
            formats = ["txt", "json"]
            host_inputs = ["/mnt/c/Users/USER/Downloads", "/mnt/y/Class Recording/UM CS"]
            host_outputs = ["/mnt/d/auto_whisper/output", "/mnt/d/auto_whisper/output"]
            pair_a = transcribe.InputOutputPair("pair-001", input_a, output_root)
            pair_b = transcribe.InputOutputPair("pair-002", input_b, output_root)

            prepared_a = transcribe.prepare_pair(pair_a, 0, state, formats, [".mp3", ".wav"], {"model": "base"}, host_inputs, host_outputs, "metadata", False, True, root / "overall")
            prepared_b = transcribe.prepare_pair(pair_b, 1, state, formats, [".mp3", ".wav"], {"model": "base"}, host_inputs, host_outputs, "metadata", False, True, root / "overall")

            mappings = [prepared_a.mapping, prepared_b.mapping]
            self.assertEqual([mapping["pair_id"] for mapping in mappings], ["pair-001", "pair-002"])
            self.assertEqual(mappings[0]["host_input_dir"], "/mnt/c/Users/USER/Downloads")
            self.assertEqual(mappings[1]["host_input_dir"], "/mnt/y/Class Recording/UM CS")
            self.assertEqual(mappings[0]["host_output_root"], "/mnt/d/auto_whisper/output")
            self.assertTrue(mappings[1]["recursive_scan_enabled"])
            self.assertEqual(mappings[1]["fingerprint_mode"], "metadata")
            self.assertFalse(mappings[1]["local_staging"])
            self.assertEqual(mappings[0]["supported_file_count"], 1)
            self.assertEqual(mappings[1]["supported_file_count"], 1)
            self.assertEqual(mappings[1]["source_folder_name"], "UM CS")
            self.assertRegex(mappings[1]["run_id"], r"^UM_CS_created-\d{14}_modified-\d{14}$")
            self.assertTrue(mappings[1]["overall_output_enabled"])
            self.assertEqual(mappings[1]["overall_output_root"], str(root / "overall"))
            self.assertEqual(mappings[1]["overall_pair_output_dir"], str(root / "overall" / "pair-002"))
            self.assertEqual(prepared_b.pending[0].output_media_path.relative_to(output_root / mappings[1]["run_id"]).as_posix(), "course/week1/b.wav")

    def test_overall_output_base_preserves_nested_path_without_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input" / "Dissertation Discussion" / "STEREO" / "FOLDER01" / "ZOOM0001.WAV"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"audio")

            output_base = transcribe.overall_output_base(
                root / "output_overall",
                "pair-002",
                "Dissertation Discussion/STEREO/FOLDER01/ZOOM0001.WAV",
                source,
            )

            relative = output_base.relative_to(root / "output_overall" / "pair-002").as_posix()
            self.assertRegex(relative, r"^Dissertation Discussion/STEREO/FOLDER01/ZOOM0001_created-\d{14}_modified-\d{14}$")

    def test_copy_overall_outputs_copies_multiple_formats_and_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input" / "course" / "week1" / "audio.mp3"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"audio")
            project_txt = root / "output" / "run" / "course" / "week1" / "audio_created-1_modified-1.txt"
            project_json = root / "output" / "run" / "course" / "week1" / "audio_created-1_modified-1.json"
            project_txt.parent.mkdir(parents=True)
            project_txt.write_text("new txt", encoding="utf-8")
            project_json.write_text('{"text":"new"}', encoding="utf-8")

            first_outputs = transcribe.copy_overall_outputs(
                [project_txt, project_json],
                source,
                "course/week1/audio.mp3",
                "pair-001",
                True,
                root / "output_overall",
                ["txt", "json"],
            )
            first_outputs[0].write_text("old txt", encoding="utf-8")

            second_outputs = transcribe.copy_overall_outputs(
                [project_txt, project_json],
                source,
                "course/week1/audio.mp3",
                "pair-001",
                True,
                root / "output_overall",
                ["txt", "json"],
            )

            self.assertEqual(first_outputs, second_outputs)
            self.assertEqual(second_outputs[0].read_text(encoding="utf-8"), "new txt")
            self.assertEqual(second_outputs[1].read_text(encoding="utf-8"), '{"text":"new"}')
            self.assertEqual(second_outputs[0].relative_to(root / "output_overall" / "pair-001").parent.as_posix(), "course/week1")

    def test_copy_overall_outputs_disabled_returns_no_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "input" / "audio.mp3"
            project_output = root / "output" / "audio.txt"
            source.parent.mkdir()
            source.write_bytes(b"audio")
            project_output.parent.mkdir()
            project_output.write_text("text", encoding="utf-8")

            overall_outputs = transcribe.copy_overall_outputs(
                [project_output],
                source,
                "audio.mp3",
                "pair-001",
                False,
                root / "output_overall",
                ["txt"],
            )

            self.assertEqual(overall_outputs, [])
            self.assertFalse((root / "output_overall").exists())

    def test_write_mapping_manifests_writes_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            mappings = [
                {
                    "pair_id": "pair-001",
                    "host_input_dir": "/host/input",
                    "container_input_dir": "/inputs/input-001",
                    "host_output_root": "/host/output",
                    "container_output_root": "/outputs/output-001",
                    "run_id": "input_created-20260520120000_modified-20260520120500",
                    "run_output_dir": "/outputs/output-001/input_created-20260520120000_modified-20260520120500",
                    "source_folder_name": "input",
                    "input_created_timestamp": "20260520120000",
                    "input_modified_timestamp": "20260520120500",
                    "formats": "txt,json",
                    "fingerprint_mode": "metadata",
                    "local_staging": False,
                    "overall_output_enabled": True,
                    "overall_output_root": "/overall-output",
                    "overall_pair_output_dir": "/overall-output/pair-001",
                    "recursive_scan_enabled": True,
                    "supported_file_count": 2,
                    "updated_at": "2026-05-20T12:05:00",
                }
            ]

            transcribe.write_mapping_manifests([output_root], mappings)

            manifest = json.loads((output_root / "input-output-mapping.json").read_text(encoding="utf-8"))
            csv_text = (output_root / "input-output-mapping.csv").read_text(encoding="utf-8")
            self.assertEqual(manifest["mappings"][0]["pair_id"], "pair-001")
            self.assertIn("recursive_scan_enabled", csv_text)
            self.assertIn("overall_pair_output_dir", csv_text)
            self.assertIn("pair-001", csv_text)

    def test_metadata_fingerprint_does_not_include_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio.mp3"
            path.write_bytes(b"audio")

            fingerprint = transcribe.file_fingerprint(path, "metadata")

            self.assertEqual(fingerprint["size"], 5)
            self.assertIn("mtime_ns", fingerprint)
            self.assertNotIn("sha256", fingerprint)

    def test_sha256_fingerprint_includes_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio.mp3"
            path.write_bytes(b"audio")

            fingerprint = transcribe.file_fingerprint(path, "sha256")

            self.assertEqual(fingerprint["size"], 5)
            self.assertEqual(fingerprint["sha256"], "6ed8919ce20490a5e3ad8630a4fab69475297abd07db73918dd5f36fcfaeb11b")

    def test_invalid_fingerprint_mode_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "FINGERPRINT_MODE"):
            transcribe.parse_fingerprint_mode("full")

    def test_metadata_fingerprint_can_match_existing_sha256_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "out.txt"
            output.write_text("done", encoding="utf-8")
            fingerprint = {"size": 5, "mtime_ns": 123}
            record = {
                "status": "complete",
                "fingerprint": {"size": 5, "mtime_ns": 123, "sha256": "old"},
                "formats": ["txt"],
                "project_outputs": [str(output)],
            }

            self.assertTrue(transcribe.is_complete(record, fingerprint, ["txt"]))

    def test_metadata_fingerprint_change_retriggers_work(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "out.txt"
            output.write_text("done", encoding="utf-8")
            record = {
                "status": "complete",
                "fingerprint": {"size": 5, "mtime_ns": 123},
                "formats": ["txt"],
                "project_outputs": [str(output)],
            }

            self.assertFalse(transcribe.is_complete(record, {"size": 6, "mtime_ns": 123}, ["txt"]))
            self.assertFalse(transcribe.is_complete(record, {"size": 5, "mtime_ns": 124}, ["txt"]))

    def test_skipped_no_audio_is_terminal_for_matching_fingerprint_and_formats(self) -> None:
        fingerprint = {"size": 5, "mtime_ns": 123}
        record = {
            "status": "skipped_no_audio",
            "fingerprint": fingerprint,
            "formats": ["txt", "json"],
        }

        self.assertTrue(transcribe.is_skipped_no_audio(record, fingerprint, ["json", "txt"]))
        self.assertTrue(transcribe.is_terminal_record(record, fingerprint, ["txt", "json"]))

    def test_skipped_no_audio_retries_when_fingerprint_changes(self) -> None:
        record = {
            "status": "skipped_no_audio",
            "fingerprint": {"size": 5, "mtime_ns": 123},
            "formats": ["txt"],
        }

        self.assertFalse(transcribe.is_skipped_no_audio(record, {"size": 6, "mtime_ns": 123}, ["txt"]))
        self.assertFalse(transcribe.is_terminal_record(record, {"size": 5, "mtime_ns": 124}, ["txt"]))

    def test_prepare_pair_skips_no_audio_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "input"
            output_dir = root / "output"
            media = input_dir / "video.mp4"
            input_dir.mkdir()
            media.write_bytes(b"video")
            fingerprint = transcribe.file_fingerprint(media, "metadata")
            state = {
                "version": 1,
                "runs": {},
                "files": {
                    "pair-001:video.mp4": {
                        "status": "skipped_no_audio",
                        "fingerprint": fingerprint,
                        "formats": ["txt"],
                    }
                },
            }

            pair = transcribe.InputOutputPair("pair-001", input_dir, output_dir)
            prepared = transcribe.prepare_pair(pair, 0, state, ["txt"], [".mp4"], {"model": "base"}, [], [], "metadata", False, True, root / "overall")

            self.assertEqual(prepared.skipped, 0)
            self.assertEqual(prepared.skipped_no_audio, 1)
            self.assertEqual(prepared.pending, [])

    def test_no_audio_error_detection_matches_ffmpeg_message(self) -> None:
        error = RuntimeError("Failed to load audio: Output file #0 does not contain any stream")

        self.assertTrue(transcribe.is_no_audio_error(error))
        self.assertFalse(transcribe.is_no_audio_error(RuntimeError("Permission denied while reading file")))

    def test_default_supported_extensions_include_common_video_formats(self) -> None:
        extensions = transcribe.parse_list(transcribe.DEFAULT_SUPPORTED_EXTENSIONS)

        for extension in [".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv", ".flv", ".ts", ".mts", ".m2ts", ".3gp", ".3g2", ".mpg", ".mpeg", ".vob", ".ogv"]:
            self.assertIn(extension, extensions)

    def test_transcription_source_stages_and_cleans_pending_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "network" / "audio.mp3"
            staging = root / "staging"
            source.parent.mkdir()
            source.write_bytes(b"audio")

            with transcribe.transcription_source(source, True, staging) as staged:
                self.assertNotEqual(staged, source)
                self.assertTrue(staged.exists())
                self.assertEqual(staged.name, source.name)
                self.assertEqual(staged.read_bytes(), b"audio")
                staged_parent = staged.parent

            self.assertFalse(staged_parent.exists())

    def test_transcription_source_without_staging_uses_original_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "audio.mp3"
            source.write_bytes(b"audio")

            with transcribe.transcription_source(source, False, Path(temp_dir) / "staging") as path:
                self.assertEqual(path, source)


if __name__ == "__main__":
    unittest.main()
