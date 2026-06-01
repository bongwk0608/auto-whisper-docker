from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from diarization.backend import DiarizationConfig, SpeakerSegment
from diarization.audio_preprocess import (
    audio_preprocess_cache_tag,
    parse_audio_preprocess_mode,
    prepared_pyannote_audio,
    should_preprocess_audio,
)
from diarization.export_speaker_transcript import export_speaker_outputs, output_base_for_whisper_json, speaker_outputs_complete
from diarization.filename_normalization import (
    choose_output_filename,
    normalize_filename,
    parse_safe_output_policy,
    safe_relative_path,
    unique_normalized_filename,
)
from diarization.merge_whisper_speakers import MULTI_SPEAKER_POSSIBLE, assign_speakers_to_whisper_segments
from diarization.progress import DiarizationProgressReporter, ProgressContext, format_duration
from diarization.pyannote_runner import PyannoteDiarizationBackend, cleanup_cuda_memory, is_cuda_oom_error, parse_oom_fallback
from diarization.raw_cache import cache_key, load_cached_segments, save_cached_segments
from scripts.backfill_diarization import process_transcript_set
from scripts.run_diarization import run_single_diarization


class DiarizationLogicTests(unittest.TestCase):
    def test_overlap_matching_chooses_largest_overlap(self) -> None:
        merged = assign_speakers_to_whisper_segments(
            [{"start": 0.0, "end": 4.0, "text": "hello"}],
            [
                SpeakerSegment(0.0, 1.0, "Speaker_00"),
                SpeakerSegment(1.0, 4.0, "Speaker_01"),
            ],
        )

        self.assertEqual(merged[0]["assigned_speaker"], "Speaker_01")
        self.assertEqual(merged[0]["overlap_ratio"], 0.75)

    def test_below_threshold_overlap_becomes_unknown(self) -> None:
        merged = assign_speakers_to_whisper_segments(
            [{"start": 0.0, "end": 10.0, "text": "hello"}],
            [SpeakerSegment(0.0, 1.0, "Speaker_00")],
        )

        self.assertEqual(merged[0]["assigned_speaker"], "UNKNOWN")

    def test_multi_speaker_overlap_adds_warning(self) -> None:
        merged = assign_speakers_to_whisper_segments(
            [{"start": 0.0, "end": 10.0, "text": "hello"}],
            [
                SpeakerSegment(0.0, 6.0, "Speaker_00"),
                SpeakerSegment(6.0, 10.0, "Speaker_01"),
            ],
        )

        self.assertEqual(merged[0]["assigned_speaker"], "Speaker_00")
        self.assertIn(MULTI_SPEAKER_POSSIBLE, merged[0]["warnings"])

    def test_export_writes_all_speaker_formats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.mp3"
            whisper_json = root / "output" / "audio.json"
            audio.write_bytes(b"audio")
            whisper_json.parent.mkdir()
            whisper_data = {"text": "hello", "segments": [{"start": 0.0, "end": 1.5, "text": "hello"}]}
            whisper_json.write_text(json.dumps(whisper_data), encoding="utf-8")
            merged = [{"start": 0.0, "end": 1.5, "text": "hello", "assigned_speaker": "Speaker_00", "warnings": []}]
            base = root / "output_pyannote" / "audio"

            paths = export_speaker_outputs(
                base,
                audio,
                whisper_json,
                whisper_data,
                [SpeakerSegment(0.0, 1.5, "Speaker_00")],
                merged,
                DiarizationConfig(),
            )

            self.assertTrue(speaker_outputs_complete(base))
            payload = json.loads(paths["speaker_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["filename_policy"], "auto")
            self.assertEqual(payload["segments"][0]["assigned_speaker"], "Speaker_00")
            self.assertIn("Speaker_00: hello", paths["speaker_srt"].read_text(encoding="utf-8"))
            self.assertIn("Speaker_00: hello", paths["speaker_vtt"].read_text(encoding="utf-8"))
            self.assertIn("start\tend\tspeaker\ttext", paths["speaker_tsv"].read_text(encoding="utf-8"))
            self.assertIn("# Speaker Transcript", paths["speaker_txt"].read_text(encoding="utf-8"))

    def test_filename_normalization_and_collision(self) -> None:
        self.assertEqual(
            normalize_filename("WhatsApp Audio 2026-05-31 at 5-04-41 PM - 1(3).mp3"),
            "whatsapp_audio_2026-05-31_at_5-04-41_pm_-_1_3.mp3",
        )
        existing: set[str] = set()
        first = unique_normalized_filename("A B.mp3", existing)
        second = unique_normalized_filename("A?B.mp3", existing)

        self.assertEqual(first.safe_filename, "a_b.mp3")
        self.assertEqual(second.safe_filename, "a_b_1.mp3")
        self.assertEqual(second.collision_index, 1)

    def test_safe_relative_path_normalizes_components_and_collisions(self) -> None:
        existing: dict[Path, set[str]] = {}

        first = safe_relative_path(Path("政治 影片/A?B.mp4"), existing)
        second = safe_relative_path(Path("政治 影片/A B.mp4"), existing)

        self.assertEqual(first.name, "a_b.mp4")
        self.assertEqual(second.name, "A B.mp4")

    def test_safe_output_policy_modes(self) -> None:
        chinese = "拉菲茲的破局之戰 19-May-2026.mp4"

        self.assertEqual(parse_safe_output_policy(None), "auto")
        self.assertEqual(choose_output_filename(chinese, "auto"), chinese)
        self.assertEqual(choose_output_filename("a<b>c?.mp4", "auto"), "a_b_c.mp4")
        self.assertEqual(choose_output_filename("CON.mp4", "auto"), "con_file.mp4")
        self.assertEqual(choose_output_filename(chinese, "true"), "19-may-2026.mp4")
        self.assertEqual(choose_output_filename(chinese, "false"), chinese)

    def test_true_policy_normalizes_and_resolves_collisions(self) -> None:
        existing: dict[Path, set[str]] = {}

        first = safe_relative_path(Path("folder/A?B.mp4"), existing, "true")
        second = safe_relative_path(Path("folder/A B.mp4"), existing, "true")

        self.assertEqual(first.as_posix(), "folder/a_b.mp4")
        self.assertEqual(second.as_posix(), "folder/a_b_1.mp4")

    def test_raw_cache_reuses_segments_and_key_changes_with_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.mp3"
            audio.write_bytes(b"audio")
            config = DiarizationConfig(model="model-a")
            other_config = DiarizationConfig(model="model-b")

            key = cache_key(audio, config)
            other_key = cache_key(audio, other_config)
            preprocessed_key = cache_key(audio, config, audio_preprocess="pcm16k-mono-v1:auto")
            save_cached_segments(root / "cache", key, audio, config, [SpeakerSegment(0.0, 1.0, "Speaker_00")])

            cached = load_cached_segments(root / "cache", key)

            self.assertNotEqual(key, other_key)
            self.assertNotEqual(key, preprocessed_key)
            self.assertEqual(cached, [SpeakerSegment(0.0, 1.0, "Speaker_00")])

    def test_audio_preprocess_policy_modes(self) -> None:
        self.assertEqual(parse_audio_preprocess_mode(None), "auto")
        self.assertTrue(should_preprocess_audio(Path("audio.m4a"), "auto"))
        self.assertTrue(should_preprocess_audio(Path("video.mp4"), "auto"))
        self.assertFalse(should_preprocess_audio(Path("audio.wav"), "auto"))
        self.assertTrue(should_preprocess_audio(Path("audio.wav"), "always"))
        self.assertFalse(should_preprocess_audio(Path("audio.m4a"), "false"))
        self.assertEqual(audio_preprocess_cache_tag("auto", Path("audio.m4a")), "pcm16k-mono-v1:auto")
        self.assertEqual(audio_preprocess_cache_tag("auto", Path("audio.wav")), "direct:auto")

    def test_prepared_pyannote_audio_converts_and_cleans_temp_wav(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.m4a"
            audio.write_bytes(b"audio")
            seen_target: list[Path] = []

            def fake_run(command, capture_output, text, check):
                target = Path(command[-1])
                target.write_bytes(b"wav")
                return mock.Mock(returncode=0, stderr="", stdout="")

            with mock.patch("diarization.audio_preprocess.subprocess.run", side_effect=fake_run):
                with prepared_pyannote_audio(audio, "auto", root / "staging") as prepared:
                    seen_target.append(prepared)
                    self.assertEqual(prepared.suffix, ".wav")
                    self.assertTrue(prepared.exists())

            self.assertFalse(seen_target[0].exists())

    def test_backfill_dry_run_records_pending_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "output"
            output = root / "output_pyannote"
            audio = root / "audio.mp3"
            whisper_json = transcripts / "audio.json"
            transcripts.mkdir()
            audio.write_bytes(b"audio")
            whisper_json.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}), encoding="utf-8")
            manifest: dict[str, object] = {"version": 1, "jobs": {}}

            result = process_transcript_set(
                transcripts,
                output,
                {str(whisper_json.resolve()): audio},
                manifest,
                root / "state" / "diarization-progress.json",
                DiarizationConfig(),
                0.3,
                root / "cache",
                None,
                force=False,
                dry_run=True,
            )

            self.assertEqual(result, (0, 0, 0, 0))
            self.assertFalse(output.exists())
            self.assertEqual(next(iter(manifest["jobs"].values()))["status"], "pending")

    def test_backfill_missing_audio_records_skipped_missing_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            transcripts = root / "output"
            output = root / "output_pyannote"
            whisper_json = transcripts / "audio.json"
            transcripts.mkdir()
            whisper_json.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}), encoding="utf-8")
            manifest: dict[str, object] = {"version": 1, "jobs": {}}

            result = process_transcript_set(
                transcripts,
                output,
                {},
                manifest,
                root / "state" / "diarization-progress.json",
                DiarizationConfig(),
                0.3,
                root / "cache",
                None,
                force=False,
                dry_run=False,
            )

            self.assertEqual(result, (0, 0, 1, 0))
            self.assertEqual(next(iter(manifest["jobs"].values()))["status"], "skipped_missing_audio")

    def test_output_base_uses_safe_nested_relative_path(self) -> None:
        base = output_base_for_whisper_json(
            Path("/tmp/output/run/course week/audio file.json"),
            Path("/tmp/output"),
            Path("/tmp/output_pyannote"),
        )

        self.assertEqual(base.as_posix(), "/tmp/output_pyannote/run/course week/audio file")

    def test_progress_duration_formats_unknown_and_clock_time(self) -> None:
        self.assertEqual(format_duration(None), "unknown")
        self.assertEqual(format_duration(3723), "01:02:03")

    def test_progress_eta_is_unknown_without_total(self) -> None:
        self.assertIsNone(DiarizationProgressReporter.eta(30.0, current=None, total=100.0))
        self.assertIsNone(DiarizationProgressReporter.eta(30.0, current=0.0, total=100.0))

    def test_progress_eta_uses_current_and_total(self) -> None:
        self.assertEqual(DiarizationProgressReporter.eta(30.0, current=25.0, total=100.0), 90.0)

    def test_progress_throttles_repeated_identical_lines(self) -> None:
        ticks = iter([0.0, 0.0, 1.0, 2.0, 3.0, 13.0])
        reporter = DiarizationProgressReporter(
            ProgressContext(file_index=1, file_total=2),
            clock=lambda: next(ticks),
        )

        self.assertTrue(reporter.update("pyannote segmentation", current=1, total=100))
        self.assertFalse(reporter.update("pyannote segmentation", current=1, total=100))
        self.assertTrue(reporter.update("pyannote segmentation", current=2, total=100))
        self.assertFalse(reporter.update("pyannote segmentation", current=2, total=100))
        self.assertTrue(reporter.update("pyannote segmentation", current=2, total=100))

    def test_pyannote_backend_unwraps_community_output(self) -> None:
        class Annotation:
            def itertracks(self, yield_label: bool = False):
                return iter(())

        class DiarizeOutput:
            speaker_diarization = Annotation()

        backend = PyannoteDiarizationBackend.__new__(PyannoteDiarizationBackend)

        self.assertIs(backend.unwrap_diarization_output(DiarizeOutput()), DiarizeOutput.speaker_diarization)

    def test_cuda_oom_detection_and_fallback_parser(self) -> None:
        self.assertTrue(is_cuda_oom_error(RuntimeError("CUDA error: out of memory")))
        self.assertTrue(is_cuda_oom_error(RuntimeError("GET was unable to find an engine to execute this computation")))
        self.assertEqual(parse_oom_fallback(None), "cpu")
        self.assertEqual(parse_oom_fallback("skip"), "skip")

    def test_cuda_cleanup_is_best_effort(self) -> None:
        fake_torch = mock.Mock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA error: out of memory")

        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            cleanup_cuda_memory(verbose=True)

    def test_run_single_diarization_retries_cuda_oom_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.mp3"
            whisper_json = root / "audio.json"
            output_base = root / "out" / "audio"
            audio.write_bytes(b"audio")
            whisper_json.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}), encoding="utf-8")
            devices: list[str | None] = []

            class FakeBackend:
                def __init__(self, config, auth_token=None, device=None, tf32_mode=None, verbose=False):
                    self.device = device
                    devices.append(device)

                def diarize(self, audio_path, progress_reporter=None):
                    if self.device is None:
                        raise RuntimeError("CUDA error: out of memory")
                    return [SpeakerSegment(0.0, 1.0, "SPEAKER_00")]

            with mock.patch("scripts.run_diarization.PyannoteDiarizationBackend", FakeBackend):
                paths, cache_hit = run_single_diarization(
                    audio,
                    whisper_json,
                    output_base,
                    DiarizationConfig(),
                    0.3,
                    root / "cache",
                    oom_fallback="cpu",
                )

            self.assertFalse(cache_hit)
            self.assertEqual(devices, [None, "cpu"])
            self.assertTrue(paths["speaker_json"].exists())

    def test_run_single_diarization_uses_preprocessed_audio_but_exports_original(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.m4a"
            whisper_json = root / "audio.json"
            output_base = root / "out" / "audio"
            audio.write_bytes(b"audio")
            whisper_json.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}), encoding="utf-8")
            backend_audio_paths: list[Path] = []

            def fake_run(command, capture_output, text, check):
                Path(command[-1]).write_bytes(b"wav")
                return mock.Mock(returncode=0, stderr="", stdout="")

            class FakeBackend:
                def __init__(self, config, auth_token=None, device=None, tf32_mode=None, verbose=False):
                    pass

                def diarize(self, audio_path, progress_reporter=None):
                    backend_audio_paths.append(Path(audio_path))
                    return [SpeakerSegment(0.0, 1.0, "SPEAKER_00")]

            with (
                mock.patch("scripts.run_diarization.PyannoteDiarizationBackend", FakeBackend),
                mock.patch("diarization.audio_preprocess.subprocess.run", side_effect=fake_run),
            ):
                paths, cache_hit = run_single_diarization(
                    audio,
                    whisper_json,
                    output_base,
                    DiarizationConfig(),
                    0.3,
                    root / "cache",
                    audio_preprocess="auto",
                    audio_preprocess_dir=root / "staging",
                )

            self.assertFalse(cache_hit)
            self.assertEqual(backend_audio_paths[0].suffix, ".wav")
            self.assertFalse(backend_audio_paths[0].exists())
            payload = json.loads(paths["speaker_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["source_audio"], str(audio))


if __name__ == "__main__":
    unittest.main()
