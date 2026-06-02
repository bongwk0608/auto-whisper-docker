from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_script(script_name: str, temp_root: Path) -> Path:
    scripts_dir = temp_root / "scripts"
    scripts_dir.mkdir()
    target = scripts_dir / script_name
    shutil.copy2(ROOT / "scripts" / script_name, target)
    return target


def assert_pipeline_override_mounts(test_case: unittest.TestCase, override_text: str) -> None:
    test_case.assertIn("  whisper:", override_text)
    test_case.assertIn("  whisper-cuda:", override_text)
    test_case.assertIn("  pipeline-cuda:", override_text)
    pipeline_section = override_text.split("  pipeline-cuda:", 1)[1]
    test_case.assertIn("target: /inputs/input-001", pipeline_section)
    test_case.assertIn("target: /outputs/output-001", pipeline_section)


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


class SetupGeneratorTests(unittest.TestCase):
    def test_powershell_generator_adds_pipeline_cuda_mounts(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("PowerShell is not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source"
            source.mkdir()
            script = copy_script("init-env.ps1", temp_root)

            subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-SourceDir",
                    str(source),
                ],
                check=True,
                cwd=temp_root,
                capture_output=True,
                text=True,
            )

            override_text = (temp_root / "docker-compose.override.yml").read_text(encoding="utf-8-sig")
            env_values = parse_env(temp_root / ".env")
            assert_pipeline_override_mounts(self, override_text)
            self.assertIn("SOURCE_DIRS", env_values)
            self.assertEqual(env_values["PYANNOTE_AUTH_TOKEN"], "")
            self.assertEqual(env_values["DIARIZATION_BACKEND"], "pyannote")
            self.assertEqual(env_values["DIARIZATION_MODEL"], "pyannote/speaker-diarization-community-1")
            self.assertEqual(env_values["DIARIZATION_WORKER_MODE"], "always")
            self.assertEqual(env_values["DIARIZATION_CACHE_DIR"], "/app/state/diarization-cache")
            self.assertEqual(env_values["HF_HOME"], "/cache/huggingface")
            self.assertEqual(env_values["TORCH_HOME"], "/cache/torch")

    def test_powershell_generator_preserves_existing_diarization_env_values(self) -> None:
        powershell = shutil.which("pwsh") or shutil.which("powershell")
        if not powershell:
            self.skipTest("PowerShell is not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source"
            source.mkdir()
            script = copy_script("init-env.ps1", temp_root)
            (temp_root / ".env").write_text(
                "\n".join(
                    [
                        "SOURCE_DIRS=/old/source",
                        "OUTPUT_DIRS=/old/output",
                        "INPUT_OUTPUT_PAIRS=[]",
                        "PYANNOTE_AUTH_TOKEN=hf_test_token",
                        "PYANNOTE_METRICS_ENABLED=1",
                        "DIARIZATION_VERBOSE=true",
                        "DIARIZATION_TF32=true",
                        "DIARIZATION_WORKER_MODE=on_oom",
                        "DIARIZATION_CACHE_DIR=/custom/diarization-cache",
                        "HF_HOME=/custom/hf",
                        "TORCH_HOME=/custom/torch",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    powershell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                    "-SourceDir",
                    str(source),
                    "-Model",
                    "small",
                ],
                check=True,
                cwd=temp_root,
                capture_output=True,
                text=True,
            )

            env_values = parse_env(temp_root / ".env")
            self.assertNotEqual(env_values["SOURCE_DIRS"], "/old/source")
            self.assertEqual(env_values["WHISPER_MODEL"], "small")
            self.assertEqual(env_values["PYANNOTE_AUTH_TOKEN"], "hf_test_token")
            self.assertEqual(env_values["PYANNOTE_METRICS_ENABLED"], "1")
            self.assertEqual(env_values["DIARIZATION_VERBOSE"], "true")
            self.assertEqual(env_values["DIARIZATION_TF32"], "true")
            self.assertEqual(env_values["DIARIZATION_WORKER_MODE"], "on_oom")
            self.assertEqual(env_values["DIARIZATION_CACHE_DIR"], "/custom/diarization-cache")
            self.assertEqual(env_values["HF_HOME"], "/custom/hf")
            self.assertEqual(env_values["TORCH_HOME"], "/custom/torch")

    def test_posix_generator_adds_pipeline_cuda_mounts(self) -> None:
        shell = shutil.which("sh")
        if not shell:
            self.skipTest("POSIX sh is not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source"
            source.mkdir()
            script = copy_script("init-env.sh", temp_root)

            subprocess.run(
                [
                    shell,
                    str(script),
                    "--source-dir",
                    str(source),
                ],
                check=True,
                cwd=temp_root,
                capture_output=True,
                text=True,
            )

            override_text = (temp_root / "docker-compose.override.yml").read_text(encoding="utf-8")
            env_values = parse_env(temp_root / ".env")
            assert_pipeline_override_mounts(self, override_text)
            self.assertEqual(env_values["PYANNOTE_AUTH_TOKEN"], "")
            self.assertEqual(env_values["DIARIZATION_BACKEND"], "pyannote")
            self.assertEqual(env_values["DIARIZATION_MODEL"], "pyannote/speaker-diarization-community-1")
            self.assertEqual(env_values["DIARIZATION_WORKER_MODE"], "always")
            self.assertEqual(env_values["DIARIZATION_CACHE_DIR"], "/app/state/diarization-cache")
            self.assertEqual(env_values["HF_HOME"], "/cache/huggingface")
            self.assertEqual(env_values["TORCH_HOME"], "/cache/torch")

    def test_posix_generator_preserves_existing_diarization_env_values(self) -> None:
        shell = shutil.which("sh")
        if not shell:
            self.skipTest("POSIX sh is not available")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source = temp_root / "source"
            source.mkdir()
            script = copy_script("init-env.sh", temp_root)
            (temp_root / ".env").write_text(
                "\n".join(
                    [
                        "SOURCE_DIRS=/old/source",
                        "OUTPUT_DIRS=/old/output",
                        "INPUT_OUTPUT_PAIRS=[]",
                        "PYANNOTE_AUTH_TOKEN=hf_test_token",
                        "PYANNOTE_METRICS_ENABLED=1",
                        "DIARIZATION_VERBOSE=true",
                        "DIARIZATION_TF32=true",
                        "DIARIZATION_WORKER_MODE=on_oom",
                        "DIARIZATION_CACHE_DIR=/custom/diarization-cache",
                        "HF_HOME=/custom/hf",
                        "TORCH_HOME=/custom/torch",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            subprocess.run(
                [
                    shell,
                    str(script),
                    "--source-dir",
                    str(source),
                    "--model",
                    "small",
                ],
                check=True,
                cwd=temp_root,
                capture_output=True,
                text=True,
            )

            env_values = parse_env(temp_root / ".env")
            self.assertNotEqual(env_values["SOURCE_DIRS"], "/old/source")
            self.assertEqual(env_values["WHISPER_MODEL"], "small")
            self.assertEqual(env_values["PYANNOTE_AUTH_TOKEN"], "hf_test_token")
            self.assertEqual(env_values["PYANNOTE_METRICS_ENABLED"], "1")
            self.assertEqual(env_values["DIARIZATION_VERBOSE"], "true")
            self.assertEqual(env_values["DIARIZATION_TF32"], "true")
            self.assertEqual(env_values["DIARIZATION_WORKER_MODE"], "on_oom")
            self.assertEqual(env_values["DIARIZATION_CACHE_DIR"], "/custom/diarization-cache")
            self.assertEqual(env_values["HF_HOME"], "/custom/hf")
            self.assertEqual(env_values["TORCH_HOME"], "/custom/torch")


if __name__ == "__main__":
    unittest.main()
