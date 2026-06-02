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
            assert_pipeline_override_mounts(self, override_text)

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
            assert_pipeline_override_mounts(self, override_text)


if __name__ == "__main__":
    unittest.main()
