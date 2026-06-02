from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PipelineScriptTests(unittest.TestCase):
    def test_pipeline_checks_pyannote_token_before_whisper_runs(self) -> None:
        script = (ROOT / "scripts" / "pipeline-cuda.sh").read_text(encoding="utf-8")

        token_check_index = script.index("PYANNOTE_AUTH_TOKEN is required before running pipeline-cuda")
        whisper_index = script.index("python /app/scripts/transcribe.py")

        self.assertLess(token_check_index, whisper_index)


if __name__ == "__main__":
    unittest.main()
