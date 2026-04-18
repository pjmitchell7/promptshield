from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from main import load_prompts


class TestPromptLoading(unittest.TestCase):
    def test_delimiter_based_loading_preserves_multiline_blocks(self) -> None:
        content = """===PROMPT===
First prompt line 1
First prompt line 2

===PROMPT===
Second prompt
with multiple lines
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.txt"
            path.write_text(content, encoding="utf-8")

            prompts = load_prompts(str(path))

        self.assertEqual(len(prompts), 2)
        self.assertIn("First prompt line 1", prompts[0])
        self.assertIn("First prompt line 2", prompts[0])
        self.assertIn("Second prompt", prompts[1])
        self.assertIn("with multiple lines", prompts[1])

    def test_line_fallback_still_works(self) -> None:
        content = "alpha\nbeta\n\n gamma \n"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "prompts.txt"
            path.write_text(content, encoding="utf-8")

            prompts = load_prompts(str(path))

        self.assertEqual(prompts, ["alpha", "beta", "gamma"])


if __name__ == "__main__":
    unittest.main()