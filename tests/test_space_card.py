"""The GitHub README stays free of Hugging Face card metadata."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _push_module():
    path = ROOT / "scripts" / "push_huggingface_space.py"
    spec = importlib.util.spec_from_file_location("push_huggingface_space", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SpaceCardTests(unittest.TestCase):
    def test_github_readme_opens_on_the_title(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue(readme.startswith("# LeaseGPT\n"))
        self.assertNotIn("colorFrom:", readme)

    def test_space_metadata_keeps_docker_settings(self):
        meta = (ROOT / "huggingface" / "space.yml").read_text(encoding="utf-8")
        self.assertIn("sdk: docker\n", meta)
        self.assertIn("app_port: 8501\n", meta)
        self.assertFalse(meta.startswith("---"))

    def test_prepend_wraps_metadata_once(self):
        module = _push_module()
        meta = "title: LeaseGPT\nsdk: docker\n"
        readme = "# LeaseGPT\n"
        card = module.with_space_card(readme, meta)
        self.assertEqual(card, "---\ntitle: LeaseGPT\nsdk: docker\n---\n\n# LeaseGPT\n")
        self.assertEqual(module.with_space_card(card, meta), card)


if __name__ == "__main__":
    unittest.main()
