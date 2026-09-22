"""Static checks that app modules never import the scraper or RentCast fetch script."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

APP_MODULES = (
    ROOT / "app.py",
    ROOT / "leasegpt" / "listings.py",
    ROOT / "leasegpt" / "retriever.py",
    ROOT / "leasegpt" / "generator.py",
    ROOT / "leasegpt" / "groq_chat.py",
    ROOT / "leasegpt" / "ui.py",
)

FORBIDDEN_MODULES = (
    "leasegpt.scraper",
    "scripts.fetch_rentcast_listings",
    "fetch_rentcast_listings",
    "selenium",
    "langchain_openai",
    "openai",
)
FORBIDDEN_NAMES = ("ChatOpenAI",)
RENTCAST_HOST = "api.rentcast.io"


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names.add(module)
            for alias in node.names:
                if module:
                    names.add(f"{module}.{alias.name}")
                names.add(alias.name)
    return names


class AppImportGuardTests(unittest.TestCase):
    def test_app_modules_exist(self):
        missing = [str(path.relative_to(ROOT)) for path in APP_MODULES if not path.is_file()]
        self.assertEqual(missing, [])

    def test_app_modules_do_not_import_scraper_or_fetch_script(self):
        for path in APP_MODULES:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            imported = _imported_names(tree)
            rel = path.relative_to(ROOT)
            for forbidden in FORBIDDEN_MODULES:
                self.assertNotIn(
                    forbidden,
                    imported,
                    f"{rel} imports forbidden module {forbidden}",
                )
            for name in FORBIDDEN_NAMES:
                self.assertNotIn(name, imported, f"{rel} imports {name}")
            self.assertNotIn(
                RENTCAST_HOST,
                source,
                f"{rel} must not call RentCast ({RENTCAST_HOST})",
            )
            self.assertNotIn("import leasegpt.scraper", source)
            self.assertNotIn("from leasegpt.scraper", source)
            self.assertNotIn("from leasegpt import scraper", source)
            self.assertNotIn("import fetch_rentcast", source)
            self.assertNotIn("from fetch_rentcast", source)
            self.assertNotIn("scripts.fetch_rentcast_listings", source)

    def test_chat_path_stays_on_groq_wrapper(self):
        generator = (ROOT / "leasegpt" / "generator.py").read_text(encoding="utf-8")
        groq_chat = (ROOT / "leasegpt" / "groq_chat.py").read_text(encoding="utf-8")
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("from leasegpt.groq_chat import", generator)
        self.assertIn("ChatGroq", generator)
        self.assertIn("class ChatGroq", groq_chat)
        self.assertNotIn("ChatOpenAI", app)
        self.assertNotIn("ChatOpenAI", generator)


if __name__ == "__main__":
    unittest.main()
