import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tender_research import paths


class OriginalTests(unittest.TestCase):
    def test_prefers_workspace_then_cache_then_explains_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            datasets, cache = Path(tmp, "datasets"), Path(tmp, "cache")
            with mock.patch.object(paths, "DATASETS", datasets), mock.patch.object(paths, "CACHE", cache):
                base, rel = datasets / "CASE-002_合肥_道路", "01-招标资料/招标文件.pdf"
                with self.assertRaisesRegex(FileNotFoundError, "store fetch"):
                    paths.original(base, rel)
                (cache / "CASE-002_合肥_道路/01-招标资料").mkdir(parents=True)
                (cache / "CASE-002_合肥_道路" / rel).write_bytes(b"cache")
                self.assertEqual(paths.original(base, rel).read_bytes(), b"cache")
                (base / "01-招标资料").mkdir(parents=True)
                (base / rel).write_bytes(b"workspace")
                self.assertEqual(paths.original(base, rel).read_bytes(), b"workspace")


if __name__ == "__main__":
    unittest.main()
