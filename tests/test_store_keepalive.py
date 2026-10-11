"""Keepalive connections keep the WAL file alive between per-call sessions."""
import os
import tempfile
import unittest

from app import store
from app.store import MetaStore, close_keepalives, keepalive


class KeepaliveTests(unittest.TestCase):
    def setUp(self) -> None:
        close_keepalives()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(close_keepalives)
        self.path = os.path.join(self._tmp.name, "k.sqlite")

    def test_idempotent_per_path(self) -> None:
        keepalive(self.path)
        keepalive(os.path.join(self._tmp.name, ".", "k.sqlite"))
        self.assertEqual(len(store._KEEPALIVES), 1)

    def test_close_keepalives_empties_registry(self) -> None:
        keepalive(self.path)
        close_keepalives()
        self.assertEqual(store._KEEPALIVES, {})

    def test_wal_survives_metastore_write(self) -> None:
        ms = MetaStore(self.path)
        ms.upsert("d1", "a.txt", "ready")
        self.assertTrue(os.path.exists(self.path + "-wal"))
        self.assertEqual(len(store._KEEPALIVES), 1)

    def test_close_single_path(self) -> None:
        keepalive(self.path)
        store.close_keepalive(self.path)
        self.assertEqual(store._KEEPALIVES, {})


if __name__ == "__main__":
    unittest.main()
