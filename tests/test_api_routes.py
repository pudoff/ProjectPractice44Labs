"""API route tests with DB mocked; not a PostgreSQL integration test."""
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

try:
    import psycopg
except ImportError:
    sys.modules["psycopg"] = types.ModuleType("psycopg")

from fastapi.testclient import TestClient
from service.api import app
from service import db

class TestApiRoutes(unittest.TestCase):
    def test_submit_and_validation(self):
        with tempfile.TemporaryDirectory() as folder, \
             mock.patch.dict(os.environ, {"DATA_DIR": folder}), \
             mock.patch.object(db, "setup"), \
             mock.patch.object(db, "enqueue") as enqueue, \
             mock.patch.object(db, "get_job", return_value=None):
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                resp = client.post("/jobs", files={"file": ("scan.png", b"\x89PNG\r\n\x1a\nabc", "image/png")})
                self.assertEqual(resp.status_code, 202)
                self.assertEqual(resp.json()["status"], "queued")
                enqueue.assert_called_once()
                self.assertEqual(client.post(
                    "/jobs", files={"file": ("scan.pdf", b"%PDF", "application/pdf")}
                ).status_code, 415)
                self.assertEqual(client.get(
                    "/jobs/00000000-0000-0000-0000-000000000001"
                ).status_code, 404)
