from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import test_paths  # noqa: F401

from backtest_lab.drive_publish import resolve_google_drive_auth_config, upsert_pdf


class _Request:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class _Files:
    def __init__(self, found):
        self.found = found
        self.updated = None
        self.created = None
        self.query = None

    def list(self, **kwargs):
        self.query = kwargs["q"]
        return _Request({"files": self.found})

    def update(self, **kwargs):
        self.updated = kwargs
        return _Request({"id": kwargs["fileId"]})

    def create(self, **kwargs):
        self.created = kwargs
        return _Request({"id": "created-id"})


class _Service:
    def __init__(self, found):
        self.resource = _Files(found)

    def files(self):
        return self.resource


class DrivePublishTest(unittest.TestCase):
    def test_updates_existing_latest_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.pdf"
            path.write_bytes(b"pdf")
            service = _Service([{"id": "existing-id", "name": "latest.pdf"}])

            file_id, action = upsert_pdf(service, "folder-id", path, "latest.pdf", media_factory=lambda *a, **k: "media")

            self.assertEqual((file_id, action), ("existing-id", "updated"))
            self.assertEqual(service.resource.updated["fileId"], "existing-id")
            self.assertIsNone(service.resource.created)

    def test_creates_latest_pdf_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.pdf"
            path.write_bytes(b"pdf")
            service = _Service([])

            file_id, action = upsert_pdf(service, "folder-id", path, "latest.pdf", media_factory=lambda *a, **k: "media")

            self.assertEqual((file_id, action), ("created-id", "created"))
            self.assertEqual(service.resource.created["body"]["parents"], ["folder-id"])

    def test_updates_fixed_file_id_without_searching_by_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.pdf"
            path.write_bytes(b"pdf")
            service = _Service([])

            file_id, action = upsert_pdf(
                service,
                "folder-id",
                path,
                "latest.pdf",
                media_factory=lambda *a, **k: "media",
                file_id="fixed-id",
            )

            self.assertEqual((file_id, action), ("fixed-id", "updated_by_id"))
            self.assertEqual(service.resource.updated["fileId"], "fixed-id")
            self.assertIsNone(service.resource.query)

    def test_prefers_complete_google_oauth_config(self) -> None:
        config = resolve_google_drive_auth_config(
            {
                "GOOGLE_OAUTH_REFRESH_TOKEN": " refresh ",
                "GOOGLE_OAUTH_CLIENT_ID": " client ",
                "GOOGLE_OAUTH_CLIENT_SECRET": " secret ",
                "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": '{"ignored": true}',
            }
        )

        self.assertEqual(config.mode, "oauth")
        self.assertEqual(config.refresh_token, "refresh")
        self.assertEqual(config.client_id, "client")
        self.assertEqual(config.client_secret, "secret")

    def test_rejects_partial_google_oauth_config(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Incomplete Google OAuth secrets"):
            resolve_google_drive_auth_config({"GOOGLE_OAUTH_REFRESH_TOKEN": "refresh"})

    def test_accepts_service_account_as_fallback(self) -> None:
        config = resolve_google_drive_auth_config({"GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON": '{"type": "service_account"}'})

        self.assertEqual(config.mode, "service_account")


if __name__ == "__main__":
    unittest.main()
