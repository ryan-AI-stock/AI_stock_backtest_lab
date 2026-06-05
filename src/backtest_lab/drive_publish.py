from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FOLDER_ID = "1O6Se-HfI7ZDTQ-LWeAO6f8vtvoLcCzIj"
DEFAULT_REMOTE_NAME = "AI股票最佳策略每日觀察報告_最新版_v20260605.pdf"
LEGACY_REMOTE_NAMES = ("AI股票凍結策略每日觀察報告_最新版.pdf",)
DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


@dataclass(frozen=True)
class GoogleDriveAuthConfig:
    mode: str
    refresh_token: str = ""
    client_id: str = ""
    client_secret: str = ""
    service_account_json: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload or replace the latest best-strategy PDF on Google Drive.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--folder-id", default=os.environ.get("FROZEN_REPORT_DRIVE_FOLDER_ID") or DEFAULT_FOLDER_ID)
    parser.add_argument("--file-id", default=os.environ.get("FROZEN_REPORT_DRIVE_FILE_ID", ""))
    parser.add_argument("--remote-name", default=DEFAULT_REMOTE_NAME)
    args = parser.parse_args()

    service, auth_mode = build_drive_service()
    file_id, action = upsert_pdf(
        service,
        args.folder_id,
        Path(args.file),
        args.remote_name,
        file_id=args.file_id.strip() or None,
        legacy_remote_names=LEGACY_REMOTE_NAMES,
    )
    print(f"DRIVE_AUTH_MODE={auth_mode}")
    print(f"DRIVE_ACTION={action}")
    print(f"DRIVE_FILE_ID={file_id}")


def resolve_google_drive_auth_config(env: Mapping[str, str] | None = None) -> GoogleDriveAuthConfig:
    source = env or os.environ
    refresh_token = source.get("GOOGLE_OAUTH_REFRESH_TOKEN", "").strip()
    client_id = source.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    client_secret = source.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    oauth_values = (refresh_token, client_id, client_secret)
    if all(oauth_values):
        return GoogleDriveAuthConfig(
            mode="oauth",
            refresh_token=refresh_token,
            client_id=client_id,
            client_secret=client_secret,
        )
    if any(oauth_values):
        raise RuntimeError(
            "Incomplete Google OAuth secrets. Set GOOGLE_OAUTH_REFRESH_TOKEN, "
            "GOOGLE_OAUTH_CLIENT_ID, and GOOGLE_OAUTH_CLIENT_SECRET together."
        )

    service_account_json = source.get("GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    if service_account_json:
        return GoogleDriveAuthConfig(mode="service_account", service_account_json=service_account_json)

    raise RuntimeError(
        "Missing Google Drive credentials. Preferred setup matches the other stock report repos: "
        "GOOGLE_OAUTH_REFRESH_TOKEN, GOOGLE_OAUTH_CLIENT_ID, and GOOGLE_OAUTH_CLIENT_SECRET. "
        "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON is only a fallback."
    )


def build_drive_service():
    from googleapiclient.discovery import build

    credentials, auth_mode = build_google_drive_credentials(resolve_google_drive_auth_config())
    return build("drive", "v3", credentials=credentials, cache_discovery=False), auth_mode


def build_google_drive_credentials(config: GoogleDriveAuthConfig):
    if config.mode == "oauth":
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        credentials = Credentials(
            token=None,
            refresh_token=config.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=config.client_id,
            client_secret=config.client_secret,
            scopes=DRIVE_SCOPES,
        )
        credentials.refresh(Request())
        return credentials, "oauth"
    if config.mode == "service_account":
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_info(
            json.loads(config.service_account_json),
            scopes=DRIVE_SCOPES,
        )
        return credentials, "service_account"
    raise ValueError(f"Unsupported Google Drive auth mode: {config.mode}")


def upsert_pdf(
    service,
    folder_id: str,
    local_path: Path,
    remote_name: str,
    media_factory=None,
    file_id: str | None = None,
    legacy_remote_names: tuple[str, ...] = (),
) -> tuple[str, str]:
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    if media_factory is None:
        from googleapiclient.http import MediaFileUpload

        media_factory = MediaFileUpload
    media = media_factory(str(local_path), mimetype="application/pdf", resumable=True)
    if file_id:
        service.files().update(
            fileId=file_id,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        return file_id, "updated_by_id"
    found = _find_by_name(service, folder_id, remote_name)
    if found:
        file_id = found[0]["id"]
        service.files().update(
            fileId=file_id,
            body={"name": remote_name},
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        return file_id, "updated"
    for legacy_name in legacy_remote_names:
        found = _find_by_name(service, folder_id, legacy_name)
        if found:
            file_id = found[0]["id"]
            service.files().update(
                fileId=file_id,
                body={"name": remote_name},
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            ).execute()
            return file_id, "updated_legacy_renamed"
    created = service.files().create(
        body={"name": remote_name, "parents": [folder_id]},
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return str(created["id"]), "created"


def _find_by_name(service, folder_id: str, remote_name: str) -> list[dict]:
    escaped = remote_name.replace("\\", "\\\\").replace("'", "\\'")
    query = f"name = '{escaped}' and '{folder_id}' in parents and trashed = false"
    return service.files().list(
        q=query,
        fields="files(id,name)",
        pageSize=10,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute().get("files", [])


if __name__ == "__main__":
    main()
