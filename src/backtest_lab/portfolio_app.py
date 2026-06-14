from __future__ import annotations

import argparse
import json
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from backtest_lab.candidate_review_decision_draft import (
    apply_candidate_review_decision_draft,
    build_candidate_review_decision_draft,
)
from backtest_lab.candidate_review_decision_store import CandidateReviewDecisionStore
from backtest_lab.config import load_config
from backtest_lab.portfolio_app_settings import (
    DEFAULT_CANDIDATE_REVIEW_BACKUP_ROOT,
    DEFAULT_CANDIDATE_REVIEW_DECISION_PATH,
    DEFAULT_GITHUB_REF,
    DEFAULT_GITHUB_REPO,
    DEFAULT_OBSERVATION_ROOT,
    DEFAULT_POOL_STORE_PATH,
    DEFAULT_SIGNAL_ROOT,
    DEFAULT_STORE_PATH,
    DEFAULT_USER_ID,
    DEFAULT_WORKFLOW_FILE,
    PORTFOLIO_SECRET_NAME,
)
from backtest_lab.portfolio_dashboard import _max_affordable_shares, build_dashboard, load_latest_signal
from backtest_lab.portfolio_github import (
    sync_portfolio_secret,
    sync_stock_pools_secret,
    trigger_report_workflow,
    trigger_stock_pool_observation_workflow,
)
from backtest_lab.portfolio_store import PortfolioStore
from backtest_lab.stock_pool_candidate_review import build_candidate_review
from backtest_lab.stock_pool_store import KNOWN_SYMBOLS, StockPoolStore


def create_handler(
    *,
    store: PortfolioStore,
    pool_store: StockPoolStore | None = None,
    candidate_decision_store: CandidateReviewDecisionStore | None = None,
    candidate_review_backup_root: str | Path = DEFAULT_CANDIDATE_REVIEW_BACKUP_ROOT,
    signal_root: str,
    observation_root: str = DEFAULT_OBSERVATION_ROOT,
    asset_types: dict[str, str],
    cost_model,
    github_repo: str = DEFAULT_GITHUB_REPO,
    workflow_file: str = DEFAULT_WORKFLOW_FILE,
    github_ref: str = DEFAULT_GITHUB_REF,
    command_runner=subprocess.run,
):
    html = Path(__file__).with_name("portfolio_app.html").read_text(encoding="utf-8")
    pool_store = pool_store or StockPoolStore(DEFAULT_POOL_STORE_PATH)
    candidate_decision_store = candidate_decision_store or CandidateReviewDecisionStore(DEFAULT_CANDIDATE_REVIEW_DECISION_PATH)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self._send(html, "text/html; charset=utf-8")
                return
            if path == "/api/pools":
                self._json(_pool_state())
                return
            if path == "/api/candidate-reviews":
                self._json(_candidate_review_state())
                return
            if path == "/api/candidate-review-decisions":
                self._json(candidate_decision_store.state())
                return
            if path == "/api/candidate-review-decision-draft":
                self._json(_candidate_review_decision_draft())
                return
            if path == "/api/observations":
                self._json(load_latest_observation_state(observation_root))
                return
            if path == "/api/state":
                user = store.get_user()
                signal = load_latest_signal(signal_root)
                self._json(build_dashboard(user, signal, asset_types, cost_model))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                user_id = str(payload.get("user_id") or DEFAULT_USER_ID)
                if path == "/api/pools":
                    pool_store.upsert_pool(payload)
                    self._json(_pool_state())
                    return
                if path == "/api/candidate-review-decisions":
                    decision = candidate_decision_store.record(payload)
                    response = candidate_decision_store.state()
                    response["recorded"] = decision
                    self._json(response)
                    return
                if path == "/api/candidate-review-decision-draft/apply":
                    self._json(_apply_candidate_review_decision_draft())
                    return
                if path == "/api/portfolio":
                    store.replace_portfolio(
                        user_id=user_id,
                        cash_twd=float(payload.get("cash_twd", 0)),
                        positions=list(payload.get("positions", [])),
                    )
                elif path == "/api/trades":
                    store.record_trade(
                        user_id=user_id,
                        trade=payload,
                        asset_types=asset_types,
                        cost_model=cost_model,
                    )
                elif path == "/api/sync-secret":
                    result = sync_portfolio_secret(store_path=store.path, repo=github_repo, runner=command_runner)
                    user = store.get_user(user_id)
                    signal = load_latest_signal(signal_root)
                    response = build_dashboard(user, signal, asset_types, cost_model)
                    response["sync_result"] = result
                    self._json(response)
                    return
                elif path == "/api/sync-secret-and-run":
                    signal = load_latest_signal(signal_root)
                    signal_date = str(payload.get("signal_date") or (signal or {}).get("signal_date") or "")
                    sync_result = sync_portfolio_secret(store_path=store.path, repo=github_repo, runner=command_runner)
                    action_result = trigger_report_workflow(
                        signal_date=signal_date,
                        repo=github_repo,
                        workflow_file=workflow_file,
                        ref=github_ref,
                        runner=command_runner,
                    )
                    user = store.get_user(user_id)
                    response = build_dashboard(user, signal, asset_types, cost_model)
                    response["sync_result"] = sync_result
                    response["action_result"] = action_result
                    self._json(response)
                    return
                elif path == "/api/sync-pools-secret":
                    result = sync_stock_pools_secret(
                        pool_store_path=pool_store.path,
                        repo=github_repo,
                        runner=command_runner,
                    )
                    response = _pool_state()
                    response["sync_result"] = result
                    self._json(response)
                    return
                elif path == "/api/sync-pools-secret-and-run":
                    signal = load_latest_signal(signal_root)
                    signal_date = str(payload.get("signal_date") or (signal or {}).get("signal_date") or "")
                    sync_result = sync_stock_pools_secret(
                        pool_store_path=pool_store.path,
                        repo=github_repo,
                        runner=command_runner,
                    )
                    action_result = trigger_stock_pool_observation_workflow(
                        signal_date=signal_date,
                        repo=github_repo,
                        ref=github_ref,
                        runner=command_runner,
                    )
                    response = _pool_state()
                    response["sync_result"] = sync_result
                    response["action_result"] = action_result
                    self._json(response)
                    return
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                user = store.get_user(user_id)
                signal = load_latest_signal(signal_root)
                self._json(build_dashboard(user, signal, asset_types, cost_model))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/pools":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                query = dict(pair.split("=", 1) for pair in parsed.query.split("&") if "=" in pair)
                pool_store.delete_pool(str(query.get("pool_id") or ""))
                self._json(_pool_state())
            except ValueError as error:
                self._json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8", status)

        def _send(self, payload: str, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = payload.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            print(f"portfolio_app: {format % args}")

    def _pool_state() -> dict:
        signal = load_latest_signal(signal_root)
        pools = pool_store.list_pools(latest_signal=signal)
        return {
            "latest_signal": signal,
            "known_symbols": KNOWN_SYMBOLS,
            "pools": pools,
            "pool_sections": {
                "official_core": [pool for pool in pools if pool.get("ui_section") == "official_core"],
                "experiment": [pool for pool in pools if pool.get("ui_section") == "experiment"],
                "legacy": [pool for pool in pools if pool.get("ui_section") == "legacy"],
            },
        }

    def _candidate_review_state() -> dict:
        signal = load_latest_signal(signal_root)
        signal_date = str((signal or {}).get("signal_date") or "")
        pools = pool_store.list_pools(latest_signal=signal)
        if not signal_date:
            return {
                "status": "missing_signal",
                "message": "尚未讀到最新模型訊號日，無法產生月頻候選審核狀態。",
                "signal_date": "",
                "reviews": [],
            }
        reviews = [
            build_candidate_review(pool, signal_date=signal_date, resolved_symbols=pool.get("resolved_symbols") or [])
            for pool in pools
            if pool.get("ui_section") == "official_core"
        ]
        return {
            "status": "ready",
            "signal_date": signal_date,
            "reviews": reviews,
        }

    def _candidate_review_decision_draft() -> dict:
        signal = load_latest_signal(signal_root)
        pools = pool_store.list_pools(latest_signal=signal)
        decision_state = candidate_decision_store.state()
        return build_candidate_review_decision_draft(
            pools=pools,
            decisions=list(decision_state.get("decisions") or []),
        )

    def _apply_candidate_review_decision_draft() -> dict:
        signal = load_latest_signal(signal_root)
        pools = pool_store.list_pools(latest_signal=signal)
        decision_state = candidate_decision_store.state()
        return apply_candidate_review_decision_draft(
            pools=pools,
            decisions=list(decision_state.get("decisions") or []),
            backup_root=candidate_review_backup_root,
        )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private best-strategy portfolio workspace.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--store", default=DEFAULT_STORE_PATH)
    parser.add_argument("--pool-store", default=DEFAULT_POOL_STORE_PATH)
    parser.add_argument("--candidate-review-decisions", default=DEFAULT_CANDIDATE_REVIEW_DECISION_PATH)
    parser.add_argument("--candidate-review-backup-root", default=DEFAULT_CANDIDATE_REVIEW_BACKUP_ROOT)
    parser.add_argument("--observation-root", default=DEFAULT_OBSERVATION_ROOT)
    parser.add_argument("--signal-root", default=DEFAULT_SIGNAL_ROOT)
    parser.add_argument("--config", default="configs/ep05_universe.json")
    parser.add_argument("--group-id", default="group_c_0050_00631l_plus_mega_caps")
    parser.add_argument("--github-repo", default=DEFAULT_GITHUB_REPO)
    parser.add_argument("--workflow-file", default=DEFAULT_WORKFLOW_FILE)
    parser.add_argument("--github-ref", default=DEFAULT_GITHUB_REF)
    args = parser.parse_args()

    config = load_config(args.config)
    group = config.group_by_id(args.group_id)
    asset_types = {asset.ticker: asset.asset_type for asset in group.assets}
    handler = create_handler(
        store=PortfolioStore(args.store),
        pool_store=StockPoolStore(args.pool_store),
        candidate_decision_store=CandidateReviewDecisionStore(args.candidate_review_decisions),
        candidate_review_backup_root=args.candidate_review_backup_root,
        signal_root=args.signal_root,
        observation_root=args.observation_root,
        asset_types=asset_types,
        cost_model=config.cost_model,
        github_repo=args.github_repo,
        workflow_file=args.workflow_file,
        github_ref=args.github_ref,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"PORTFOLIO_APP_URL=http://{args.host}:{args.port}")
    server.serve_forever()


def load_latest_observation_state(root: str | Path) -> dict:
    base = Path(root)
    manifests = sorted(base.glob("*/stock_pool_observation_manifest.json"))
    if not manifests:
        return {
            "status": "missing",
            "message": "尚未產出股票池觀察結果。",
            "manifest": None,
        }
    latest = manifests[-1]
    try:
        manifest = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {
            "status": "error",
            "message": f"股票池觀察 manifest 讀取失敗：{error}",
            "manifest_path": str(latest),
            "manifest": None,
        }
    return {
        "status": "ready",
        "manifest_path": str(latest),
        "manifest": manifest,
    }


if __name__ == "__main__":
    main()
