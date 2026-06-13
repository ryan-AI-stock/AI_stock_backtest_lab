from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

from backtest_lab.stock_pool_store import normalize_ticker


DEFAULT_SEED_PATH = Path("data/tw50_constituents.seed.csv")
DEFAULT_OUTPUT_PATH = Path("data/tw50_constituents.csv")
REQUIRED_OUTPUT_COLUMNS = ["effective_date", "ticker", "name", "source", "source_updated_at"]
FTSE_TW50_PDF_MARKER = b"%PDF"
FTSE_TW50_DEFAULT_URL = (
    "https://research.ftserussell.com/analytics/factsheets/Home/"
    "DownloadConstituentsWeights/?indexdetails=TW50"
)


FTSE_TW50_NAME_ALIASES: dict[str, tuple[str, str]] = {
    "acctontechnology": ("2345.TW", "智邦"),
    "advantech": ("2395.TW", "研華"),
    "alchiptechnologiesinc": ("3661.TW", "世芯-KY"),
    "asetechnologyholding": ("3711.TW", "日月光投控"),
    "asiavitalcomponents": ("3017.TW", "奇鋐"),
    "asustekcomputerinc": ("2357.TW", "華碩"),
    "caliwaybiopharmaceuticals": ("6919.TW", "康霈*"),
    "cathayfinancialholding": ("2882.TW", "國泰金"),
    "chinasteel": ("2002.TW", "中鋼"),
    "chromaate": ("2360.TW", "致茂"),
    "chunghwatelecom": ("2412.TW", "中華電"),
    "ctbcfinancialholding": ("2891.TW", "中信金"),
    "deltaelectronics": ("2308.TW", "台達電"),
    "esunfinancialholding": ("2884.TW", "玉山金"),
    "elitematerial": ("2383.TW", "台光電"),
    "evergreenmarine": ("2603.TW", "長榮"),
    "fareastonetelecommunications": ("4904.TW", "遠傳"),
    "firstfinancialholding": ("2892.TW", "第一金"),
    "formosapetrochemical": ("6505.TW", "台塑化"),
    "formosaplasticscorp": ("1301.TW", "台塑"),
    "fubonfinancialholdings": ("2881.TW", "富邦金"),
    "goldcircuitelectronics": ("2368.TW", "金像電"),
    "honhaiprecisionindustry": ("2317.TW", "鴻海"),
    "honprecision": ("7769.TW", "鴻勁"),
    "hotaimotor": ("2207.TW", "和泰車"),
    "huananfinancialholdings": ("2880.TW", "華南金"),
    "jentechprecisionindustrial": ("3653.TW", "健策"),
    "kgifinancialholding": ("2883.TW", "凱基金"),
    "kingslideworks": ("2059.TW", "川湖"),
    "kingyuanelectronics": ("2449.TW", "京元電子"),
    "larganprecision": ("3008.TW", "大立光"),
    "liteontechnology": ("2301.TW", "光寶科"),
    "mediatek": ("2454.TW", "聯發科"),
    "megafinancialholding": ("2886.TW", "兆豐金"),
    "nanyaplastics": ("1303.TW", "南亞"),
    "nanyatechnology": ("2408.TW", "南亞科"),
    "quantacomputer": ("2382.TW", "廣達"),
    "sinopacfinancialholdingscoltd": ("2890.TW", "永豐金"),
    "taiwancooperativefinancialholding": ("5880.TW", "合庫金"),
    "taiwanmobile": ("3045.TW", "台灣大"),
    "taiwansemiconductormanufacturing": ("2330.TW", "台積電"),
    "tsfinancialholding": ("2887.TW", "台新新光金"),
    "unipresidententerprises": ("1216.TW", "統一"),
    "unitedmicroelectronics": ("2303.TW", "聯電"),
    "winbondelectronics": ("2344.TW", "華邦電"),
    "wistroncorp": ("3231.TW", "緯創"),
    "wiwynn": ("6669.TW", "緯穎"),
    "yageo": ("2327.TW", "國巨*"),
    "yuantafinancialholding": ("2885.TW", "元大金"),
}


@dataclass(frozen=True)
class UpdateResult:
    output_path: str
    source: str
    effective_date: str
    row_count: int
    total_row_count: int
    used_fallback: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "source": self.source,
            "effective_date": self.effective_date,
            "row_count": self.row_count,
            "total_row_count": self.total_row_count,
            "used_fallback": self.used_fallback,
            "message": self.message,
        }


def update_tw50_constituents(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    as_of_date: str,
    source_csv: str | Path | None = None,
    source_url: str | None = None,
    seed_path: str | Path = DEFAULT_SEED_PATH,
    allow_seed_fallback: bool = True,
    min_count: int = 45,
    max_count: int = 60,
) -> UpdateResult:
    output = Path(output_path)
    source_label = ""
    used_fallback = False
    source_frame: pd.DataFrame | None = None
    source_error = ""

    try:
        source_frame, source_label = _read_primary_source(source_csv=source_csv, source_url=source_url)
    except (OSError, ValueError, URLError) as error:
        source_error = str(error)

    if source_frame is None:
        if not allow_seed_fallback:
            raise ValueError(source_error or "No TW50 constituent source provided.")
        source_frame = pd.read_csv(seed_path)
        source_label = "seed_snapshot"
        used_fallback = True

    normalized = normalize_constituent_frame(
        source_frame,
        as_of_date=as_of_date,
        source=source_label,
        min_count=min_count,
        max_count=max_count,
    )
    merged = _merge_with_existing(output, normalized)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8-sig")

    effective_date = str(normalized["effective_date"].iloc[0])
    message = (
        "TW50 constituents updated from fallback seed."
        if used_fallback
        else "TW50 constituents updated from primary source."
    )
    if source_error and used_fallback:
        message = f"{message} Primary source failed: {source_error}"
    return UpdateResult(
        output_path=str(output),
        source=source_label,
        effective_date=effective_date,
        row_count=len(normalized),
        total_row_count=len(merged),
        used_fallback=used_fallback,
        message=message,
    )


def normalize_constituent_frame(
    frame: pd.DataFrame,
    *,
    as_of_date: str,
    source: str,
    min_count: int = 45,
    max_count: int = 60,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("TW50 constituent source is empty.")
    ticker_column = _first_existing_column(frame, ["ticker", "symbol", "stock_id", "code", "證券代號", "股票代號", "代號"])
    if ticker_column is None:
        raise ValueError("TW50 constituent source missing ticker column.")
    name_column = _first_existing_column(frame, ["name", "stock_name", "security_name", "證券名稱", "股票名稱", "名稱"])
    effective_column = _first_existing_column(frame, ["effective_date", "date", "as_of_date", "report_date", "資料日期"])

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    effective_default = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    for _, row in frame.iterrows():
        raw_ticker = str(row.get(ticker_column, "")).strip()
        if not raw_ticker or raw_ticker.lower() == "nan":
            continue
        ticker = normalize_ticker(raw_ticker)
        if ticker in seen:
            continue
        raw_effective = str(row.get(effective_column, "")).strip() if effective_column else ""
        effective_date = _normalize_date(raw_effective) if raw_effective and raw_effective.lower() != "nan" else effective_default
        name = str(row.get(name_column, "")).strip() if name_column else ""
        rows.append(
            {
                "effective_date": effective_date,
                "ticker": ticker,
                "name": "" if name.lower() == "nan" else name,
                "source": source,
                "source_updated_at": effective_default,
            }
        )
        seen.add(ticker)
    if not (min_count <= len(rows) <= max_count):
        raise ValueError(f"TW50 constituent count out of range: {len(rows)}")
    return pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS).sort_values(["effective_date", "ticker"])


def _read_primary_source(*, source_csv: str | Path | None, source_url: str | None) -> tuple[pd.DataFrame | None, str]:
    if source_csv:
        path = Path(source_csv)
        if not path.exists():
            raise FileNotFoundError(f"TW50 source CSV not found: {path}")
        payload = path.read_bytes()
        if _looks_like_pdf(payload):
            return parse_ftse_tw50_pdf(payload), f"pdf:{path}"
        return pd.read_csv(io.BytesIO(payload)), f"csv:{path}"
    if source_url:
        with urlopen(source_url, timeout=30) as response:
            payload = response.read()
        if _looks_like_pdf(payload):
            return parse_ftse_tw50_pdf(payload), f"ftse_pdf:{source_url}"
        return pd.read_csv(io.BytesIO(payload)), f"url:{source_url}"
    return None, ""


def parse_ftse_tw50_pdf(payload: bytes) -> pd.DataFrame:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - exercised only when dependency is missing.
        raise ValueError("pypdf is required to parse FTSE TWSE 50 constituent PDF.") from error

    reader = PdfReader(io.BytesIO(payload))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_ftse_tw50_text(text)


def parse_ftse_tw50_text(text: str, *, min_count: int = 45, max_count: int = 60) -> pd.DataFrame:
    names = _extract_ftse_tw50_names(text, min_count=min_count, max_count=max_count)
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = _normalize_ftse_name_key(name)
        mapped = FTSE_TW50_NAME_ALIASES.get(key)
        if mapped is None:
            missing.append(name)
            continue
        ticker, chinese_name = mapped
        if ticker in seen:
            continue
        rows.append({"ticker": ticker, "name": chinese_name})
        seen.add(ticker)
    if missing:
        raise ValueError(f"Unmapped FTSE TWSE 50 constituent names: {', '.join(missing)}")
    if len(rows) != len(names):
        raise ValueError(f"FTSE TWSE 50 constituent duplicate mapping: {len(rows)} mapped from {len(names)} names")
    return pd.DataFrame(rows)


def _extract_ftse_tw50_names(text: str, *, min_count: int = 45, max_count: int = 60) -> list[str]:
    names: list[str] = []
    pending: list[str] = []
    in_table = False
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if "Country/Market" in line:
            in_table = True
            pending = []
            continue
        if not in_table:
            continue
        if line.startswith("Data Explanation") or line.startswith("Source: FTSE Russell"):
            break
        if _is_ftse_header_or_footer(line):
            continue
        match = re.match(r"^(?P<name>.+?)\s+(?P<weight><?\d+(?:\.\s*\d+)?)\s+TAIWAN$", line)
        if match:
            name_parts = pending + [match.group("name")]
            names.append(_normalize_ftse_display_name(" ".join(name_parts)))
            pending = []
            continue
        weight_only = re.match(r"^(?P<weight><?\d+(?:\.\s*\d+)?)\s+TAIWAN$", line)
        if weight_only and pending:
            names.append(_normalize_ftse_display_name(" ".join(pending)))
            pending = []
            continue
        pending.append(line)
    if not (min_count <= len(names) <= max_count):
        raise ValueError(f"FTSE TWSE 50 PDF parse expected {min_count}-{max_count} names, got {len(names)}")
    return names


def _is_ftse_header_or_footer(line: str) -> bool:
    if line in {"Constituent", "Index weight", "(%)", "Country/Market", "CORPORATE"}:
        return True
    if re.fullmatch(r"\d+", line):
        return True
    if re.fullmatch(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", line):
        return True
    if line.startswith("F TSE TWSE") or line.startswith("FTSE Russell Publications"):
        return True
    if line.startswith("Indicative Index Weight Data"):
        return True
    return False


def _normalize_ftse_display_name(value: str) -> str:
    return " ".join(value.replace(" - ", "-").split())


def _normalize_ftse_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _looks_like_pdf(payload: bytes) -> bool:
    return payload.lstrip().startswith(FTSE_TW50_PDF_MARKER)


def _merge_with_existing(output_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    if output_path.exists():
        existing = pd.read_csv(output_path)
        for column in REQUIRED_OUTPUT_COLUMNS:
            if column not in existing.columns:
                existing[column] = ""
        effective_dates = set(new_rows["effective_date"].astype(str))
        existing = existing[~existing["effective_date"].astype(str).isin(effective_dates)]
        merged = pd.concat([existing[REQUIRED_OUTPUT_COLUMNS], new_rows], ignore_index=True)
    else:
        merged = new_rows.copy()
    return merged.drop_duplicates(subset=["effective_date", "ticker"], keep="last").sort_values(
        ["effective_date", "ticker"]
    )


def _normalize_date(value: str) -> str:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ValueError(f"Invalid effective date: {value}")
    return parsed.strftime("%Y-%m-%d")


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(column).strip(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Update point-in-time Taiwan 50 constituent CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--source-csv", default="")
    parser.add_argument("--source-url", default="")
    parser.add_argument("--seed", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--status-json", default="")
    parser.add_argument("--min-count", type=int, default=45)
    parser.add_argument("--max-count", type=int, default=60)
    parser.add_argument("--allow-seed-fallback", action="store_true")
    args = parser.parse_args()

    result = update_tw50_constituents(
        output_path=args.output,
        as_of_date=args.as_of_date,
        source_csv=args.source_csv or None,
        source_url=args.source_url or None,
        seed_path=args.seed,
        allow_seed_fallback=args.allow_seed_fallback,
        min_count=args.min_count,
        max_count=args.max_count,
    )
    if args.status_json:
        status_path = Path(args.status_json)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
