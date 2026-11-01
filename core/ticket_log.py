"""CSV ticket log: one row per complaint (assignment Section 7 columns + extras)."""
from __future__ import annotations

import csv
import io
import threading
from datetime import datetime

import pandas as pd

import config

COLUMNS = [
    "ticket_id",
    "timestamp",
    "customer_name",
    "order_id",
    "complaint",
    "category",
    "urgency",
    "sentiment",
    "summary",
    "language",
    "kb_sources",
    "response",
    "confidence",
    "status",
    "escalation_reason",
    "guardrail_flags",
    "reviewed_by",
]

_lock = threading.Lock()


def _ensure_file() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not config.TICKET_LOG.exists():
        with config.TICKET_LOG.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(COLUMNS)


def next_ticket_id() -> str:
    # Count CSV rows, not file lines: replies contain line breaks.
    ids = read_log()["ticket_id"].str.extract(r"TKT-(\d+)")[0].dropna().astype(int)
    return f"TKT-{(ids.max() if len(ids) else 0) + 1:04d}"


def append_row(row: dict) -> dict:
    with _lock:
        _ensure_file()
        row = {col: row.get(col, "") for col in COLUMNS}
        row["timestamp"] = row["timestamp"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with config.TICKET_LOG.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
    return row


def update_row(ticket_id: str, **fields: str) -> None:
    """Used by the human-review screen to approve or edit escalated replies."""
    with _lock:
        df = read_log()
        mask = df["ticket_id"] == ticket_id
        for key, value in fields.items():
            df.loc[mask, key] = value
        df.to_csv(config.TICKET_LOG, index=False)


def read_log() -> pd.DataFrame:
    _ensure_file()
    return pd.read_csv(config.TICKET_LOG, dtype=str).fillna("")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="tickets")
    return buf.getvalue()
