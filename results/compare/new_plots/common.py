from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


STRATEGY_ORDER = [
    "hpwp",
    "xanadu",
    "kraken_vomm",
    "lass",
    "rl_q",
    "hptd",
    "kpa",
    "depth_breadth",
    "oracle",
]

STRATEGY_LABELS = {
    "hpwp": "ConScale",
    "xanadu": "Xanadu",
    "kraken_vomm": "Kraken",
    "lass": "LaSS",
    "rl_q": "QLAS",
    "hptd": "HPTD",
    "kpa": "KPA",
    "depth_breadth": "DBW",
    "oracle": "Oracle",
}

STRATEGY_COLORS = {
    "hpwp": "#264653",  # 深海蓝绿
    "xanadu": "#287271",  # 墨青绿
    "kraken_vomm": "#2A9D8C",  # 湖水青
    "lass": "#5FA49A",  # 灰调青绿
    "rl_q": "#8AB07D",  # 鼠尾草绿
    "hptd": "#E9C46B",  # 暖芥末黄
    "kpa": "#F3A261",  # 杏橙色
    "depth_breadth": "#D98573",  # 柔和珊瑚粉橘
    "oracle": "#E66F51",  # 陶土橘红
}

SCENARIO_ORDER = ["low", "mid", "high"]
SCENARIO_LABELS = {"low": "LOW", "mid": "MID", "high": "HIGH"}


def setup_plot_font(plt: Any) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def load_metric_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing metric file: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(raw: str | None) -> float:
    if raw is None:
        return float("nan")
    text = str(raw).strip()
    if not text:
        return float("nan")
    return float(text)


def strategy_sort_key(strategy: str) -> tuple[int, str]:
    if strategy in STRATEGY_ORDER:
        return (STRATEGY_ORDER.index(strategy), strategy)
    return (999, strategy)


def write_table(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
