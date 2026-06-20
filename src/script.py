from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Loader
# ============================================================

def load_json_or_jsonl(path: str | Path) -> pd.DataFrame:
    """
    Load CooHistoryEventRecorder data and flatten it into one row per event.

    Supported input formats:

    1. JSON Lines file:
       Each line is one envelope or one event.

    2. JSON file:
       - one envelope:
         {
           "schema": "...",
           "timestamp": "...",
           "events": [...]
         }

       - list of envelopes:
         [
           {"events": [...]},
           {"events": [...]}
         ]

       - list of events:
         [
           {"kind": "method", ...},
           {"kind": "completion", ...}
         ]
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    objects: list[dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    loaded = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error}"
                    ) from error

                if isinstance(loaded, dict):
                    objects.append(loaded)
                elif isinstance(loaded, list):
                    objects.extend(item for item in loaded if isinstance(item, dict))

    else:
        with path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)

        if isinstance(loaded, dict):
            objects = [loaded]
        elif isinstance(loaded, list):
            objects = [item for item in loaded if isinstance(item, dict)]
        else:
            raise ValueError("Unsupported JSON structure.")

    rows: list[dict[str, Any]] = []

    for obj in objects:
        # Case 1: envelope containing events
        if "events" in obj and isinstance(obj["events"], list):
            envelope_timestamp = obj.get("timestamp")
            schema = obj.get("schema")
            recorder = obj.get("recorder")
            category = obj.get("category")
            image = obj.get("image", {}) or {}

            for event in obj["events"]:
                if not isinstance(event, dict):
                    continue

                row = dict(event)

                # Add envelope-level metadata to each event row
                row["envelopeTimestamp"] = envelope_timestamp
                row["schema"] = schema
                row["recorder"] = recorder
                row["category"] = category

                # Add useful image/session metadata
                row["imageVersion"] = image.get("imageVersion")
                row["latestUpdate"] = image.get("latestUpdate")
                row["sessionUUID"] = image.get("sessionUUID")
                row["sessionCreationTime"] = image.get("sessionCreationTime")
                row["computerUUID"] = image.get("computerUUID")

                rows.append(row)

        # Case 2: already one event
        else:
            rows.append(obj)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Normalize timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.day_name()

    if "envelopeTimestamp" in df.columns:
        df["envelopeTimestamp"] = pd.to_datetime(
            df["envelopeTimestamp"],
            errors="coerce",
        )

    # Normalize completion selected index
    if "index" in df.columns:
        df["index"] = pd.to_numeric(df["index"], errors="coerce")

    # Useful derived field for completion token length
    if "token" in df.columns:
        df["token"] = df["token"].fillna("").astype(str)
        df["tokenLength"] = df["token"].str.len()

    return df


# ============================================================
# Plot helpers
# ============================================================

def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def save_bar_chart(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    horizontal: bool = False,
) -> None:
    if data.empty:
        return

    plt.figure(figsize=(11, 6))

    if horizontal:
        data.sort_values().plot(kind="barh")
    else:
        data.plot(kind="bar")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_line_chart(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    if data.empty:
        return

    plt.figure(figsize=(11, 5))
    data.plot(kind="line", marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_histogram(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    bins: int = 20,
) -> None:
    data = data.dropna()

    if data.empty:
        return

    plt.figure(figsize=(10, 5))
    data.plot(kind="hist", bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# Basic dataset visualizations
# ============================================================

def plot_event_kinds(df: pd.DataFrame, output_dir: Path) -> None:
    if "kind" not in df.columns:
        return

    counts = df["kind"].fillna("unknown").astype(str).value_counts()

    save_bar_chart(
        counts,
        title="Recorded events by kind",
        xlabel="Event kind",
        ylabel="Count",
        output_path=output_dir / "01_event_kinds.png",
    )


def plot_event_actions(df: pd.DataFrame, output_dir: Path) -> None:
    if "event" not in df.columns:
        return

    counts = df["event"].fillna("unknown").astype(str).value_counts()

    save_bar_chart(
        counts,
        title="Recorded events by action",
        xlabel="Event action",
        ylabel="Count",
        output_path=output_dir / "02_event_actions.png",
    )


def plot_events_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    if "timestamp" not in df.columns:
        return

    timeline = (
        df.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .resample("D")
        .size()
    )

    save_line_chart(
        timeline,
        title="Events over time",
        xlabel="Date",
        ylabel="Number of events",
        output_path=output_dir / "03_events_over_time.png",
    )


def plot_events_by_hour(df: pd.DataFrame, output_dir: Path) -> None:
    if "hour" not in df.columns:
        return

    counts = df["hour"].dropna().astype(int).value_counts().sort_index()

    save_bar_chart(
        counts,
        title="Events by hour of day",
        xlabel="Hour",
        ylabel="Number of events",
        output_path=output_dir / "04_events_by_hour.png",
    )


def plot_events_by_weekday(df: pd.DataFrame, output_dir: Path) -> None:
    if "weekday" not in df.columns:
        return

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    counts = df["weekday"].dropna().astype(str).value_counts()
    counts = counts.reindex(weekday_order).dropna()

    save_bar_chart(
        counts,
        title="Events by weekday",
        xlabel="Weekday",
        ylabel="Number of events",
        output_path=output_dir / "05_events_by_weekday.png",
    )


# ============================================================
# Code-history visualizations
# ============================================================

def plot_top_selectors(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    if "selector" not in df.columns:
        return

    selectors = (
        df["selector"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.len() > 0]
        .value_counts()
        .head(top_n)
    )

    save_bar_chart(
        selectors,
        title=f"Top {top_n} method selectors in history",
        xlabel="Count",
        ylabel="Selector",
        output_path=output_dir / "06_top_method_selectors.png",
        horizontal=True,
    )


def plot_top_classes(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    if "className" not in df.columns:
        return

    classes = (
        df["className"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.len() > 0]
        .value_counts()
        .head(top_n)
    )

    save_bar_chart(
        classes,
        title=f"Top {top_n} classes in history",
        xlabel="Count",
        ylabel="Class",
        output_path=output_dir / "07_top_classes.png",
        horizontal=True,
    )


def plot_method_events_by_action(df: pd.DataFrame, output_dir: Path) -> None:
    if "kind" not in df.columns or "event" not in df.columns:
        return

    method_df = df[df["kind"].astype(str) == "method"]

    if method_df.empty:
        return

    counts = method_df["event"].fillna("unknown").astype(str).value_counts()

    save_bar_chart(
        counts,
        title="Method events by action",
        xlabel="Action",
        ylabel="Count",
        output_path=output_dir / "08_method_events_by_action.png",
    )


def plot_class_events_by_action(df: pd.DataFrame, output_dir: Path) -> None:
    if "kind" not in df.columns or "event" not in df.columns:
        return

    class_df = df[df["kind"].astype(str) == "class"]

    if class_df.empty:
        return

    counts = class_df["event"].fillna("unknown").astype(str).value_counts()

    save_bar_chart(
        counts,
        title="Class events by action",
        xlabel="Action",
        ylabel="Count",
        output_path=output_dir / "09_class_events_by_action.png",
    )


def plot_top_delta_messages(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    if "deltaMessages" not in df.columns:
        return

    messages: list[str] = []

    for value in df["deltaMessages"].dropna():
        if isinstance(value, list):
            messages.extend(str(item) for item in value)
        elif isinstance(value, str):
            cleaned = value.strip()

            if not cleaned:
                continue

            # Handles simple string versions such as "#(foo bar)" or "['foo', 'bar']"
            cleaned = (
                cleaned.replace("#(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace('"', "")
                .replace(",", " ")
            )

            messages.extend(part for part in cleaned.split() if part)

    if not messages:
        return

    counts = pd.Series(messages).value_counts().head(top_n)

    save_bar_chart(
        counts,
        title=f"Top {top_n} delta messages",
        xlabel="Count",
        ylabel="Message selector",
        output_path=output_dir / "10_top_delta_messages.png",
        horizontal=True,
    )


# ============================================================
# Completion visualizations
# ============================================================

def get_completion_df(df: pd.DataFrame) -> pd.DataFrame:
    if "kind" not in df.columns:
        return pd.DataFrame()

    return df[df["kind"].astype(str) == "completion"].copy()


def plot_top_completion_items(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    if "selectedItem" not in df.columns:
        return

    completion_df = get_completion_df(df)

    if completion_df.empty:
        return

    items = (
        completion_df["selectedItem"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.len() > 0]
        .value_counts()
        .head(top_n)
    )

    save_bar_chart(
        items,
        title=f"Top {top_n} selected completion items",
        xlabel="Count",
        ylabel="Selected item",
        output_path=output_dir / "11_top_completion_items.png",
        horizontal=True,
    )


def plot_top_tokens(df: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    if "token" not in df.columns:
        return

    completion_df = get_completion_df(df)

    if completion_df.empty:
        return

    tokens = (
        completion_df["token"]
        .dropna()
        .astype(str)
        .loc[lambda s: s.str.len() > 0]
        .value_counts()
        .head(top_n)
    )

    save_bar_chart(
        tokens,
        title=f"Top {top_n} completion tokens",
        xlabel="Count",
        ylabel="Token",
        output_path=output_dir / "12_top_completion_tokens.png",
        horizontal=True,
    )


def plot_completion_rank_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "index" not in df.columns:
        return

    completion_df = get_completion_df(df)

    if completion_df.empty:
        return

    ranks = completion_df["index"].dropna()

    save_histogram(
        ranks,
        title="Distribution of selected completion rank",
        xlabel="Selected index in completion list",
        ylabel="Number of completion selections",
        output_path=output_dir / "13_completion_rank_distribution.png",
        bins=20,
    )


def plot_completion_top1_top5_pie(df: pd.DataFrame, output_dir: Path) -> None:
    if "index" not in df.columns:
        return

    completion_df = get_completion_df(df).dropna(subset=["index"])

    if completion_df.empty:
        return

    top1 = int((completion_df["index"] == 1).sum())
    top5_not_top1 = int(((completion_df["index"] > 1) & (completion_df["index"] <= 5)).sum())
    after_top5 = int((completion_df["index"] > 5).sum())

    values = [top1, top5_not_top1, after_top5]
    labels = ["Top 1", "Top 2-5", "After top 5"]

    if sum(values) == 0:
        return

    plt.figure(figsize=(7, 7))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Selected completion position")
    plt.tight_layout()
    plt.savefig(output_dir / "14_completion_top1_top5_pie.png", dpi=200)
    plt.close()


def plot_token_length_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "tokenLength" not in df.columns:
        return

    completion_df = get_completion_df(df)

    if completion_df.empty:
        return

    lengths = completion_df["tokenLength"].dropna()

    save_histogram(
        lengths,
        title="Distribution of completion token length",
        xlabel="Token length",
        ylabel="Number of completion selections",
        output_path=output_dir / "15_token_length_distribution.png",
        bins=20,
    )


def plot_average_rank_by_token_length(df: pd.DataFrame, output_dir: Path) -> None:
    if "tokenLength" not in df.columns or "index" not in df.columns:
        return

    completion_df = get_completion_df(df).dropna(subset=["tokenLength", "index"])

    if completion_df.empty:
        return

    grouped = completion_df.groupby("tokenLength")["index"].mean()

    save_line_chart(
        grouped,
        title="Average selected rank by token length",
        xlabel="Token length",
        ylabel="Average selected rank",
        output_path=output_dir / "16_average_rank_by_token_length.png",
    )


# ============================================================
# History + completion intersection
# ============================================================

def compute_history_reuse_counts(df: pd.DataFrame) -> tuple[int, int]:
    """
    Count whether each selected completion item appeared earlier
    as a method selector.

    Returns:
        reused_count, not_reused_count
    """

    required = {"timestamp", "kind", "selector", "selectedItem"}

    if not required.issubset(set(df.columns)):
        return 0, 0

    ordered = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()

    seen_selectors: set[str] = set()
    reused_count = 0
    not_reused_count = 0

    for _, row in ordered.iterrows():
        kind = str(row.get("kind"))

        if kind == "method":
            selector = row.get("selector")

            if pd.notna(selector) and str(selector):
                seen_selectors.add(str(selector))

        elif kind == "completion":
            selected = row.get("selectedItem")

            if pd.notna(selected) and str(selected):
                if str(selected) in seen_selectors:
                    reused_count += 1
                else:
                    not_reused_count += 1

    return reused_count, not_reused_count


def plot_history_reuse_pie(df: pd.DataFrame, output_dir: Path) -> None:
    reused_count, not_reused_count = compute_history_reuse_counts(df)

    total = reused_count + not_reused_count

    if total == 0:
        return

    values = [reused_count, not_reused_count]
    labels = [
        "Appeared earlier in method history",
        "Did not appear earlier",
    ]

    plt.figure(figsize=(7, 7))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Completion selections and previous method history")
    plt.tight_layout()
    plt.savefig(output_dir / "17_history_reuse_pie.png", dpi=200)
    plt.close()


def plot_reuse_delay_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """
    For each completion selection, find the last previous method event
    with the same selector and plot the delay in minutes.
    """

    required = {"timestamp", "kind", "selector", "selectedItem"}

    if not required.issubset(set(df.columns)):
        return

    ordered = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()

    last_seen_selector_time: dict[str, pd.Timestamp] = {}
    delays_minutes: list[float] = []

    for _, row in ordered.iterrows():
        kind = str(row.get("kind"))

        if kind == "method":
            selector = row.get("selector")

            if pd.notna(selector) and str(selector):
                last_seen_selector_time[str(selector)] = row["timestamp"]

        elif kind == "completion":
            selected = row.get("selectedItem")

            if pd.notna(selected) and str(selected):
                selected = str(selected)

                if selected in last_seen_selector_time:
                    delay = row["timestamp"] - last_seen_selector_time[selected]
                    delay_minutes = delay.total_seconds() / 60

                    if delay_minutes >= 0:
                        delays_minutes.append(delay_minutes)

    if not delays_minutes:
        return

    save_histogram(
        pd.Series(delays_minutes),
        title="Delay between method history and completion reuse",
        xlabel="Delay in minutes",
        ylabel="Number of completion selections",
        output_path=output_dir / "18_reuse_delay_distribution.png",
        bins=30,
    )


def plot_cumulative_events(df: pd.DataFrame, output_dir: Path) -> None:
    if "timestamp" not in df.columns:
        return

    ordered = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()

    if ordered.empty:
        return

    cumulative = (
        ordered.set_index("timestamp")
        .resample("D")
        .size()
        .cumsum()
    )

    save_line_chart(
        cumulative,
        title="Cumulative recorded events",
        xlabel="Date",
        ylabel="Cumulative events",
        output_path=output_dir / "19_cumulative_events.png",
    )


# ============================================================
# Console summary
# ============================================================

def print_summary(df: pd.DataFrame, output_dir: Path) -> None:
    print()
    print("=== Dataset summary ===")
    print(f"Total events: {len(df)}")

    if "kind" in df.columns:
        print()
        print("Events by kind:")
        print(df["kind"].fillna("unknown").astype(str).value_counts())

    if "event" in df.columns:
        print()
        print("Events by action:")
        print(df["event"].fillna("unknown").astype(str).value_counts())

    if "index" in df.columns and "kind" in df.columns:
        completion_df = get_completion_df(df).dropna(subset=["index"])

        if not completion_df.empty:
            top1 = (completion_df["index"] == 1).mean()
            top5 = (completion_df["index"] <= 5).mean()
            mrr = (1 / completion_df["index"]).mean()
            mean_rank = completion_df["index"].mean()

            print()
            print("Completion metrics from recorded index:")
            print(f"Completion events with rank: {len(completion_df)}")
            print(f"Top-1 rate: {top1:.2%}")
            print(f"Top-5 rate: {top5:.2%}")
            print(f"MRR: {mrr:.4f}")
            print(f"Mean selected rank: {mean_rank:.2f}")

    reused_count, not_reused_count = compute_history_reuse_counts(df)

    if reused_count + not_reused_count > 0:
        reuse_rate = reused_count / (reused_count + not_reused_count)

        print()
        print("History/completion intersection:")
        print(f"Completion selections reused from previous method history: {reuse_rate:.2%}")
        print(f"Reused: {reused_count}")
        print(f"Not reused: {not_reused_count}")

    pngs = sorted(output_dir.glob("*.png"))

    print()
    print(f"PNG files created: {len(pngs)}")
    print(f"Output directory: {output_dir.resolve()}")

    for png in pngs:
        print(f"- {png.name}")


# ============================================================
# Main
# ============================================================

def run_analysis(input_path: Path, output_dir: Path, top_n: int) -> None:
    ensure_output_dir(output_dir)

    df = load_json_or_jsonl(input_path)

    if df.empty:
        print("No events found. No PNGs created.")
        return

    # Basic dataset plots
    plot_event_kinds(df, output_dir)
    plot_event_actions(df, output_dir)
    plot_events_over_time(df, output_dir)
    plot_events_by_hour(df, output_dir)
    plot_events_by_weekday(df, output_dir)

    # Code-history plots
    plot_top_selectors(df, output_dir, top_n)
    plot_top_classes(df, output_dir, top_n)
    plot_method_events_by_action(df, output_dir)
    plot_class_events_by_action(df, output_dir)
    plot_top_delta_messages(df, output_dir, top_n)

    # Completion plots
    plot_top_completion_items(df, output_dir, top_n)
    plot_top_tokens(df, output_dir, top_n)
    plot_completion_rank_distribution(df, output_dir)
    plot_completion_top1_top5_pie(df, output_dir)
    plot_token_length_distribution(df, output_dir)
    plot_average_rank_by_token_length(df, output_dir)

    # Intersection between history and completion
    plot_history_reuse_pie(df, output_dir)
    plot_reuse_delay_distribution(df, output_dir)
    plot_cumulative_events(df, output_dir)

    print_summary(df, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PNG visualizations from CooHistoryEventRecorder JSON/JSONL data."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to your Coo history JSON or JSONL file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("coo_png_output"),
        help="Directory where PNG files will be saved. Default: coo_png_output",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top items to show in ranking charts. Default: 20",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(args.input, args.output, args.top_n)


if __name__ == "__main__":
    main()