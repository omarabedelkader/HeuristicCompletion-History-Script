from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


def load_json_or_jsonl(path: str | Path) -> pd.DataFrame:
    """
    Load CooHistoryEventRecorder data.

    Supports:
    - JSON Lines: one envelope/event per line
    - JSON file: one envelope, list of envelopes, or list of events

    It flattens envelopes like:
        { ..., "events": [event1, event2, ...] }

    into one row per event.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    objects: list[dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    objects.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_number}: {e}") from e
    else:
        with path.open("r", encoding="utf-8") as f:
            content = json.load(f)

        if isinstance(content, list):
            objects = content
        elif isinstance(content, dict):
            objects = [content]
        else:
            raise ValueError("Unsupported JSON structure.")

    rows: list[dict[str, Any]] = []

    for obj in objects:
        # Case 1: envelope with events
        if isinstance(obj, dict) and "events" in obj:
            envelope_timestamp = obj.get("timestamp")
            schema = obj.get("schema")
            recorder = obj.get("recorder")
            category = obj.get("category")
            image = obj.get("image", {}) or {}

            for event in obj.get("events", []):
                if not isinstance(event, dict):
                    continue

                row = dict(event)
                row["envelopeTimestamp"] = envelope_timestamp
                row["schema"] = schema
                row["recorder"] = recorder
                row["category"] = category

                # Keep useful image/session metadata
                row["imageVersion"] = image.get("imageVersion")
                row["sessionUUID"] = image.get("sessionUUID")
                row["computerUUID"] = image.get("computerUUID")

                rows.append(row)

        # Case 2: already a single event
        elif isinstance(obj, dict):
            rows.append(obj)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Normalize timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "envelopeTimestamp" in df.columns:
        df["envelopeTimestamp"] = pd.to_datetime(df["envelopeTimestamp"], errors="coerce")

    # Normalize completion index
    if "index" in df.columns:
        df["index"] = pd.to_numeric(df["index"], errors="coerce")

    # Useful derived fields
    if "timestamp" in df.columns:
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.day_name()

    if "token" in df.columns:
        df["tokenLength"] = df["token"].fillna("").astype(str).str.len()

    return df


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def plot_event_kinds(df: pd.DataFrame, output_dir: Path) -> None:
    if "kind" not in df.columns:
        return

    counts = df["kind"].fillna("unknown").value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Recorded events by kind")
    plt.xlabel("Event kind")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "event_kinds.png", dpi=200)
    plt.close()


def plot_events_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    if "timestamp" not in df.columns:
        return

    timeline = (
        df.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .resample("D")
        .size()
    )

    if timeline.empty:
        return

    plt.figure(figsize=(10, 5))
    timeline.plot(kind="line", marker="o")
    plt.title("Events over time")
    plt.xlabel("Date")
    plt.ylabel("Number of events")
    plt.tight_layout()
    plt.savefig(output_dir / "events_over_time.png", dpi=200)
    plt.close()


def plot_top_selectors(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    if "selector" not in df.columns:
        return

    selectors = df["selector"].dropna().astype(str).value_counts().head(top_n)

    if selectors.empty:
        return

    plt.figure(figsize=(10, 6))
    selectors.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} method selectors")
    plt.xlabel("Count")
    plt.ylabel("Selector")
    plt.tight_layout()
    plt.savefig(output_dir / "top_selectors.png", dpi=200)
    plt.close()


def plot_top_classes(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    if "className" not in df.columns:
        return

    classes = df["className"].dropna().astype(str).value_counts().head(top_n)

    if classes.empty:
        return

    plt.figure(figsize=(10, 6))
    classes.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} classes in history")
    plt.xlabel("Count")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(output_dir / "top_classes.png", dpi=200)
    plt.close()


def plot_completion_rank_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "kind" not in df.columns or "index" not in df.columns:
        return

    completions = df[df["kind"] == "completion"].dropna(subset=["index"])

    if completions.empty:
        return

    plt.figure(figsize=(8, 5))
    completions["index"].plot(kind="hist", bins=20)
    plt.title("Distribution of selected completion rank")
    plt.xlabel("Selected index in completion list")
    plt.ylabel("Number of selections")
    plt.tight_layout()
    plt.savefig(output_dir / "completion_rank_distribution.png", dpi=200)
    plt.close()


def plot_top_completion_items(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    if "selectedItem" not in df.columns:
        return

    items = df["selectedItem"].dropna().astype(str).value_counts().head(top_n)

    if items.empty:
        return

    plt.figure(figsize=(10, 6))
    items.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} selected completion items")
    plt.xlabel("Count")
    plt.ylabel("Selected item")
    plt.tight_layout()
    plt.savefig(output_dir / "top_completion_items.png", dpi=200)
    plt.close()


def plot_events_by_hour(df: pd.DataFrame, output_dir: Path) -> None:
    if "hour" not in df.columns:
        return

    by_hour = df["hour"].dropna().astype(int).value_counts().sort_index()

    if by_hour.empty:
        return

    plt.figure(figsize=(10, 5))
    by_hour.plot(kind="bar")
    plt.title("Events by hour of day")
    plt.xlabel("Hour")
    plt.ylabel("Number of events")
    plt.tight_layout()
    plt.savefig(output_dir / "events_by_hour.png", dpi=200)
    plt.close()


def compute_completion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple completion-quality metrics.

    Important:
    This uses the recorded selected index.
    If index = 1, the selected item was first in the completion list.
    If index <= 5, the selected item was in top 5.
    """
    if "kind" not in df.columns or "index" not in df.columns:
        return pd.DataFrame()

    completions = df[df["kind"] == "completion"].dropna(subset=["index"]).copy()

    if completions.empty:
        return pd.DataFrame()

    completions["top1"] = completions["index"] == 1
    completions["top5"] = completions["index"] <= 5
    completions["mrr"] = 1 / completions["index"]

    metrics = {
        "completion_events": len(completions),
        "top1_accuracy": completions["top1"].mean(),
        "top5_accuracy": completions["top5"].mean(),
        "mean_rank": completions["index"].mean(),
        "median_rank": completions["index"].median(),
        "mrr": completions["mrr"].mean(),
    }

    return pd.DataFrame([metrics])


def compute_history_reuse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple history-reuse analysis.

    Question:
    When a completion item is selected, did that same selector appear earlier
    in method history?

    This is useful for the paper:
    "Does development history help code completion?"
    """
    required = {"timestamp", "kind", "selector", "selectedItem"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()

    seen_selectors: set[str] = set()
    rows = []

    for _, row in df.iterrows():
        kind = row.get("kind")

        if kind == "method":
            selector = row.get("selector")
            if pd.notna(selector):
                seen_selectors.add(str(selector))

        elif kind == "completion":
            selected = row.get("selectedItem")
            if pd.notna(selected):
                selected = str(selected)
                rows.append({
                    "timestamp": row.get("timestamp"),
                    "token": row.get("token"),
                    "selectedItem": selected,
                    "was_in_previous_method_history": selected in seen_selectors,
                    "previous_method_history_size": len(seen_selectors),
                })

    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python analyze_coo_history.py coo_history_events.jsonl")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("coo_analysis_output")
    output_dir.mkdir(exist_ok=True)

    df = load_json_or_jsonl(input_path)

    if df.empty:
        print("No events found.")
        sys.exit(0)

    # Save cleaned flat dataset
    df.to_csv(output_dir / "clean_events.csv", index=False)

    print("\n=== Dataset overview ===")
    print(f"Rows/events: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "kind" in df.columns:
        print("\n=== Events by kind ===")
        print(df["kind"].fillna("unknown").value_counts())

    if "event" in df.columns:
        print("\n=== Events by action ===")
        print(df["event"].fillna("unknown").value_counts())

    # Summary tables
    if "kind" in df.columns:
        save_table(
            df["kind"].fillna("unknown").value_counts().to_frame("count"),
            output_dir / "event_kinds.csv",
        )

    if "selector" in df.columns:
        save_table(
            df["selector"].dropna().astype(str).value_counts().head(50).to_frame("count"),
            output_dir / "top_selectors.csv",
        )

    if "className" in df.columns:
        save_table(
            df["className"].dropna().astype(str).value_counts().head(50).to_frame("count"),
            output_dir / "top_classes.csv",
        )

    if "selectedItem" in df.columns:
        save_table(
            df["selectedItem"].dropna().astype(str).value_counts().head(50).to_frame("count"),
            output_dir / "top_completion_items.csv",
        )

    # Completion metrics
    metrics = compute_completion_metrics(df)
    if not metrics.empty:
        print("\n=== Completion metrics ===")
        print(metrics.T)
        metrics.to_csv(output_dir / "completion_metrics.csv", index=False)

    # History reuse
    reuse = compute_history_reuse(df)
    if not reuse.empty:
        reuse.to_csv(output_dir / "history_reuse.csv", index=False)
        reuse_rate = reuse["was_in_previous_method_history"].mean()
        print("\n=== History reuse ===")
        print(f"Completion selected item already appeared in previous method history: {reuse_rate:.2%}")

    # Figures
    plot_event_kinds(df, output_dir)
    plot_events_over_time(df, output_dir)
    plot_top_selectors(df, output_dir)
    plot_top_classes(df, output_dir)
    plot_completion_rank_distribution(df, output_dir)
    plot_top_completion_items(df, output_dir)
    plot_events_by_hour(df, output_dir)

    print(f"\nDone. Results saved in: {output_dir.resolve()}")
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


def load_json_or_jsonl(path: str | Path) -> pd.DataFrame:
    """
    Load CooHistoryEventRecorder data.

    Supports:
    - JSON Lines: one envelope/event per line
    - JSON file: one envelope, list of envelopes, or list of events

    It flattens envelopes like:
        { ..., "events": [event1, event2, ...] }

    into one row per event.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    objects: list[dict[str, Any]] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    objects.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_number}: {e}") from e
    else:
        with path.open("r", encoding="utf-8") as f:
            content = json.load(f)

        if isinstance(content, list):
            objects = content
        elif isinstance(content, dict):
            objects = [content]
        else:
            raise ValueError("Unsupported JSON structure.")

    rows: list[dict[str, Any]] = []

    for obj in objects:
        # Case 1: envelope with events
        if isinstance(obj, dict) and "events" in obj:
            envelope_timestamp = obj.get("timestamp")
            schema = obj.get("schema")
            recorder = obj.get("recorder")
            category = obj.get("category")
            image = obj.get("image", {}) or {}

            for event in obj.get("events", []):
                if not isinstance(event, dict):
                    continue

                row = dict(event)
                row["envelopeTimestamp"] = envelope_timestamp
                row["schema"] = schema
                row["recorder"] = recorder
                row["category"] = category

                # Keep useful image/session metadata
                row["imageVersion"] = image.get("imageVersion")
                row["sessionUUID"] = image.get("sessionUUID")
                row["computerUUID"] = image.get("computerUUID")

                rows.append(row)

        # Case 2: already a single event
        elif isinstance(obj, dict):
            rows.append(obj)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Normalize timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "envelopeTimestamp" in df.columns:
        df["envelopeTimestamp"] = pd.to_datetime(df["envelopeTimestamp"], errors="coerce")

    # Normalize completion index
    if "index" in df.columns:
        df["index"] = pd.to_numeric(df["index"], errors="coerce")

    # Useful derived fields
    if "timestamp" in df.columns:
        df["date"] = df["timestamp"].dt.date
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.day_name()

    if "token" in df.columns:
        df["tokenLength"] = df["token"].fillna("").astype(str).str.len()

    return df


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def plot_event_kinds(df: pd.DataFrame, output_dir: Path) -> None:
    if "kind" not in df.columns:
        return

    counts = df["kind"].fillna("unknown").value_counts()

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.title("Recorded events by kind")
    plt.xlabel("Event kind")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "event_kinds.png", dpi=200)
    plt.close()


def plot_events_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    if "timestamp" not in df.columns:
        return

    timeline = (
        df.dropna(subset=["timestamp"])
        .set_index("timestamp")
        .resample("D")
        .size()
    )

    if timeline.empty:
        return

    plt.figure(figsize=(10, 5))
    timeline.plot(kind="line", marker="o")
    plt.title("Events over time")
    plt.xlabel("Date")
    plt.ylabel("Number of events")
    plt.tight_layout()
    plt.savefig(output_dir / "events_over_time.png", dpi=200)
    plt.close()


def plot_top_selectors(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    if "selector" not in df.columns:
        return

    selectors = df["selector"].dropna().astype(str).value_counts().head(top_n)

    if selectors.empty:
        return

    plt.figure(figsize=(10, 6))
    selectors.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} method selectors")
    plt.xlabel("Count")
    plt.ylabel("Selector")
    plt.tight_layout()
    plt.savefig(output_dir / "top_selectors.png", dpi=200)
    plt.close()


def plot_top_classes(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    if "className" not in df.columns:
        return

    classes = df["className"].dropna().astype(str).value_counts().head(top_n)

    if classes.empty:
        return

    plt.figure(figsize=(10, 6))
    classes.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} classes in history")
    plt.xlabel("Count")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(output_dir / "top_classes.png", dpi=200)
    plt.close()


def plot_completion_rank_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "kind" not in df.columns or "index" not in df.columns:
        return

    completions = df[df["kind"] == "completion"].dropna(subset=["index"])

    if completions.empty:
        return

    plt.figure(figsize=(8, 5))
    completions["index"].plot(kind="hist", bins=20)
    plt.title("Distribution of selected completion rank")
    plt.xlabel("Selected index in completion list")
    plt.ylabel("Number of selections")
    plt.tight_layout()
    plt.savefig(output_dir / "completion_rank_distribution.png", dpi=200)
    plt.close()


def plot_top_completion_items(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    if "selectedItem" not in df.columns:
        return

    items = df["selectedItem"].dropna().astype(str).value_counts().head(top_n)

    if items.empty:
        return

    plt.figure(figsize=(10, 6))
    items.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} selected completion items")
    plt.xlabel("Count")
    plt.ylabel("Selected item")
    plt.tight_layout()
    plt.savefig(output_dir / "top_completion_items.png", dpi=200)
    plt.close()


def plot_events_by_hour(df: pd.DataFrame, output_dir: Path) -> None:
    if "hour" not in df.columns:
        return

    by_hour = df["hour"].dropna().astype(int).value_counts().sort_index()

    if by_hour.empty:
        return

    plt.figure(figsize=(10, 5))
    by_hour.plot(kind="bar")
    plt.title("Events by hour of day")
    plt.xlabel("Hour")
    plt.ylabel("Number of events")
    plt.tight_layout()
    plt.savefig(output_dir / "events_by_hour.png", dpi=200)
    plt.close()


def compute_completion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple completion-quality metrics.

    Important:
    This uses the recorded selected index.
    If index = 1, the selected item was first in the completion list.
    If index <= 5, the selected item was in top 5.
    """
    if "kind" not in df.columns or "index" not in df.columns:
        return pd.DataFrame()

    completions = df[df["kind"] == "completion"].dropna(subset=["index"]).copy()

    if completions.empty:
        return pd.DataFrame()

    completions["top1"] = completions["index"] == 1
    completions["top5"] = completions["index"] <= 5
    completions["mrr"] = 1 / completions["index"]

    metrics = {
        "completion_events": len(completions),
        "top1_accuracy": completions["top1"].mean(),
        "top5_accuracy": completions["top5"].mean(),
        "mean_rank": completions["index"].mean(),
        "median_rank": completions["index"].median(),
        "mrr": completions["mrr"].mean(),
    }

    return pd.DataFrame([metrics])


def compute_history_reuse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Very simple history-reuse analysis.

    Question:
    When a completion item is selected, did that same selector appear earlier
    in method history?

    This is useful for the paper:
    "Does development history help code completion?"
    """
    required = {"timestamp", "kind", "selector", "selectedItem"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").copy()

    seen_selectors: set[str] = set()
    rows = []

    for _, row in df.iterrows():
        kind = row.get("kind")

        if kind == "method":
            selector = row.get("selector")
            if pd.notna(selector):
                seen_selectors.add(str(selector))

        elif kind == "completion":
            selected = row.get("selectedItem")
            if pd.notna(selected):
                selected = str(selected)
                rows.append({
                    "timestamp": row.get("timestamp"),
                    "token": row.get("token"),
                    "selectedItem": selected,
                    "was_in_previous_method_history": selected in seen_selectors,
                    "previous_method_history_size": len(seen_selectors),
                })

    return pd.DataFrame(rows)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python analyze_coo_history.py coo_history_events.jsonl")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path("coo_analysis_output")
    output_dir.mkdir(exist_ok=True)

    df = load_json_or_jsonl(input_path)

    if df.empty:
        print("No events found.")
        sys.exit(0)

    # Save cleaned flat dataset
    df.to_csv(output_dir / "clean_events.csv", index=False)

    print("\n=== Dataset overview ===")
    print(f"Rows/events: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    if "kind" in df.columns:
        print("\n=== Events by kind ===")
        print(df["kind"].fillna("unknown").value_counts())

    if "event" in df.columns:
        print("\n=== Events by action ===")
        print(df["event"].fillna("unknown").value_counts())

    # Summary tables
    if "kind" in df.columns:
        save_table(
            df["kind"].fillna("unknown").value_counts().to_frame("count"),
            output_dir / "event_kinds.csv",
        )

    if "selector" in df.columns:
        save_table(
            df["selector"].dropna().astype(str).value_counts().head(50).to_frame("count"),
            output_dir / "top_selectors.csv",
        )

    if "className" in df.columns:
        save_table(
            df["className"].dropna().astype(str).value_counts().head(50).to_frame("count"),
            output_dir / "top_classes.csv",
        )

    if "selectedItem" in df.columns:
        save_table(
            df["selectedItem"].dropna().astype(str).value_counts().head(50).to_frame("count"),
            output_dir / "top_completion_items.csv",
        )

    # Completion metrics
    metrics = compute_completion_metrics(df)
    if not metrics.empty:
        print("\n=== Completion metrics ===")
        print(metrics.T)
        metrics.to_csv(output_dir / "completion_metrics.csv", index=False)

    # History reuse
    reuse = compute_history_reuse(df)
    if not reuse.empty:
        reuse.to_csv(output_dir / "history_reuse.csv", index=False)
        reuse_rate = reuse["was_in_previous_method_history"].mean()
        print("\n=== History reuse ===")
        print(f"Completion selected item already appeared in previous method history: {reuse_rate:.2%}")

    # Figures
    plot_event_kinds(df, output_dir)
    plot_events_over_time(df, output_dir)
    plot_top_selectors(df, output_dir)
    plot_top_classes(df, output_dir)
    plot_completion_rank_distribution(df, output_dir)
    plot_top_completion_items(df, output_dir)
    plot_events_by_hour(df, output_dir)

    print(f"\nDone. Results saved in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()