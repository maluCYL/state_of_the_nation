from pathlib import Path
import os
import textwrap
import math
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from scipy.stats import norm


# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ENV_FILE = BASE_DIR / ".env"
DB_TABLE = "state_of_the_nation_update"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- LOAD DATA FROM SQL DATABASE ---
def _load_env_file(env_path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs from .env without requiring python-dotenv."""
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def _database_url_from_env() -> str:
    _load_env_file()

    required_vars = ["DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_NAME"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise RuntimeError(
            "Missing database settings in .env: " + ", ".join(missing_vars)
        )

    user = quote_plus(os.environ["DB_USER"])
    password = quote_plus(os.environ["DB_PASSWORD"])
    host = os.environ["DB_HOST"]
    port = os.environ["DB_PORT"]
    database = os.environ["DB_NAME"]

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"


def _validate_table_name(table_name: str) -> None:
    parts = table_name.split(".")

    if not all(part.replace("_", "").isalnum() for part in parts):
        raise ValueError(f"Invalid SQL table name: {table_name}")


def load_data(table_name: str | None = None) -> pd.DataFrame:
    """Load State of the Nation data from the SQL database configured in .env."""
    _load_env_file()
    table_name = table_name or os.getenv("DB_TABLE", DB_TABLE)
    _validate_table_name(table_name)

    query = f"""
        SELECT
            t.indicator_id,
            t.indicator_name,
            t.value,
            t.year,
            t.geography_id,
            t.source_id,
            t.source_name
        FROM {table_name} t
    """

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError(
            "Loading data from SQL requires SQLAlchemy and PyMySQL. "
            "Install them in this notebook environment with: "
            "pip install sqlalchemy pymysql"
        ) from exc

    engine = create_engine(_database_url_from_env())

    with engine.connect() as connection:
        return pd.read_sql(query, connection)


# --- LIST AVAILABLE INDICATORS ---
def list_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["indicator_id", "indicator_name"]]
        .drop_duplicates()
        .sort_values("indicator_id")
        .reset_index(drop=True)
    )

### DEFINING NON PARAMETRIC TESTS ####

def mann_kendall_test(values: pd.Series, alpha: float = 0.05) -> dict:
    """
    Basic Mann-Kendall trend test.

    Tests whether a time-ordered series has a monotonic upward or downward trend.

    Returns
    -------
    dict with:
        s: Mann-Kendall S statistic
        z: standardized test statistic
        p_value: two-sided p-value
        trend: "increasing", "decreasing", or "no trend"
        significant: True/False
    """

    x = pd.Series(values).dropna().to_numpy()
    n = len(x)

    if n < 3:
        return {
            "s": np.nan,
            "z": np.nan,
            "p_value": np.nan,
            "trend": "not enough data",
            "significant": False,
        }

    # Mann-Kendall S statistic
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])

    # Variance of S, adjusted for ties
    unique_x, counts = np.unique(x, return_counts=True)

    tie_sum = sum(
        count * (count - 1) * (2 * count + 5)
        for count in counts
        if count > 1
    )

    var_s = (
        n * (n - 1) * (2 * n + 5) - tie_sum
    ) / 18

    if var_s == 0:
        z = 0
        p_value = 1.0
    else:
        if s > 0:
            z = (s - 1) / math.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / math.sqrt(var_s)
        else:
            z = 0

        p_value = 2 * (1 - norm.cdf(abs(z)))

    significant = p_value < alpha

    if significant and s > 0:
        trend = "Estimated Trend: increasing"
    elif significant and s < 0:
        trend = "Estimated Trend: Decreasing"
    else:
        trend = "No significant monotonic trend"

    return {
        "s": s,
        "z": z,
        "p_value": p_value,
        "trend": trend,
        "significant": significant,
    }

## ADJUSTING ANOTATION IN THE FINAL GRAPH OUTPUT

import numpy as np

def add_smart_note(
    ax,
    text: str,
    y_values,
    *,
    fontsize: int = 12,
    avoid_top_right: bool = True,
):
    """
    Place a text box in the least crowded corner of the plot.

    Parameters
    ----------
    ax : matplotlib axis
    text : str
        Text to display.
    y_values : array-like
        The plotted y-values (used to estimate where the plot is crowded).
    fontsize : int
        Font size for the note.
    avoid_top_right : bool
        Set True if the legend is in the top-right corner.
    """

    y = np.asarray(y_values, dtype=float)
    n = len(y)

    # Approximate x positions as evenly spaced across the axes
    x_norm = np.linspace(0.08, 0.92, n)

    # Normalize y-values into [0, 1] axes space
    y_min = np.nanmin(y)
    y_max = np.nanmax(y)

    if y_max == y_min:
        y_norm = np.full(n, 0.5)
    else:
        y_norm = 0.10 + 0.80 * (y - y_min) / (y_max - y_min)

    # Candidate boxes in axes coordinates:
    # (x, y, horizontalalignment, verticalalignment)
    candidates = [
        (0.02, 0.95, "left", "top"),      # top-left
        (0.98, 0.95, "right", "top"),     # top-right
        (0.02, 0.05, "left", "bottom"),   # bottom-left
        (0.98, 0.05, "right", "bottom"),  # bottom-right
    ]

    if avoid_top_right:
        candidates = [c for c in candidates if c[:2] != (0.98, 0.95)]

    # Approximate box width/height in axes coordinates
    box_width = 0.34
    box_height = 0.10 + 0.045 * max(1, text.count("\n") + 1)

    def overlap_score(candidate):
        x0, y0, ha, va = candidate

        if ha == "left":
            left, right = x0, x0 + box_width
        else:
            left, right = x0 - box_width, x0

        if va == "top":
            bottom, top = y0 - box_height, y0
        else:
            bottom, top = y0, y0 + box_height

        # Count how many points fall inside or very near the box
        pad = 0.03
        count = np.sum(
            (x_norm >= left - pad) & (x_norm <= right + pad) &
            (y_norm >= bottom - pad) & (y_norm <= top + pad)
        )

        return count

    best = min(candidates, key=overlap_score)
    x0, y0, ha, va = best

    ax.text(
        x0,
        y0,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#cccccc",
            alpha=0.9,
        ),
    )


# --- PLOT FUNCTION: LINE CHART ---

def plot_indicator(
    indicator_id: str,
    source_id: str | int | None = None,
    year_filter: str | list[str] | None = None,
    df: pd.DataFrame | None = None,
    title: str | None = None,
    note: str | None = None,
    figsize: tuple = (14, 8),
    save_path: Path | str | None = None,
    value_type: str = "count",  # "count" or "proportion"
) -> None:
    """
    Plot a line graph for a given indicator_id using year vs value.

    Analysis:
    - total percentage change over the full period
    - CAGR-style average annual percentage change
    - latest annual percentage change

    value_type:
    - "count": values are numbers of CYP
    - "proportion": values are proportions, e.g. 0.10 = 10%
    """

    if df is None:
        df = load_data()

    # Indicator filtering
    filtered = df[df["indicator_id"] == indicator_id].copy()

    # Source filtering
    if source_id is not None:
        filtered = filtered[
            filtered["source_id"].astype(str) == str(source_id)
        ].copy()

    # Year filtering
    if year_filter is not None:
        if isinstance(year_filter, str):
            year_filter = [year_filter]

        filtered = filtered[
            filtered["year"].astype(str).isin([str(y) for y in year_filter])
        ].copy()

    # Empty check
    filters = [f"indicator_id='{indicator_id}'"]

    if source_id is not None:
        filters.append(f"source_id='{source_id}'")

    if year_filter is not None:
        filters.append(f"year_filter={year_filter}")

    if filtered.empty:
        raise ValueError("No rows found for " + " and ".join(filters))

    # Clean data
    filtered["year"] = filtered["year"].astype(str)
    filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
    filtered = filtered.dropna(subset=["value"])

    # Sort by first 4-digit year if possible
    filtered["sort_year"] = filtered["year"].str.extract(r"(\d{4})")[0]
    filtered["sort_year"] = pd.to_numeric(
        filtered["sort_year"],
        errors="coerce",
    )

    filtered = filtered.sort_values(["sort_year", "year"]).reset_index(drop=True)

    if filtered.empty:
        raise ValueError(
            f"No valid numeric values found for indicator_id='{indicator_id}'"
        )

    if len(filtered) < 3:
        raise ValueError(
            f"At least three valid observations are needed for summary calculations "
            f"for indicator_id='{indicator_id}'"
        )

    # Value type handling
    if value_type not in ["count", "proportion"]:
        raise ValueError("value_type must be either 'count' or 'proportion'")

    if value_type == "proportion":
        filtered["display_value"] = filtered["value"] * 100
        y_label = "Percentage of Children and Young People (%)"
        value_formatter = lambda x: f"{x:.1f}%"
    else:
        filtered["display_value"] = filtered["value"]
        y_label = "Number of Children and Young People (CYP)"
        value_formatter = lambda x: f"{x:,.0f}"

    # Create time variable
    # Uses actual year where available, otherwise falls back to sequence order
    if filtered["sort_year"].notna().all():
        filtered["time"] = filtered["sort_year"]
        time_label = "year"
    else:
        filtered["time"] = range(len(filtered))
        time_label = "time period"

    # Indicator name fallback
    indicator_name = (
        filtered["indicator_name"].dropna().iloc[0]
        if "indicator_name" in filtered.columns
        and not filtered["indicator_name"].dropna().empty
        else indicator_id
    )

    source_name = (
        filtered["source_name"].dropna().iloc[0]
        if "source_name" in filtered.columns
        and not filtered["source_name"].dropna().empty
        else "Source not available"
    )

    plot_title = title if title else f"{indicator_name} ({indicator_id})"

    # ------------------------------------------------------------------
    # Summary calculations
    # ------------------------------------------------------------------

    start_row = filtered.iloc[0]
    previous_row = filtered.iloc[-2]
    end_row = filtered.iloc[-1]

    start_time = float(start_row["time"])
    end_time = float(end_row["time"])

    start_year_label = start_row["year"]
    end_year_label = end_row["year"]

    start_value = float(start_row["display_value"])
    previous_value = float(previous_row["display_value"])
    end_value = float(end_row["display_value"])

    periods_elapsed = end_time - start_time

    absolute_change = end_value - start_value

    if value_type == "proportion":
        total_percentage_change = absolute_change
    else:
        if start_value == 0:
            total_percentage_change = np.nan
        else:
            total_percentage_change = (absolute_change / start_value) * 100

    if value_type == "proportion" and periods_elapsed > 0:
        average_annual_change = absolute_change / periods_elapsed
    elif periods_elapsed <= 0 or start_value <= 0 or end_value < 0:
        average_annual_change = np.nan
    else:
        average_annual_change = (
            ((end_value / start_value) ** (1 / periods_elapsed)) - 1
        ) * 100

    if value_type == "proportion":
        last_year_percentage_change = end_value - previous_value
    else:
        if previous_value == 0:
            last_year_percentage_change = np.nan
        else:
            last_year_percentage_change = (
                (end_value - previous_value) / previous_value
            ) * 100

    def percentage_formatter(value):
        if np.isnan(value):
            return "n/a"
        return f"{value:+,.1f}%"

    def change_formatter(value):
        if np.isnan(value):
            return "n/a"
        if value_type == "proportion":
            return f"{value:+,.1f} pp"
        return f"{value:+,.1f}%"

    def printed_change_formatter(value):
        if np.isnan(value):
            return "n/a"
        if value_type == "proportion":
            return f"{value:+,.1f} percentage points"
        return value_formatter(value)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(f"Summary calculations for {indicator_id}: {indicator_name}")
    print("=" * 80)

    print(f"Period: {start_year_label} to {end_year_label}")
    print(f"Observations: {len(filtered)}")

    print("\nStart-to-end change")
    print("-" * 80)
    print(f"Start value: {value_formatter(start_value)}")
    print(f"End value: {value_formatter(end_value)}")
    print(f"Absolute change: {printed_change_formatter(absolute_change)}")

    if not np.isnan(total_percentage_change):
        if value_type == "proportion":
            print(
                "Total percentage-point change: "
                f"{total_percentage_change:+,.1f} percentage points"
            )
        else:
            print(f"Total percentage change: {total_percentage_change:,.1f}%")
    else:
        print("Total percentage change: not available because start value is zero")

    if not np.isnan(average_annual_change):
        if value_type == "proportion":
            print(
                "Average annual percentage-point change: "
                f"{average_annual_change:+,.2f} percentage points per {time_label}"
            )
        else:
            print(
                f"Average annual change (CAGR-style): "
                f"{average_annual_change:,.2f}% per {time_label}"
            )
    else:
        print("Average annual change: not available")

    print("\nLatest annual change")
    print("-" * 80)
    print(f"Previous value: {value_formatter(previous_value)}")
    print(
        f"Latest annual {'percentage-point' if value_type == 'proportion' else 'percentage'} change: "
        f"{change_formatter(last_year_percentage_change)}"
    )

    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.rcParams["font.family"] = "Roboto Slab"

    fig, ax = plt.subplots(figsize=figsize)
    plot_x = np.arange(len(filtered))

    # Plot observed line
    ax.plot(
        plot_x,
        filtered["display_value"],
        color="#ff4d5e",
        linewidth=3.5,
        marker="o",
        markersize=9,
        label="Observed",
    )

    # Title and labels
    ax.set_title(plot_title, fontsize=22, pad=25)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, labelpad=20)
    ax.tick_params(axis="both", labelsize=16)

    # Remove gridlines
    ax.grid(False)

    # Remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Format y-axis
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: value_formatter(x))
    )

    ax.set_xticks(plot_x)
    ax.set_xticklabels(filtered["year"], rotation=45, ha="right")

    annotation_lines = [
        f"Total change = {change_formatter(total_percentage_change)}",
        f"Avg annual change = {change_formatter(average_annual_change)}",
        f"Latest annual change = {change_formatter(last_year_percentage_change)}",
    ]
    annotation_text = "\n".join(annotation_lines)

    add_smart_note(
        ax,
        annotation_text,
        filtered["display_value"],
        fontsize=12,
        avoid_top_right=True,
    )

    # Annotate every point
    for i, row in filtered.iterrows():
        if i == len(filtered) - 1:
            xytext = (-8, 11)
            ha = "right"
        else:
            xytext = (0, 10)
            ha = "center"

        ax.annotate(
            value_formatter(row["display_value"]),
            xy=(plot_x[i], row["display_value"]),
            xytext=xytext,
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color="black",
            ha=ha,
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )

    # Footer block
    footer_lines = []

    if note:
        wrapped_note = textwrap.fill(note, width=85)
        footer_lines.append(f"Note: {wrapped_note}")
        footer_lines.append("")

    footer_lines.append(f"Source: {source_name}")
    if value_type == "proportion":
        footer_lines.append(
            "Summary method: total percentage-point change, average annual "
            "percentage-point change, and latest annual percentage-point change."
        )
    else:
        footer_lines.append(
            "Summary method: total percentage change, CAGR-style average annual "
            "change, and latest annual percentage change."
        )

    footer_text = "\n".join(footer_lines)

    fig.text(
        0.01,
        0.02,
        footer_text,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#444444",
        linespacing=1.4,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    if save_path:
        save_path = Path(save_path)

        if not save_path.is_absolute():
            save_path = OUTPUT_DIR / save_path

        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)

    plt.show()

# --- PLOT FUNCTION: DUMBBELL CHART ---

def plot_indicator_dumbbell(
    indicator_id: str,
    source_id: str | int | None = None,
    year_filter: str | list[str] | None = None,
    df: pd.DataFrame | None = None,
    title: str | None = None,
    note: str | None = None,
    figsize: tuple = (10, 6),
    save_path: Path | str | None = None,
) -> None:

    if df is None:
        df = load_data()

    filtered = df[df["indicator_id"] == indicator_id].copy()

    if source_id is not None:
        filtered = filtered[
            filtered["source_id"].astype(str) == str(source_id)
        ].copy()

    if year_filter is not None:
        if isinstance(year_filter, str):
            year_filter = [year_filter]

        filtered = filtered[
            filtered["year"].astype(str).isin([str(y) for y in year_filter])
        ].copy()

    filters = [f"indicator_id='{indicator_id}'"]

    if source_id is not None:
        filters.append(f"source_id='{source_id}'")

    if year_filter is not None:
        filters.append(f"year_filter={year_filter}")

    if filtered.empty:
        raise ValueError("No rows found for " + " and ".join(filters))

    filtered["year"] = filtered["year"].astype(str)
    filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
    filtered = filtered.dropna(subset=["value"])

    filtered["sort_year"] = filtered["year"].str.extract(r"(\d{4})")[0]
    filtered["sort_year"] = pd.to_numeric(filtered["sort_year"], errors="coerce")
    filtered = filtered.sort_values(["sort_year", "year"]).reset_index(drop=True)

    if len(filtered) != 2:
        raise ValueError("Dumbbell chart works best with exactly 2 years/time periods.")

    indicator_name = (
        filtered["indicator_name"].dropna().iloc[0]
        if "indicator_name" in filtered.columns and not filtered["indicator_name"].dropna().empty
        else indicator_id
    )

    source_name = (
        filtered["source_name"].dropna().iloc[0]
        if "source_name" in filtered.columns and not filtered["source_name"].dropna().empty
        else "Source not available"
    )

    plot_title = title if title else f"{indicator_name} ({indicator_id})"

    plt.rcParams["font.family"] = "Roboto Slab"

    fig, ax = plt.subplots(figsize=figsize)

    year_1 = filtered["year"].iloc[0]
    year_2 = filtered["year"].iloc[1]
    value_1 = filtered["value"].iloc[0]
    value_2 = filtered["value"].iloc[1]

    change = value_2 - value_1
    pct_change = change / value_1 * 100

    y_pos = 0

    # Connecting line
    ax.hlines(
        y=y_pos,
        xmin=value_1,
        xmax=value_2,
        color="#bbbbbb",
        linewidth=4,
        zorder=1,
    )

    # Points
    ax.scatter(value_1, y_pos, color="#5a97bb", s=220, zorder=2, label=year_1)
    ax.scatter(value_2, y_pos, color="#ff4d5e", s=220, zorder=3, label=year_2)

    # Value labels
    ax.annotate(
        f"{value_1/1_000:,.0f}k",
        xy=(value_1, y_pos),
        xytext=(-12, 18),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9),
    )

    ax.annotate(
        f"{value_2/1_000:,.0f}k",
        xy=(value_2, y_pos),
        xytext=(12, 18),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.9),
    )

    # Change label
    ax.text(
        0.5,
        0.78,
        f"Change: {change/1_000:+,.0f}k ({pct_change:+.1f}%)",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.95),
    )

    ax.set_title(plot_title, fontsize=22, pad=25)
    ax.set_xlabel("Number of Children and Young People (CYP, thousands)", fontsize=12, labelpad=20)

    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=14)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.ticklabel_format(style="plain", axis="x")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1_000:,.0f}k"))

    ax.legend(frameon=False, fontsize=12, loc="lower center", ncol=2)

    footer_lines = []

    if note:
        wrapped_note = textwrap.fill(note, width=85)
        footer_lines.append(f"Note: {wrapped_note}")
        footer_lines.append("")

    footer_lines.append(f"Source: {source_name}")

    fig.text(
        0.01,
        0.02,
        "\n".join(footer_lines),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#444444",
        linespacing=1.4,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if save_path:
        save_path = Path(save_path)

        if not save_path.is_absolute():
            save_path = OUTPUT_DIR / save_path

        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)

    plt.show()


# --- PLOT FUNCTION: HBAI METHODOLOGY CHANGE CHART ---

def _add_hbai_smart_note(
    ax,
    text: str,
    x_values,
    y_values,
    cutoff_x: float,
    *,
    fontsize: int = 9,
) -> None:
    """Place a note in the pre-change white area, avoiding plotted lines."""
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    points = np.column_stack([x, y])
    points_axes = ax.transAxes.inverted().transform(ax.transData.transform(points))
    cutoff_axes = ax.transAxes.inverted().transform(
        ax.transData.transform([[cutoff_x, np.nanmedian(y)]])
    )[0, 0]

    line_count = text.count("\n") + 1
    box_width = 0.23
    box_height = 0.08 + (0.038 * line_count)

    candidates = [
        (0.02, 0.95, "left", "top"),
        (0.37, 0.95, "left", "top"),
        (0.02, 0.58, "left", "top"),
        (0.37, 0.58, "left", "top"),
        (0.46, 0.42, "left", "top"),
        (0.50, 0.40, "left", "top"),
        (0.54, 0.38, "left", "top"),
        (0.02, 0.04, "left", "bottom"),
        (0.37, 0.04, "left", "bottom"),
        (0.50, 0.04, "left", "bottom"),
    ]

    def bounds(candidate):
        x0, y0, ha, va = candidate
        left = x0 if ha == "left" else x0 - box_width
        right = x0 + box_width if ha == "left" else x0
        bottom = y0 - box_height if va == "top" else y0
        top = y0 if va == "top" else y0 + box_height
        return left, right, bottom, top

    def score(candidate):
        left, right, bottom, top = bounds(candidate)
        penalty = 0.0

        if left < 0.01 or right > 0.99 or bottom < 0.04 or top > 0.98:
            penalty += 1000

        # Avoid the legend and keep the statistical note in the pre-change area.
        if left < 0.36 and top > 0.70:
            penalty += 400
        if right > cutoff_axes - 0.03:
            penalty += 1000
        if left < 0.32 and bottom < 0.22:
            penalty += 220

        expanded_left = left - 0.035
        expanded_right = right + 0.035
        expanded_bottom = bottom - 0.035
        expanded_top = top + 0.035

        for px, py in points_axes:
            inside = (
                expanded_left <= px <= expanded_right
                and expanded_bottom <= py <= expanded_top
            )
            if inside:
                penalty += 90

            dx = max(left - px, 0, px - right)
            dy = max(bottom - py, 0, py - top)
            penalty += 0.6 / (np.hypot(dx, dy) + 0.05)

        return penalty

    x0, y0, ha, va = min(candidates, key=score)

    ax.text(
        x0,
        y0,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="#444444",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="none",
            alpha=0.92,
        ),
    )


def HBAI_graphs(
    data: pd.DataFrame | Path | str | None = None,
    *,
    indicator_id: str = "CYPHPHARMPVT_2",
    source_id: str | int | None = "HBAI2026",
    dataset_id: str | int | None = None,
    geography_id: str | int = "ENG",
    pre_methodology_years: list[str] | None = None,
    post_methodology_years: list[str] | None = None,
    title: str = "Children in material deprivation and severe low income (England)",
    y_label: str = "Number of Children and Young People (millions)",
    source_name: str = "HBAI2026",
    omitted_years: list[str] | None = None,
    figsize: tuple = (10, 7),
    save_path: Path | str | None = "HBAI20262_England_methodology_change.png",
    show: bool = True,
) -> dict:
    """
    Plot an HBAI time-series graph with a visible methodology-change break.

    Summary calculations are made for the pre-methodology-change period only.
    Pass either a dataframe or a CSV path as ``data``. If ``data`` is omitted,
    the function uses ``data/HBAI20262_final.csv``.
    """
    pre_methodology_years = pre_methodology_years or [
        "10/11-12/13",
        "11/12-13/14",
        "12/13-14/15",
        "13/14-15/16",
        "14/15-16/17",
        "15/16-17/18",
        "16/17-18/19",
        "17/18-19/20",
        "18/19-20/21",
    ]
    post_methodology_years = post_methodology_years or [
        "20/21-22/23",
        "22/23-24/25",
    ]
    omitted_years = omitted_years or ["19/20-21/22"]
    plot_years = pre_methodology_years + post_methodology_years

    if data is None:
        data = BASE_DIR / "data" / "HBAI20262_final.csv"

    if isinstance(data, pd.DataFrame):
        hbai = data.copy()
    else:
        data_path = Path(data)
        if not data_path.is_absolute() and not data_path.exists():
            data_path = BASE_DIR / data_path
        hbai = pd.read_csv(data_path)

    filtered = hbai[
        hbai["indicator_id"].astype(str).eq(str(indicator_id))
        & hbai["geography_id"].astype(str).eq(str(geography_id))
        & hbai["year"].astype(str).isin(plot_years)
    ].copy()

    if source_id is not None:
        filtered = filtered[
            filtered["source_id"].astype(str).eq(str(source_id))
        ].copy()

    if dataset_id is not None:
        filtered = filtered[
            filtered["dataset_id"].astype(str).eq(str(dataset_id))
        ].copy()

    filtered["year"] = filtered["year"].astype(str)
    filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
    filtered = filtered.dropna(subset=["value"])
    filtered["year"] = pd.Categorical(
        filtered["year"],
        plot_years,
        ordered=True,
    )
    filtered = filtered.sort_values("year").reset_index(drop=True)
    filtered["year"] = filtered["year"].astype(str)

    missing_years = set(plot_years) - set(filtered["year"])
    if missing_years:
        raise ValueError(f"Missing expected HBAI years: {sorted(missing_years)}")

    filtered["x"] = np.arange(len(filtered))
    pre = filtered[filtered["year"].isin(pre_methodology_years)].copy()
    post = filtered[filtered["year"].isin(post_methodology_years)].copy()

    if len(pre) < 3:
        raise ValueError("At least three pre-methodology observations are needed")

    start_row = pre.iloc[0]
    previous_row = pre.iloc[-2]
    end_row = pre.iloc[-1]
    periods_elapsed = end_row["x"] - start_row["x"]
    absolute_change = end_row["value"] - start_row["value"]

    if start_row["value"] == 0:
        percentage_change = np.nan
    else:
        percentage_change = (absolute_change / start_row["value"]) * 100

    if periods_elapsed <= 0 or start_row["value"] <= 0 or end_row["value"] < 0:
        average_annual_change = np.nan
    else:
        average_annual_change = (
            ((end_row["value"] / start_row["value"]) ** (1 / periods_elapsed)) - 1
        ) * 100

    if previous_row["value"] == 0:
        last_year_percentage_change = np.nan
    else:
        last_year_percentage_change = (
            (end_row["value"] - previous_row["value"]) / previous_row["value"]
        ) * 100

    def percentage_formatter(value):
        if np.isnan(value):
            return "n/a"
        return f"{value:+.1f}%"

    print("\n" + "=" * 80)
    print(
        f"Summary calculations for {indicator_id}: "
        f"{geography_id}, previous methodology only"
    )
    print("=" * 80)
    print(f"Period: {start_row['year']} to {end_row['year']}")
    print(f"Observations: {len(pre)}")
    print("\nStart-to-end change")
    print("-" * 80)
    print(f"Start value: {start_row['value']:.3f}m")
    print(f"End value: {end_row['value']:.3f}m")
    print(f"Absolute change: {absolute_change:.3f}m")
    print(f"Total percentage change: {percentage_formatter(percentage_change)}")
    print(
        f"Average annual change (CAGR-style): "
        f"{percentage_formatter(average_annual_change)} per period"
    )
    print("\nLatest annual change")
    print("-" * 80)
    print(f"Previous value: {previous_row['value']:.3f}m")
    print(
        f"Latest annual percentage change: "
        f"{percentage_formatter(last_year_percentage_change)}"
    )
    print("=" * 80 + "\n")

    plt.rcParams["font.family"] = "Roboto Slab"
    fig, ax = plt.subplots(figsize=figsize)

    cutoff_x = len(pre_methodology_years) - 0.5
    ax.axvspan(cutoff_x, len(plot_years) - 0.5, color="#5a97bb", alpha=0.08, linewidth=0)
    ax.axvline(cutoff_x, color="#444444", linewidth=1.8, linestyle="--")

    ax.plot(
        pre["x"],
        pre["value"],
        color="#ff4d5e",
        linewidth=3.5,
        marker="o",
        markersize=9,
        label="Observed, previous methodology",
    )
    ax.plot(
        post["x"],
        post["value"],
        color="#ff4d5e",
        linewidth=2.6,
        linestyle="--",
        marker="o",
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2.4,
        label="Observed, new methodology",
    )

    ax.text(
        cutoff_x + 0.08,
        ax.get_ylim()[1],
        f"Methodology changed\nfrom {post_methodology_years[0]}",
        ha="left",
        va="top",
        fontsize=11,
        color="#444444",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=4),
    )

    stats_text = (
        "Pre-change only\n"
        f"Total change: {percentage_formatter(percentage_change)}\n"
        f"Avg annual change: {percentage_formatter(average_annual_change)}\n"
        f"Latest annual change: {percentage_formatter(last_year_percentage_change)}"
    )

    pre_sample_x = np.linspace(pre["x"].min(), pre["x"].max(), 80)
    post_sample_x = np.linspace(post["x"].min(), post["x"].max(), 30)
    note_x_values = np.r_[
        filtered["x"],
        pre_sample_x,
        post_sample_x,
    ]
    note_y_values = np.r_[
        filtered["value"],
        np.interp(pre_sample_x, pre["x"], pre["value"]),
        np.interp(post_sample_x, post["x"], post["value"]),
    ]
    _add_hbai_smart_note(
        ax,
        stats_text,
        note_x_values,
        note_y_values,
        cutoff_x,
        fontsize=9,
    )

    for _, row in filtered.iterrows():
        ax.annotate(
            f"{row['value']:.1f}m",
            xy=(row["x"], row["value"]),
            xytext=(0, 10),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="black",
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
        )

    ax.set_title(title, fontsize=22, pad=25)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, labelpad=20)
    ax.tick_params(axis="both", labelsize=14)
    ax.set_xticks(filtered["x"])
    ax.set_xticklabels(filtered["year"], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.1f}m"))

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper left")

    omitted_text = ", ".join(omitted_years)
    footer_text = (
        f"Note: {omitted_text} is omitted because it is not available in the "
        "source file. Values from "
        f"{post_methodology_years[0]} onward are shown after the methodology "
        "change and should not be treated as directly comparable with the "
        "earlier trend.\n"
        f"Source: {source_name}. Summary method: total percentage change, "
        "CAGR-style average annual change, and latest annual percentage "
        "change for the previous methodology period."
    )
    fig.text(
        0.01,
        0.02,
        footer_text,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#444444",
        linespacing=1.4,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 1])

    if save_path:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = OUTPUT_DIR / save_path
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "figure": fig,
        "axis": ax,
        "data": filtered,
        "trend": {
            "average_annual_change": average_annual_change,
            "absolute_change": absolute_change,
            "percentage_change": percentage_change,
            "last_year_percentage_change": last_year_percentage_change,
        },
        "save_path": save_path,
    }


# --- PLOT FUNCTION: HOUSING COSTS COMPARISON CHART ---

def housing_costs_graph(
    data: pd.DataFrame | Path | str | None = None,
    *,
    sheet_name: str = "Sheet1",
    with_children_col: str = "With Children\n0\nAll items",
    without_children_col: str = "Without Children\n0\nAll items",
    time_col: str = "time",
    start_date: str = "2010-01-01",
    end_date: str = "2025-12-31",
    with_children_label: str = "Household costs index (With Children, All Items)",
    without_children_label: str = "Household costs index (Without Children, All Items)",
    title: str = "Household costs index by household type",
    y_label: str = "Index",
    source_name: str = "Housing costs data",
    figsize: tuple = (10, 7),
    save_path: Path | str | None = "HOUSINGCOSTS_children_comparison.png",
    show: bool = True,
) -> dict:
    """
    Plot housing costs indices for households with and without children.

    The x-axis is monthly, but only December labels are shown to keep the
    chart clean. Pass either a dataframe or an Excel path as ``data``. If
    ``data`` is omitted, the function uses ``data/HOUSINGCOSTS_int.xlsx``.
    """
    if data is None:
        data = BASE_DIR / "data" / "HOUSINGCOSTS_int.xlsx"

    if isinstance(data, pd.DataFrame):
        housing = data.copy()
    else:
        data_path = Path(data)
        if not data_path.is_absolute() and not data_path.exists():
            data_path = BASE_DIR / data_path
        housing = pd.read_excel(data_path, sheet_name=sheet_name)

    required_cols = [with_children_col, without_children_col, time_col]
    missing_cols = [col for col in required_cols if col not in housing.columns]
    if missing_cols:
        raise ValueError(
            "Missing expected columns in housing costs data: "
            + ", ".join(repr(col) for col in missing_cols)
        )

    filtered = housing[required_cols].copy()
    filtered["time_label"] = filtered[time_col].astype(str).str.strip()
    filtered["time_clean"] = (
        filtered["time_label"]
        .str.replace(r"\s*\[p\]\s*$", "", regex=True)
        .str.strip()
    )
    filtered["date"] = pd.to_datetime(
        filtered["time_clean"],
        format="%b-%Y",
        errors="coerce",
    )

    filtered[with_children_col] = pd.to_numeric(
        filtered[with_children_col],
        errors="coerce",
    )
    filtered[without_children_col] = pd.to_numeric(
        filtered[without_children_col],
        errors="coerce",
    )
    filtered = filtered.dropna(
        subset=["date", with_children_col, without_children_col]
    )

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    filtered = filtered[
        filtered["date"].between(start, end, inclusive="both")
    ].sort_values("date")

    if filtered.empty:
        raise ValueError(
            f"No valid housing costs observations found between "
            f"{start_date} and {end_date}"
        )

    filtered = filtered.reset_index(drop=True)
    filtered["x"] = np.arange(len(filtered))

    december_ticks = filtered[filtered["date"].dt.month.eq(12)].copy()

    plt.rcParams["font.family"] = "Roboto Slab"
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        filtered["x"],
        filtered[with_children_col],
        color="#ff4d5e",
        linewidth=3.2,
        label=with_children_label,
    )
    ax.plot(
        filtered["x"],
        filtered[without_children_col],
        color="#5a97bb",
        linewidth=3.2,
        label=without_children_label,
    )

    ax.set_title(title, fontsize=22, pad=25)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12, labelpad=20)
    ax.tick_params(axis="both", labelsize=14)

    ax.set_xticks(december_ticks["x"])
    ax.set_xticklabels(
        december_ticks["date"].dt.strftime("Dec-%Y"),
        rotation=45,
        ha="right",
    )

    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:,.0f}"))

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper left")

    footer_text = (
        f"Source: {source_name}. Monthly values shown from "
        f"{filtered['date'].min().strftime('%b-%Y')} to "
        f"{filtered['date'].max().strftime('%b-%Y')}; x-axis labels show "
        "December only."
    )
    fig.text(
        0.01,
        0.02,
        footer_text,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#444444",
        linespacing=1.4,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])

    if save_path:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = OUTPUT_DIR / save_path
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "figure": fig,
        "axis": ax,
        "data": filtered,
        "save_path": save_path,
    }
