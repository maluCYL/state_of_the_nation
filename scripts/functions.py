from pathlib import Path
import textwrap
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from scipy.stats import theilslopes, norm


# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CSV_FILE = DATA_DIR / "newstats.csv"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- LOAD DATA ---
def load_data(csv_path: Path = CSV_FILE) -> pd.DataFrame:
    """Load the CSV file from the data folder."""
    return pd.read_csv(csv_path, sep=";")


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
    # You can tweak these if your note is longer/shorter
    box_width = 0.34
    box_height = 0.16

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
    - average annual change
    - Theil-Sen robust trend estimate: non-parametric estimator. 
    - Mann-Kendall monotonic trend test: test for non-linear relationships. 

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

    if len(filtered) < 2:
        raise ValueError(
            f"At least two valid observations are needed for trend analysis "
            f"for indicator_id='{indicator_id}'"
        )

    # Value type handling
    if value_type not in ["count", "proportion"]:
        raise ValueError("value_type must be either 'count' or 'proportion'")

    if value_type == "proportion":
        filtered["display_value"] = filtered["value"] * 100
        y_label = "Percentage of Children and Young People (%)"
        slope_unit = "percentage points"
        value_formatter = lambda x: f"{x:.1f}%"
    else:
        filtered["display_value"] = filtered["value"]
        y_label = "Number of Children and Young People (CYP)"
        slope_unit = "CYP"
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
    # Trend calculations
    # ------------------------------------------------------------------

    x = filtered["time"].astype(float)
    y = filtered["display_value"].astype(float)

    start_row = filtered.iloc[0]
    end_row = filtered.iloc[-1]

    start_time = float(start_row["time"])
    end_time = float(end_row["time"])

    start_year_label = start_row["year"]
    end_year_label = end_row["year"]

    start_value = float(start_row["display_value"])
    end_value = float(end_row["display_value"])

    periods_elapsed = end_time - start_time

    if periods_elapsed == 0:
        average_annual_change = np.nan
    else:
        average_annual_change = (end_value - start_value) / periods_elapsed

    absolute_change = end_value - start_value

    if start_value == 0:
        percentage_change = np.nan
    else:
        percentage_change = (absolute_change / start_value) * 100

    # Theil-Sen regression
    theil_result = theilslopes(y, x, alpha=0.95)

    theil_slope = theil_result.slope
    theil_intercept = theil_result.intercept
    theil_low_slope = theil_result.low_slope
    theil_high_slope = theil_result.high_slope

    filtered["theil_sen_fitted"] = theil_intercept + theil_slope * x

    # Mann-Kendall trend test
    mk_result = mann_kendall_test(y)

    mk_trend = mk_result["trend"]
    mk_p_value = mk_result["p_value"]
    mk_s = mk_result["s"]
    mk_z = mk_result["z"]

    # ------------------------------------------------------------------
    # Print trend summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print(f"Trend analysis for {indicator_id}: {indicator_name}")
    print("=" * 80)

    print(f"Period: {start_year_label} to {end_year_label}")
    print(f"Observations: {len(filtered)}")

    print("\nStart-to-end change")
    print("-" * 80)
    print(f"Start value: {value_formatter(start_value)}")
    print(f"End value: {value_formatter(end_value)}")
    print(f"Absolute change: {value_formatter(absolute_change)}")

    if not np.isnan(percentage_change):
        print(f"Percentage change: {percentage_change:,.1f}%")
    else:
        print("Percentage change: not available because start value is zero")

    if not np.isnan(average_annual_change):
        print(
            f"Average annual change: "
            f"{average_annual_change:,.2f} {slope_unit} per {time_label}"
        )
    else:
        print("Average annual change: not available")

    print("\nTheil-Sen robust trend estimate")
    print("-" * 80)
    print(
        f"Theil-Sen slope: "
        f"{theil_slope:,.2f} {slope_unit} per {time_label}"
    )
    print(
        f"95% slope interval: "
        f"{theil_low_slope:,.2f} to {theil_high_slope:,.2f} "
        f"{slope_unit} per {time_label}"
    )

    print("\nMann-Kendall monotonic trend test")
    print("-" * 80)
    print(f"S statistic: {mk_s}")
    print(f"Z statistic: {mk_z:,.3f}" if not np.isnan(mk_z) else "Z statistic: n/a")
    print(
        f"p-value: {mk_p_value:,.4f}"
        if not np.isnan(mk_p_value)
        else "p-value: n/a"
    )
    print(f"{mk_trend}")

    if len(filtered) < 8:
        print(
            "\nNote: very small sample. Interpret statistical significance "
            "cautiously."
        )

    print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    plt.rcParams["font.family"] = "Roboto Slab"

    fig, ax = plt.subplots(figsize=figsize)

    # Plot observed line
    ax.plot(
        filtered["year"],
        filtered["display_value"],
        color="#ff4d5e",
        linewidth=3.5,
        marker="o",
        markersize=9,
        label="Observed",
    )

    # Plot Theil-Sen fitted line
    ax.plot(
        filtered["year"],
        filtered["theil_sen_fitted"],
        color="#5a97bb",
        linewidth=2.8,
        linestyle=":",
        label="Theil-Sen trend",
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

    # Rotate x labels
    plt.xticks(rotation=45)

    # Stats annotation
    if np.isnan(mk_p_value):
        mk_text = "Mann-Kendall p = n/a"
    else:
        mk_text = f"Mann-Kendall p = {mk_p_value:.3f}"

    annotation_text = (
        f"Avg annual change = {average_annual_change:,.2f} "
        f"{slope_unit} per {time_label}\n"
        f"Theil-Sen trend = {theil_slope:,.2f} "
        f"{slope_unit} per {time_label}\n"
        f"{mk_text}\n"
        f"{mk_trend}"
    )

    add_smart_note(
    ax,
    annotation_text,
    filtered["display_value"],
    fontsize=12,
    avoid_top_right=True,
    )

    ax.legend(frameon=False, fontsize=12)

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
            xy=(row["year"], row["display_value"]),
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
    footer_lines.append(
        "Trend method: average annual change, Theil-Sen robust trend, "
        "and Mann-Kendall monotonic trend test."
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

        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)

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