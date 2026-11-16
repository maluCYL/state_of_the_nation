from pathlib import Path
import json
import os
import textwrap
import math
import re
import urllib.request
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex, to_rgb
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ENV_FILE = BASE_DIR / ".env"
DB_TABLE = "state_of_the_nation_update"
OBSERVATIONS_TABLE = "observations"
ENGLAND_REGIONS_GEOJSON_URL = (
    "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/electoral/gb/eer.json"
)
ENGLAND_REGIONS_GEOJSON_FILE = DATA_DIR / "england_regions.geojson"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# --- BRAND COLOURS ---
BRAND_RED = "#ff4d5e"
BRAND_CREAM = "#fff5d2"
BRAND_BLUE = "#b7d6ea"
BRAND_CHARCOAL = "#333333"
BRAND_WHITE = "#ffffff"
HEATMAP_BACKGROUND = "#fffdf5"
ENGLAND_REGIONS_FIGSIZE = (8, 14)
ENGLAND_REGIONS_FONT_SIZES = {
    "title": 18,
    "year": 14,
    "region_name": 14,
    "region_value": 13,
    "colorbar_label": 13,
    "colorbar_ticks": 12,
    "footer": 11,
}
# Label offsets in British National Grid metres (east, north).
REGION_LABEL_OFFSETS = {
    "E12000007": (-15_000, 25_000),  # London
    "E12000008": (12_000, -22_000),  # South East
    "E12000002": (0, 20_000), # North West
}
DEFAULT_LINE_COLOURS = [
    BRAND_RED,
    "#5a97bb",
    "#CBBC87",
]


def _mix_colours(colour_a: str, colour_b: str, weight_b: float) -> str:
    """Blend two hex colours, where weight_b controls the share of colour_b."""
    rgb_a = np.array(to_rgb(colour_a))
    rgb_b = np.array(to_rgb(colour_b))
    mixed = (rgb_a * (1 - weight_b)) + (rgb_b * weight_b)
    return to_hex(mixed)


def _brand_line_colours(n_colours: int) -> list[str]:
    """
    Build muted, readable line colours from the project brand palette.

    The first three colours are fixed to the project's established chart
    defaults. Additional colours are generated from the brand palette, avoiding
    white and darkening very pale tones so they remain visible as plotted lines.
    """
    if n_colours < 1:
        return []

    extension_colours = [
        _mix_colours(BRAND_CHARCOAL, BRAND_CREAM, 0.28),
        _mix_colours(BRAND_RED, BRAND_CHARCOAL, 0.30),
        _mix_colours(BRAND_BLUE, BRAND_CHARCOAL, 0.48),
        _mix_colours(BRAND_CREAM, BRAND_CHARCOAL, 0.42),
        _mix_colours(BRAND_RED, BRAND_CREAM, 0.34),
        _mix_colours(BRAND_BLUE, BRAND_RED, 0.22),
        _mix_colours(BRAND_CHARCOAL, BRAND_BLUE, 0.40),
    ]
    core_colours = DEFAULT_LINE_COLOURS + extension_colours

    if n_colours <= len(core_colours):
        return core_colours[:n_colours]

    colours = core_colours.copy()
    blend_pairs = [
        (BRAND_RED, BRAND_CHARCOAL),
        (BRAND_BLUE, BRAND_CHARCOAL),
        (BRAND_CREAM, BRAND_CHARCOAL),
        (BRAND_RED, BRAND_BLUE),
        (BRAND_CREAM, BRAND_RED),
        (BRAND_BLUE, BRAND_CREAM),
    ]
    blend_weights = [0.18, 0.30, 0.42, 0.54, 0.66]

    for base, target in blend_pairs:
        for weight in blend_weights:
            colour = _mix_colours(base, target, weight)

            if colour.lower() not in {c.lower() for c in colours}:
                colours.append(colour)

            if len(colours) == n_colours:
                return colours

    return colours[:n_colours]


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

        os.environ[key] = value


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def _database_connect_args_from_env() -> dict:
    _load_env_file()

    if _env_bool("DB_SSL_DISABLED"):
        return {}

    host = os.getenv("DB_HOST", "")
    ssl_enabled = _env_bool(
        "DB_SSL",
        default=host.endswith(".mysql.database.azure.com"),
    )

    if not ssl_enabled:
        return {}

    ssl_config = {}
    ssl_ca = os.getenv("DB_SSL_CA")

    if ssl_ca:
        ssl_config["ca"] = str(Path(ssl_ca).expanduser())
        ssl_config["check_hostname"] = _env_bool("DB_SSL_VERIFY_IDENTITY", True)
        ssl_config["verify_mode"] = _env_bool("DB_SSL_VERIFY_CERT", True)
    else:
        ssl_config["check_hostname"] = False
        ssl_config["verify_mode"] = False

    return {"ssl": ssl_config}


def _validate_table_name(table_name: str) -> None:
    parts = table_name.split(".")

    if not all(part.replace("_", "").isalnum() for part in parts):
        raise ValueError(f"Invalid SQL table name: {table_name}")


def _create_db_engine():
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise ImportError(
            "Loading data from SQL requires SQLAlchemy and PyMySQL. "
            "Install them in this notebook environment with: "
            "pip install sqlalchemy pymysql"
        ) from exc

    return create_engine(
        _database_url_from_env(),
        connect_args=_database_connect_args_from_env(),
    )


def _read_sql_query(query: str) -> pd.DataFrame:
    _load_env_file()

    with _create_db_engine().connect() as connection:
        return pd.read_sql(query, connection)


def load_data(table_name: str | None = None) -> pd.DataFrame:
    """Load State of the Nation data from the SQL database configured in .env."""
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

    return _read_sql_query(query)


def load_observations(table_name: str | None = None) -> pd.DataFrame:
    """
    Load observation-level data joined with indicator, geography, and source metadata.

    This is the primary table for sub-national values such as English regions and
    local authorities. ``state_of_the_nation_update`` remains available through
    ``load_data()`` for the aggregated national reporting view.
    """
    table_name = table_name or os.getenv("DB_OBSERVATIONS_TABLE", OBSERVATIONS_TABLE)
    _validate_table_name(table_name)

    query = f"""
        SELECT
            o.observation_id,
            o.indicator_id,
            i.indicator_name,
            i.indicator_type,
            i.unit,
            o.source_id,
            s.source_name,
            o.dataset_id,
            o.dataset_key,
            o.year,
            o.value,
            o.geography_id,
            g.geography_name,
            g.geography_type
        FROM {table_name} o
        LEFT JOIN indicators i
            ON i.indicator_id = o.indicator_id
        LEFT JOIN geographies g
            ON g.geography_id = o.geography_id
        LEFT JOIN sources s
            ON s.source_id = o.source_id
    """

    return _read_sql_query(query)


def list_geographies(
    df: pd.DataFrame,
    geography_type: str | None = None,
) -> pd.DataFrame:
    """Return unique geographies, optionally filtered by ``geography_type``."""
    if "geography_id" not in df.columns:
        raise ValueError("Dataframe must include geography_id")

    geographies = df[
        ["geography_id", "geography_name", "geography_type"]
    ].drop_duplicates()

    if geography_type is not None:
        geographies = geographies[
            geographies["geography_type"].astype(str).eq(str(geography_type))
        ]

    return geographies.sort_values("geography_name").reset_index(drop=True)


def _brand_heatmap_cmap() -> LinearSegmentedColormap:
    """Low values in brand cream, high values in brand red."""
    return LinearSegmentedColormap.from_list(
        "brand_heatmap",
        [BRAND_CREAM, BRAND_RED],
    )


def _statistical_region_id_from_feature(properties: dict) -> str | None:
    """Map open GeoJSON region codes to the E120000xx ids used in the database."""
    for key in ("RGN23CD", "RGN22CD", "RGN21CD", "EER13CD"):
        code = properties.get(key)
        if not code:
            continue

        code = str(code)
        if code.startswith("E150"):
            return "E120" + code[4:]
        if code.startswith("E120"):
            return code

    return None


def _load_england_regions_geojson(force_download: bool = False) -> dict:
    """Load cached England region boundaries, downloading them on first use."""
    if (
        not force_download
        and ENGLAND_REGIONS_GEOJSON_FILE.exists()
        and ENGLAND_REGIONS_GEOJSON_FILE.stat().st_size > 0
    ):
        return json.loads(ENGLAND_REGIONS_GEOJSON_FILE.read_text())

    with urllib.request.urlopen(ENGLAND_REGIONS_GEOJSON_URL, timeout=60) as response:
        payload = json.load(response)

    england_features = []
    for feature in payload.get("features", []):
        region_id = _statistical_region_id_from_feature(feature.get("properties", {}))
        if region_id and region_id.startswith("E120"):
            england_features.append(feature)

    if not england_features:
        raise RuntimeError("Could not find England region boundaries in GeoJSON source")

    geojson = {"type": "FeatureCollection", "features": england_features}
    ENGLAND_REGIONS_GEOJSON_FILE.write_text(json.dumps(geojson))
    return geojson


def _iter_polygon_rings(geometry: dict):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        for ring in coordinates:
            yield ring
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates:
            for ring in polygon:
                yield ring


def _polygon_centroid(ring: list[list[float]]) -> tuple[float, float]:
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return float(np.mean(xs)), float(np.mean(ys))


def _is_percentage_indicator(unit: str | None) -> bool:
    unit_label = str(unit or "").strip().lower()
    return "percent" in unit_label or "proportion" in unit_label


def _to_percentage_points(values: pd.Series) -> pd.Series:
    """
    Normalise percentage indicators to percentage points for display.

    Some indicators are stored as decimals (0.18 = 18%), while others are
    already stored as percentage points (12.0 = 12%). Indicators whose
    maximum value is above 1 are treated as percentage points already.
    Values stored to one decimal place (for example 0.3 = 0.3%) are also
    treated as percentage points.
    """
    numeric = pd.to_numeric(values, errors="coerce")

    if numeric.max() > 1.0:
        return numeric

    if np.allclose(numeric, numeric.round(1), atol=1e-9, equal_nan=True):
        return numeric

    return numeric * 100


def _format_percentage_point(value: float) -> str:
    rounded = round(float(value), 1)
    if float(rounded).is_integer():
        return f"{int(rounded)}.0%"
    return f"{rounded:.1f}%"


def _get_bng_transformer():
    try:
        from pyproj import Transformer
    except ImportError as exc:
        raise ImportError(
            "British National Grid projection requires pyproj. "
            "Install it with: pip install pyproj"
        ) from exc

    return Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)


def _project_ring_to_bng(ring: list[list[float]]) -> list[tuple[float, float]]:
    transformer = _get_bng_transformer()
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    xs, ys = transformer.transform(lons, lats)
    return list(zip(xs, ys))


def _filter_england_regions_indicator(
    df: pd.DataFrame,
    indicator_id: str,
    source_id: str | int | None = None,
) -> pd.DataFrame:
    filtered = df[
        df["indicator_id"].astype(str).eq(str(indicator_id))
        & df["geography_type"].astype(str).eq("region")
    ].copy()

    if source_id is not None:
        filtered = filtered[
            filtered["source_id"].astype(str).eq(str(source_id))
        ].copy()

    if filtered.empty:
        raise ValueError(
            f"No regional observations found for indicator_id='{indicator_id}'"
        )

    indicator_unit = (
        str(filtered["unit"].dropna().iloc[0])
        if filtered["unit"].notna().any()
        else ""
    )
    if not _is_percentage_indicator(indicator_unit):
        raise ValueError(
            f"Indicator '{indicator_id}' is not a percentage indicator "
            f"(unit='{indicator_unit}'). The heatmap only supports proportions."
        )

    if filtered[["year", "geography_id"]].duplicated().any():
        filtered = (
            filtered.sort_values(["year", "geography_id", "source_id"])
            .drop_duplicates(subset=["year", "geography_id"], keep="last")
        )

    filtered["percentage_value"] = _to_percentage_points(filtered["value"])
    return filtered


def _build_england_regions_geometry(
    value_by_region: dict[str, float],
    name_by_region: dict[str, str],
) -> tuple[list[PathPatch], list[str], list[tuple[float, float, str, str]]]:
    geojson = _load_england_regions_geojson()
    patches: list[PathPatch] = []
    region_ids: list[str] = []
    label_points: list[tuple[float, float, str, str]] = []

    for feature in geojson["features"]:
        region_id = _statistical_region_id_from_feature(feature.get("properties", {}))
        if region_id is None or region_id not in value_by_region:
            continue

        rings = list(_iter_polygon_rings(feature["geometry"]))
        if not rings:
            continue

        main_ring = max(rings, key=len)
        projected_ring = _project_ring_to_bng(main_ring)
        path = MplPath(projected_ring, closed=True)
        patches.append(PathPatch(path, linewidth=0.6, edgecolor=BRAND_CHARCOAL))
        region_ids.append(region_id)

        centroid = _polygon_centroid(projected_ring)
        label_points.append(
            (
                centroid[0],
                centroid[1],
                region_id,
                name_by_region.get(region_id, region_id),
            )
        )

    if not patches:
        raise ValueError(
            "Regional values were found in the database but could not be matched "
            "to England region boundaries."
        )

    return patches, region_ids, label_points


def _add_england_regions_label_artists(
    map_ax,
    label_points: list[tuple[float, float, str, str, float]],
    fonts: dict[str, float] | None = None,
) -> list[AnnotationBbox]:
    fonts = fonts or _scaled_england_regions_font_sizes()
    label_artists: list[AnnotationBbox] = []

    for x_coord, y_coord, region_id, region_name, percentage_value in label_points:
        offset_x, offset_y = REGION_LABEL_OFFSETS.get(region_id, (0.0, 0.0))
        label_x = x_coord + offset_x
        label_y = y_coord + offset_y

        name_text = TextArea(
            region_name,
            textprops={
                "fontproperties": _roboto_slab_fontproperties(
                    size=fonts["region_name"],
                    weight=700,
                ),
                "color": BRAND_CHARCOAL,
            },
        )
        value_text = TextArea(
            _format_percentage_point(percentage_value),
            textprops={
                "fontproperties": _roboto_slab_fontproperties(size=fonts["region_value"]),
                "color": BRAND_CHARCOAL,
            },
        )
        label_box = VPacker(
            children=[name_text, value_text],
            align="center",
            pad=0,
            sep=3,
        )
        label_artist = AnnotationBbox(
            label_box,
            (label_x, label_y),
            frameon=True,
            bboxprops={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            zorder=6,
        )
        map_ax.add_artist(label_artist)
        label_artists.append(label_artist)

    return label_artists


HEADER_TITLE_YEAR_HSPACE = 0.10


def _roboto_slab_fontproperties(*, size: float, weight: int | str = 400) -> FontProperties:
    for font in fontManager.ttflist:
        if font.name == "Roboto Slab":
            return FontProperties(fname=font.fname, size=size, weight=weight)
    return FontProperties(family="Roboto Slab", size=size, weight=weight)


def _scaled_england_regions_font_sizes(font_scale: float = 1.0) -> dict[str, float]:
    return {
        key: size * font_scale for key, size in ENGLAND_REGIONS_FONT_SIZES.items()
    }


def _create_england_regions_figure_layout(
    figsize: tuple[float, float],
    show_colorbar: bool,
) -> tuple[plt.Figure, plt.Axes, plt.Axes, plt.Axes, plt.Axes]:
    plt.rcParams["font.family"] = "Roboto Slab"
    fig = plt.figure(figsize=figsize, facecolor=HEATMAP_BACKGROUND)
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[2.0, 14, 0.65],
        hspace=0.10,
        left=0.06,
        right=0.86 if show_colorbar else 0.97,
        top=0.98,
        bottom=0.04,
    )

    header_gs = gs[0].subgridspec(
        2,
        1,
        height_ratios=[4.0, 1.0],
        hspace=HEADER_TITLE_YEAR_HSPACE,
    )
    title_ax = fig.add_subplot(header_gs[0])
    title_ax.axis("off")
    year_ax = fig.add_subplot(header_gs[1])
    year_ax.axis("off")

    map_ax = fig.add_subplot(gs[1])
    map_ax.set_facecolor(HEATMAP_BACKGROUND)

    footer_ax = fig.add_subplot(gs[2])
    footer_ax.axis("off")

    return fig, map_ax, title_ax, year_ax, footer_ax


def _draw_england_regions_header_block(
    title_ax: plt.Axes,
    year_ax: plt.Axes,
    title: str,
    year: str,
    fonts: dict[str, float] | None = None,
) -> plt.Text:
    """Draw title and year in separate header rows for reliable spacing."""
    fonts = fonts or _scaled_england_regions_font_sizes()
    wrapped_title = textwrap.fill(title, width=60)
    title_ax.text(
        0.0,
        1.0,
        wrapped_title,
        transform=title_ax.transAxes,
        ha="left",
        va="top",
        fontproperties=_roboto_slab_fontproperties(size=fonts["title"], weight=700),
        color=BRAND_CHARCOAL,
        linespacing=1.25,
    )

    return year_ax.text(
        0.0,
        1.0,
        f"Year: {_format_year_label(year)}",
        transform=year_ax.transAxes,
        ha="left",
        va="top",
        fontproperties=_roboto_slab_fontproperties(size=fonts["year"], weight=600),
        color=BRAND_CHARCOAL,
    )


def _draw_england_regions_footer(
    footer_ax: plt.Axes,
    source_name: str,
    note: str | None,
    show_footer_note: bool,
    fonts: dict[str, float] | None = None,
) -> None:
    if not show_footer_note:
        return

    fonts = fonts or _scaled_england_regions_font_sizes()
    footer_lines = []
    if note:
        footer_lines.append(f"Note: {textwrap.fill(note, width=90)}")
    footer_lines.append(f"Source: {source_name}")
    footer_ax.text(
        0.0,
        1.0,
        "\n".join(footer_lines),
        transform=footer_ax.transAxes,
        ha="left",
        va="top",
        fontproperties=_roboto_slab_fontproperties(size=fonts["footer"]),
        color=BRAND_CHARCOAL,
        linespacing=1.35,
    )


def _style_england_regions_map_ax(map_ax: plt.Axes) -> None:
    map_ax.autoscale()
    map_ax.set_aspect("equal", adjustable="box")
    map_ax.margins(0.05)
    map_ax.axis("off")


def _add_england_regions_colorbar(
    fig: plt.Figure,
    collection: PatchCollection,
    map_ax: plt.Axes,
    fonts: dict[str, float] | None = None,
) -> None:
    fonts = fonts or _scaled_england_regions_font_sizes()
    divider = make_axes_locatable(map_ax)
    cbar_ax = divider.append_axes("right", size="1.2%", pad=0.20)
    cbar_ax.set_facecolor(HEATMAP_BACKGROUND)
    colorbar = fig.colorbar(collection, cax=cbar_ax)
    colorbar.set_label(
        "Percentage %",
        fontproperties=_roboto_slab_fontproperties(size=fonts["colorbar_label"]),
        color=BRAND_CHARCOAL,
        labelpad=14,
    )
    colorbar.ax.tick_params(
        labelsize=fonts["colorbar_ticks"],
        colors=BRAND_CHARCOAL,
        width=0.5,
    )
    colorbar.outline.set_edgecolor(BRAND_CHARCOAL)
    colorbar.outline.set_linewidth(0.5)


def _render_england_regions_heatmap_figure(
    *,
    year: str,
    indicator_name: str,
    source_name: str,
    patches: list[PathPatch],
    patch_values: np.ndarray,
    label_points: list[tuple[float, float, str, str, float]],
    norm: Normalize,
    title: str | None = None,
    note: str | None = None,
    figsize: tuple[float, float] = ENGLAND_REGIONS_FIGSIZE,
    show_labels: bool = True,
    show_colorbar: bool = True,
    show_footer_note: bool = True,
    font_scale: float = 1.15,
) -> plt.Figure:
    cmap = _brand_heatmap_cmap()
    cmap.set_bad(color=HEATMAP_BACKGROUND)

    plot_title = title or f"{indicator_name} (Regions in England)"
    fonts = _scaled_england_regions_font_sizes(font_scale)
    fig, map_ax, title_ax, year_ax, footer_ax = _create_england_regions_figure_layout(
        figsize,
        show_colorbar,
    )
    _draw_england_regions_header_block(title_ax, year_ax, plot_title, year, fonts)

    collection = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        edgecolor=BRAND_CHARCOAL,
        linewidth=0.6,
    )
    collection.set_array(patch_values)
    map_ax.add_collection(collection)
    _style_england_regions_map_ax(map_ax)

    if show_labels:
        _add_england_regions_label_artists(map_ax, label_points, fonts)

    if show_colorbar:
        _add_england_regions_colorbar(fig, collection, map_ax, fonts)
    _draw_england_regions_footer(footer_ax, source_name, note, show_footer_note, fonts)

    return fig


def _resolve_england_regions_save_path(
    save_path: Path | str | None,
    default_name: str,
    *,
    default_dir: Path | None = None,
) -> Path:
    default_dir = default_dir or OUTPUT_DIR

    if save_path is None:
        save_path = default_dir / default_name
    else:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = default_dir / save_path

    save_path = save_path.resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def _save_england_regions_heatmap_animation(
    *,
    years: list[str],
    values_by_year: dict[str, dict[str, float]],
    region_ids: list[str],
    base_label_points: list[tuple[float, float, str, str]],
    patches: list[PathPatch],
    norm: Normalize,
    indicator_name: str,
    source_name: str,
    title: str | None,
    note: str | None,
    figsize: tuple[float, float],
    show_labels: bool,
    show_colorbar: bool,
    show_footer_note: bool,
    save_path: Path,
    duration_ms: int,
    dpi: int,
    font_scale: float = 1.0,
) -> None:
    cmap = _brand_heatmap_cmap()
    cmap.set_bad(color=HEATMAP_BACKGROUND)

    plot_title = title or f"{indicator_name} (Regions in England)"
    fonts = _scaled_england_regions_font_sizes(font_scale)
    fig, map_ax, title_ax, year_ax, footer_ax = _create_england_regions_figure_layout(
        figsize,
        show_colorbar,
    )
    year_text = _draw_england_regions_header_block(
        title_ax, year_ax, plot_title, years[0], fonts
    )
    _draw_england_regions_footer(footer_ax, source_name, note, show_footer_note, fonts)

    def _label_points_for_year(year: str) -> list[tuple[float, float, str, str, float]]:
        value_by_region = values_by_year[str(year)]
        return [
            (
                x_coord,
                y_coord,
                region_id,
                region_name,
                value_by_region[region_id],
            )
            for x_coord, y_coord, region_id, region_name in base_label_points
            if region_id in value_by_region
        ]

    def _patch_values_for_year(year: str) -> np.ndarray:
        value_by_region = values_by_year[str(year)]
        return np.array(
            [value_by_region.get(region_id, np.nan) for region_id in region_ids],
            dtype=float,
        )

    collection = PatchCollection(
        patches,
        cmap=cmap,
        norm=norm,
        edgecolor=BRAND_CHARCOAL,
        linewidth=0.6,
    )
    collection.set_array(_patch_values_for_year(years[0]))
    map_ax.add_collection(collection)
    _style_england_regions_map_ax(map_ax)

    label_artists: list[AnnotationBbox] = []

    def _refresh_labels(year: str) -> None:
        nonlocal label_artists
        for artist in label_artists:
            artist.remove()
        label_artists = []
        if show_labels:
            label_artists = _add_england_regions_label_artists(
                map_ax,
                _label_points_for_year(year),
                fonts,
            )

    if show_colorbar:
        _add_england_regions_colorbar(fig, collection, map_ax, fonts)

    _refresh_labels(years[0])

    def _update(frame_index: int):
        year = years[frame_index]
        collection.set_array(_patch_values_for_year(year))
        year_text.set_text(f"Year: {_format_year_label(year)}")
        _refresh_labels(year)
        return [collection, year_text, *label_artists]

    fps = max(0.2, 1000 / duration_ms)
    animation = FuncAnimation(
        fig,
        _update,
        frames=len(years),
        interval=duration_ms,
        blit=False,
        repeat=True,
    )
    writer = PillowWriter(fps=fps)
    animation.save(str(save_path), writer=writer, dpi=dpi)
    plt.close(fig)


def plot_england_regions_heatmap(
    indicator_id: str,
    *,
    year: str | None = None,
    source_id: str | int | None = None,
    df: pd.DataFrame | None = None,
    title: str | None = None,
    note: str | None = None,
    figsize: tuple[float, float] = ENGLAND_REGIONS_FIGSIZE,
    save_path: Path | str | None = None,
    show_labels: bool = True,
    show_colorbar: bool = True,
    show_footer_note: bool = True,
    show: bool = True,
    font_scale: float = 1.0,
) -> dict:
    """
    Plot a proportion-only choropleth heatmap of English regions for one indicator.

    Values are loaded from the ``observations`` table (via ``load_observations``)
    and matched to ONS region boundaries. Only percentage/proportion indicators
    are supported. The colour scale runs from brand cream (#fff5d2) for low
    values to brand red (#ff4d5e) for high values.
    """
    if df is None:
        df = load_observations()

    filtered = _filter_england_regions_indicator(df, indicator_id, source_id)
    available_years = _sort_observation_years(filtered["year"].astype(str).unique())

    if year is None:
        year = available_years[-1]

    year = str(year)
    year_data = filtered[filtered["year"].astype(str).eq(year)].copy()

    if year_data.empty:
        raise ValueError(
            f"No regional observations found for indicator_id='{indicator_id}' "
            f"in year '{year}'. Available years: {available_years}"
        )

    value_by_region = {
        str(row["geography_id"]): float(row["percentage_value"])
        for _, row in year_data.iterrows()
    }
    name_by_region = {
        str(row["geography_id"]): str(row["geography_name"])
        for _, row in year_data.iterrows()
    }
    indicator_name = str(year_data["indicator_name"].iloc[0])
    source_name = (
        str(year_data["source_name"].dropna().iloc[0])
        if year_data["source_name"].notna().any()
        else "Unknown source"
    )

    region_values = year_data["percentage_value"].astype(float)
    highest_row = year_data.loc[region_values.idxmax()]
    lowest_row = year_data.loc[region_values.idxmin()]

    patches, region_ids, base_label_points = _build_england_regions_geometry(
        value_by_region,
        name_by_region,
    )
    patch_values = np.array(
        [value_by_region[region_id] for region_id in region_ids],
        dtype=float,
    )
    label_points = [
        (x_coord, y_coord, region_id, region_name, value_by_region[region_id])
        for x_coord, y_coord, region_id, region_name in base_label_points
    ]
    norm = Normalize(vmin=patch_values.min(), vmax=patch_values.max())

    fig = _render_england_regions_heatmap_figure(
        year=year,
        indicator_name=indicator_name,
        source_name=source_name,
        patches=patches,
        patch_values=patch_values,
        label_points=label_points,
        norm=norm,
        title=title,
        note=note,
        figsize=figsize,
        show_labels=show_labels,
        show_colorbar=show_colorbar,
        show_footer_note=show_footer_note,
        font_scale=font_scale,
    )

    if save_path:
        save_path = _resolve_england_regions_save_path(save_path, "england_regions_heatmap.png")
        fig.savefig(
            save_path,
            dpi=300,
            facecolor=HEATMAP_BACKGROUND,
        )

    if show:
        plt.show()
    else:
        plt.close(fig)

    result = year_data[
        [
            "indicator_id",
            "indicator_name",
            "year",
            "geography_id",
            "geography_name",
            "value",
            "percentage_value",
            "source_id",
            "source_name",
        ]
    ].sort_values("geography_name").reset_index(drop=True)

    return {
        "figure": fig,
        "axis": map_ax,
        "data": result,
        "year": year,
        "indicator_id": str(indicator_id),
        "highest_region": str(highest_row["geography_name"]),
        "lowest_region": str(lowest_row["geography_name"]),
    }


def plot_england_regions_heatmap_gif(
    indicator_id: str,
    *,
    years: list[str] | None = None,
    source_id: str | int | None = None,
    df: pd.DataFrame | None = None,
    title: str | None = None,
    note: str | None = None,
    figsize: tuple[float, float] = ENGLAND_REGIONS_FIGSIZE,
    save_path: Path | str | None = None,
    show_labels: bool = True,
    show_colorbar: bool = True,
    show_footer_note: bool = True,
    duration_ms: int = 1000,
    dpi: int = 150,
    font_scale: float = 1.0,
) -> dict:
    """
    Build a GIF showing how regional percentages change over time.

    Uses the same layout as ``plot_england_regions_heatmap`` and keeps one
    fixed colour scale across all frames so year-on-year changes are comparable.

    The GIF is saved under the project ``output/`` folder by default. Drag that
    file into Substack to post it.

    font_scale:
    - multiplier for title, labels, colorbar, and footer text sizes.
      Defaults to 1.15 for better GIF readability.
    """
    if save_path is not None and str(save_path).lower().endswith(".mp4"):
        save_path = Path(str(save_path)).with_suffix(".gif")

    if df is None:
        df = load_observations()

    filtered = _filter_england_regions_indicator(df, indicator_id, source_id)
    available_years = _sort_observation_years(filtered["year"].astype(str).unique())

    if years is None:
        years = available_years
    else:
        years = _sort_observation_years(years)
        missing_years = [year for year in years if year not in available_years]
        if missing_years:
            raise ValueError(
                f"No regional observations found for indicator_id='{indicator_id}' "
                f"in years {missing_years}. Available years: {available_years}"
            )

    if len(years) < 2:
        raise ValueError(
            f"At least two years are required to create an animation. "
            f"Available years: {available_years}"
        )

    indicator_name = str(filtered["indicator_name"].iloc[0])
    source_name = (
        str(filtered["source_name"].dropna().iloc[0])
        if filtered["source_name"].notna().any()
        else "Unknown source"
    )

    values_by_year: dict[str, dict[str, float]] = {}
    name_by_region: dict[str, str] = {}
    all_region_ids: set[str] = set()
    all_values: list[float] = []

    for year in years:
        year_data = filtered[filtered["year"].astype(str).eq(str(year))]
        value_by_region = {
            str(row["geography_id"]): float(row["percentage_value"])
            for _, row in year_data.iterrows()
        }
        values_by_year[str(year)] = value_by_region
        all_region_ids.update(value_by_region.keys())
        all_values.extend(value_by_region.values())
        for _, row in year_data.iterrows():
            name_by_region[str(row["geography_id"])] = str(row["geography_name"])

    geometry_values = {region_id: 0.0 for region_id in all_region_ids}
    patches, region_ids, base_label_points = _build_england_regions_geometry(
        geometry_values,
        name_by_region,
    )
    norm = Normalize(vmin=min(all_values), vmax=max(all_values))

    save_path = _resolve_england_regions_save_path(
        save_path,
        f"{indicator_id}_england_regions.gif",
        default_dir=OUTPUT_DIR,
    )
    _save_england_regions_heatmap_animation(
        years=years,
        values_by_year=values_by_year,
        region_ids=region_ids,
        base_label_points=base_label_points,
        patches=patches,
        norm=norm,
        indicator_name=indicator_name,
        source_name=source_name,
        title=title,
        note=note,
        figsize=figsize,
        show_labels=show_labels,
        show_colorbar=show_colorbar,
        show_footer_note=show_footer_note,
        save_path=save_path,
        duration_ms=duration_ms,
        dpi=dpi,
        font_scale=font_scale,
    )

    if not save_path.exists():
        raise RuntimeError(f"GIF file was not created at {save_path}")

    animation_data = filtered[filtered["year"].astype(str).isin(years)][
        [
            "indicator_id",
            "indicator_name",
            "year",
            "geography_id",
            "geography_name",
            "value",
            "percentage_value",
            "source_id",
            "source_name",
        ]
    ].sort_values(["year", "geography_name"]).reset_index(drop=True)

    return {
        "save_path": save_path,
        "years": years,
        "frame_count": len(years),
        "data": animation_data,
        "indicator_id": str(indicator_id),
        "color_min": float(norm.vmin),
        "color_max": float(norm.vmax),
    }


# --- LIST AVAILABLE INDICATORS ---
def list_indicators(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[["indicator_id", "indicator_name"]]
        .drop_duplicates()
        .sort_values("indicator_id")
        .reset_index(drop=True)
    )


def _format_year_label(year) -> str:
    """Format financial-year labels such as 202224 or 2022/2024 as 2022/24."""
    year_label = str(year)

    if year_label.isdigit() and len(year_label) == 6:
        return f"{year_label[:4]}/{year_label[4:]}"

    if "/" in year_label:
        start_year, end_year = year_label.split("/", 1)

        if (
            len(start_year) == 4
            and len(end_year) == 4
            and start_year.isdigit()
            and end_year.isdigit()
        ):
            return f"{start_year}/{end_year[-2:]}"

    return year_label


def _normalise_year_period(year) -> dict:
    """Return comparable year-period parts for mixed calendar/FY labels."""
    year_label = str(year).strip()

    compact_financial_year = re.fullmatch(r"(\d{4})(\d{2})", year_label)
    if compact_financial_year:
        start_year, end_year = compact_financial_year.groups()
        return {
            "period_start_year": int(start_year),
            "period_exact_key": f"fy:{start_year}/{end_year}",
            "period_label": f"{start_year}/{end_year}",
            "period_is_financial": True,
        }

    separated_financial_year = re.fullmatch(
        r"(\d{4})\s*[/\-_]\s*(\d{2}|\d{4})",
        year_label,
    )
    if separated_financial_year:
        start_year, end_year = separated_financial_year.groups()
        end_year = end_year[-2:]
        return {
            "period_start_year": int(start_year),
            "period_exact_key": f"fy:{start_year}/{end_year}",
            "period_label": f"{start_year}/{end_year}",
            "period_is_financial": True,
        }

    calendar_year = re.fullmatch(r"\d{4}", year_label)
    if calendar_year:
        return {
            "period_start_year": int(year_label),
            "period_exact_key": f"cy:{year_label}",
            "period_label": year_label,
            "period_is_financial": False,
        }

    first_year = re.search(r"(\d{4})", year_label)
    if first_year:
        start_year = first_year.group(1)
        return {
            "period_start_year": int(start_year),
            "period_exact_key": f"raw:{year_label}",
            "period_label": _format_year_label(year_label),
            "period_is_financial": False,
        }

    return {
        "period_start_year": np.nan,
        "period_exact_key": f"raw:{year_label}",
        "period_label": _format_year_label(year_label),
        "period_is_financial": False,
    }


def _sort_observation_years(years) -> list[str]:
    return sorted(
        [str(year) for year in years],
        key=lambda year: (
            _normalise_year_period(year)["period_start_year"],
            str(year),
        ),
    )


YEAR_PERIOD_COLUMNS = [
    "period_start_year",
    "period_exact_key",
    "period_label",
    "period_is_financial",
]


def _expand_year_period_columns(years: pd.Series) -> pd.DataFrame:
    """Expand year labels into comparable period columns."""
    if years.empty:
        return pd.DataFrame(columns=YEAR_PERIOD_COLUMNS)

    period_parts = pd.DataFrame(years.map(_normalise_year_period).tolist())
    for column in YEAR_PERIOD_COLUMNS:
        if column not in period_parts.columns:
            period_parts[column] = pd.NA
    return period_parts[YEAR_PERIOD_COLUMNS]


def _nice_axis_step(span: float) -> float:
    """Return a clean tick-style step for rounding axis limits."""
    if span <= 0 or np.isnan(span):
        return 1

    raw_step = span / 5
    magnitude = 10 ** math.floor(math.log10(raw_step))
    scaled_step = raw_step / magnitude

    if scaled_step <= 1:
        nice_step = 1
    elif scaled_step <= 2:
        nice_step = 2
    elif scaled_step <= 5:
        nice_step = 5
    else:
        nice_step = 10

    return nice_step * magnitude


def _round_axis_limits(
    lower: float,
    upper: float,
    *,
    min_limit: float | None = None,
    max_limit: float | None = None,
) -> tuple[float, float]:
    """Round y-axis limits to clean values and keep optional hard bounds."""
    if upper <= lower:
        upper = lower + 1

    step = _nice_axis_step(upper - lower)
    rounded_lower = math.floor(lower / step) * step
    rounded_upper = math.ceil(upper / step) * step

    if min_limit is not None:
        rounded_lower = max(min_limit, rounded_lower)

    if max_limit is not None:
        rounded_upper = min(max_limit, rounded_upper)

    if rounded_upper <= rounded_lower:
        rounded_upper = rounded_lower + step

        if max_limit is not None and rounded_upper > max_limit:
            rounded_upper = max_limit
            rounded_lower = max(min_limit or 0, rounded_upper - step)

    return rounded_lower, rounded_upper


def _get_contextual_y_limits(
    values,
    value_type: str,
    y_axis: str = "contextual",
    y_limits: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """
    Decide suitable y-axis limits for an indicator chart.

    The aim is to avoid overly zoomed-in y-axes that make small changes look
    more volatile than they are, while still keeping the line readable.
    """
    values = pd.Series(values).dropna()

    # Manual limits always take priority over automated scaling.
    if y_limits is not None:
        if len(y_limits) != 2 or y_limits[1] <= y_limits[0]:
            raise ValueError("y_limits must be a tuple like (lower, upper)")

        return y_limits

    if values.empty:
        return None

    valid_y_axis_options = {"contextual", "zero", "auto"}
    if y_axis not in valid_y_axis_options:
        raise ValueError(
            "y_axis must be one of: "
            + ", ".join(sorted(valid_y_axis_options))
        )

    # "auto" keeps Matplotlib's default scaling.
    if y_axis == "auto":
        return None

    min_value = values.min()
    max_value = values.max()
    observed_range = max_value - min_value

    # "zero" is useful when the chart needs to emphasise absolute scale.
    if y_axis == "zero":
        upper = max_value * 1.10 if max_value > 0 else 1
        return _round_axis_limits(0, upper, min_limit=0)

    # Count indicators are scaled against their typical size.
    # A movement of 1,000 is small around 30,000, but large around 2,000.
    if value_type == "count":
        typical_value = values.median()

        # If the line is flat, still give it enough vertical space to read.
        if observed_range == 0:
            visible_range = max(abs(typical_value) * 0.25, 1)
        else:
            # Use whichever is larger:
            # - the observed movement plus padding, so real change is visible
            # - 25% of the typical value, so small change is not over-zoomed
            visible_range = max(
                observed_range * 1.5,
                abs(typical_value) * 0.25,
            )

        center = (min_value + max_value) / 2
        lower = center - visible_range / 2
        upper = center + visible_range / 2

        # Count data cannot go below zero.
        if lower < 0:
            upper = max(upper, visible_range)
            lower = 0

        return _round_axis_limits(lower, upper, min_limit=0)

    # Proportion indicators are plotted as percentages, so use percentage
    # points of padding and keep the scale within the natural 0%-100% bounds.
    if value_type == "proportion":
        padding = max(observed_range * 0.5, 2)
        lower = max(0, min_value - padding)
        upper = min(100, max_value + padding)

        return _round_axis_limits(lower, upper, min_limit=0, max_limit=100)

    return None


## ADJUSTING ANOTATION IN THE FINAL GRAPH OUTPUT

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
    indicator_id: str | None = None,
    source_id: str | int | None = None,
    year_filter: str | list[str] | None = None,
    df: pd.DataFrame | None = None,
    title: str | None = None,
    note: str | None = None,
    figsize: tuple = (14, 8),
    save_path: Path | str | None = None,
    value_type: str = "count",  # "count" or "proportion"
    y_label: str | None = None,
    y_axis: str = "contextual",  # "contextual", "zero", or "auto"
    y_limits: tuple[float, float] | None = None,
    sum_indicator_ids: list[str] | tuple[str, ...] | None = None,
    confirm_mixed_sources: bool = True,
    show_footer_note: bool = True,
    footer_note: str | None = None,
    font_scale: float = 1.15,
) -> None:
    """
    Plot a line graph for a given indicator_id using year vs value.

    If sum_indicator_ids is provided, values for those indicators are summed
    by year before plotting. This summed mode only accepts count values and
    requires a manual title.

    Analysis:
    - total percentage change over the full period
    - CAGR-style average annual percentage change
    - latest annual percentage change

    value_type:
    - "count": values are numbers of CYP
    - "proportion": values are proportions, e.g. 0.10 = 10%

    y_label:
    - optional manual y-axis label. If omitted, a default label is chosen
      from value_type.

    y_axis:
    - "contextual": automatically widens the y-axis when values vary little
      relative to their typical size
    - "zero": starts the y-axis at zero
    - "auto": leaves y-axis scaling to Matplotlib

    y_limits:
    - optional manual override, e.g. (25000, 35000)

    show_footer_note:
    - True: show source and methodology text at the bottom of the figure
    - False: hide the bottom footer text

    footer_note:
    - optional custom footer text. When provided, this replaces the
      auto-generated source and methodology footer. ``note`` is ignored
      when ``footer_note`` is set.
    """

    if df is None:
        df = load_data()

    # Indicator filtering
    sum_mode = sum_indicator_ids is not None

    if sum_mode:
        if value_type != "count":
            raise ValueError(
                "sum_indicator_ids can only be used with value_type='count'"
            )

        if not title:
            raise ValueError(
                "A manual title is required when using sum_indicator_ids"
            )

        if isinstance(sum_indicator_ids, str):
            sum_indicator_ids = [sum_indicator_ids]

        if len(sum_indicator_ids) < 2:
            raise ValueError(
                "sum_indicator_ids must contain at least two indicator ids"
            )

        sum_indicator_ids = [str(i) for i in sum_indicator_ids]
        indicator_label = " + ".join(sum_indicator_ids)
        filtered = df[
            df["indicator_id"].astype(str).isin(sum_indicator_ids)
        ].copy()
    else:
        if indicator_id is None:
            raise ValueError(
                "indicator_id is required unless sum_indicator_ids is provided"
            )

        indicator_label = str(indicator_id)
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
    if sum_mode:
        filters = [f"indicator_ids={sum_indicator_ids}"]
    else:
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

    if sum_mode:
        found_indicator_ids = set(filtered["indicator_id"].astype(str).unique())
        missing_indicator_ids = [
            i for i in sum_indicator_ids if i not in found_indicator_ids
        ]

        if missing_indicator_ids:
            raise ValueError(
                "No valid numeric rows found for summed indicator ids: "
                + ", ".join(missing_indicator_ids)
            )

        coverage = filtered.groupby("year")["indicator_id"].nunique()
        incomplete_years = coverage[coverage < len(sum_indicator_ids)]

        if not incomplete_years.empty:
            raise ValueError(
                "Cannot sum indicators because these years are missing at least "
                "one requested indicator: "
                + ", ".join(incomplete_years.index.astype(str))
            )

        source_summary = (
            filtered[["indicator_id", "source_id", "source_name"]]
            .drop_duplicates()
            .sort_values(["indicator_id", "source_id"])
        )
        unique_sources = source_summary["source_id"].astype(str).nunique()

        if unique_sources > 1 and confirm_mixed_sources:
            print("\nWARNING: You are summing indicators from different sources.")
            print("Please review the source mapping before continuing:\n")
            print(source_summary.to_string(index=False))

            response = input(
                "\nContinue summing indicators from different sources? "
                "Type 'yes' to continue: "
            )

            if response.strip().lower() != "yes":
                raise ValueError(
                    "Summed indicator plot cancelled because indicators come "
                    "from different sources."
                )

        source_names = source_summary["source_name"].dropna().unique()
        summed_source_name = (
            source_names[0] if len(source_names) == 1 else "Multiple sources"
        )

        filtered = (
            filtered
            .groupby(["year", "sort_year"], dropna=False, as_index=False)
            .agg(value=("value", "sum"))
            .sort_values(["sort_year", "year"])
            .reset_index(drop=True)
        )

        filtered["indicator_id"] = indicator_label
        filtered["indicator_name"] = title
        filtered["source_name"] = summed_source_name

    filtered["year_label"] = filtered["year"].map(_format_year_label)

    if filtered.empty:
        raise ValueError(
            f"No valid numeric values found for indicator_id='{indicator_label}'"
        )

    if len(filtered) < 3:
        raise ValueError(
            f"At least three valid observations are needed for summary calculations "
            f"for indicator_id='{indicator_label}'"
        )

    # Value type handling
    if value_type not in ["count", "proportion"]:
        raise ValueError("value_type must be either 'count' or 'proportion'")

    if value_type == "proportion":
        filtered["display_value"] = filtered["value"] * 100
        default_y_label = "Percentage of Children and Young People (%)"
        value_formatter = lambda x: f"{x:.1f}%"
    else:
        filtered["display_value"] = filtered["value"]
        default_y_label = "Number of Children and Young People (CYP)"
        value_formatter = lambda x: f"{x:,.0f}"

    plot_y_label = y_label if y_label is not None else default_y_label

    # Decide whether to apply custom y-axis limits.
    # For proportions, this happens after conversion to percentages.
    computed_y_limits = _get_contextual_y_limits(
        filtered["display_value"],
        value_type=value_type,
        y_axis=y_axis,
        y_limits=y_limits,
    )

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
        else indicator_label
    )

    source_name = (
        filtered["source_name"].dropna().iloc[0]
        if "source_name" in filtered.columns
        and not filtered["source_name"].dropna().empty
        else "Source not available"
    )

    plot_title = title if title else f"{indicator_name} ({indicator_label})"

    # ------------------------------------------------------------------
    # Summary calculations
    # ------------------------------------------------------------------

    start_row = filtered.iloc[0]
    previous_row = filtered.iloc[-2]
    end_row = filtered.iloc[-1]

    start_time = float(start_row["time"])
    end_time = float(end_row["time"])

    start_year_label = start_row["year_label"]
    end_year_label = end_row["year_label"]

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
    print(f"Summary calculations for {indicator_label}: {indicator_name}")
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

    # Apply y-axis limits unless the helper returns None.
    # None means "leave the axis to Matplotlib".
    if computed_y_limits is not None:
        ax.set_ylim(computed_y_limits)

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
    ax.set_ylabel(plot_y_label, fontsize=12, labelpad=20)
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
    ax.set_xticklabels(filtered["year_label"], rotation=45, ha="right")

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
    if footer_note is not None:
        footer_text = textwrap.fill(footer_note.strip(), width=85)
    else:
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

    if show_footer_note:
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

    bottom_margin = 0.10 if show_footer_note else 0.03
    plt.tight_layout(rect=[0, bottom_margin, 1, 1])

    if save_path:
        save_path = Path(save_path)

        if not save_path.is_absolute():
            save_path = OUTPUT_DIR / save_path

        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)

    plt.show()


def ploting_multiple_indicators(
    indicator_ids: list[str] | tuple[str, ...],
    source_id: str | int | None = None,
    source_ids: list[str | int | None] | tuple[str | int | None, ...] | None = None,
    year_filter: str | list[str] | None = None,
    df: pd.DataFrame | None = None,
    title: str | None = None,
    labels: list[str] | tuple[str, ...] | None = None,
    note: str | None = None,
    figsize: tuple | None = None,
    paper_size: str = "a4_landscape",
    save_path: Path | str | None = None,
    value_type: str = "count",
    y_label: str | None = None,
    y_axis: str = "contextual",
    y_limits: tuple[float, float] | None = None,
    x_axis_alignment: str = "period_start",
    comparison_mode: str = "values",
    point_labels: str = "all",
    point_label_position: str = "smart",
    point_label_offsets: tuple[tuple[int, int], ...] | None = None,
    point_label_fontsize: int = 11,
    legend_position: str = "smart",
    summary_note_position: str = "smart",
    show_footer_note: bool = True,
    footer_note: str | None = None,
) -> None:
    """
    Plot multiple indicators on the same line graph using the plot_indicator style.

    Parameters are intentionally close to plot_indicator. Use source_id when
    the same source filter should apply to every indicator, or source_ids when
    each indicator needs a different source filter.

    value_type:
    - "count": values are numbers of CYP
    - "proportion": values are proportions, e.g. 0.10 = 10%

    x_axis_alignment:
    - "period_start": align calendar years and financial years by the first
      year in the period, e.g. 2018 and 2018/19 share one tick
    - "exact": only merge equivalent financial-year formats, e.g. 201819,
      2018/19, and 2018/2019

    comparison_mode:
    - "values": plot the original values on one shared y-axis
    - "index": rebase each line to its first observation = 100, which makes
      relative change comparable when the indicators have different magnitudes

    paper_size:
    - "a4_landscape": 11.69 x 8.27 inches
    - "a4_portrait": 8.27 x 11.69 inches
    - "custom": use figsize instead

    point_labels:
    - "all": label every point
    - "endpoints": label only the first and last point in each line
    - "latest": label only the latest point in each line
    - "none": do not show point labels

    point_label_position:
    - "smart": automatically places labels above or below each point based on
      the other line's position
    - "fixed": uses one fixed offset for each line

    point_label_offsets:
    - optional manual (x, y) text offsets, one for each line, e.g.
      ((0, 14), (0, -22), (0, 30)). If provided, these override smart positioning.

    legend_position:
    - "smart": inside for wide figures, above the plot for narrow figures
    - "inside": upper-right inside the plot
    - "top": above the plot
    - "none": hide the legend

    summary_note_position:
    - "smart": place the summary note in the least crowded plot corner
    - "top_left", "top_right", "bottom_left", or "bottom_right"
    - "none": hide the summary note

    show_footer_note:
    - True: show source and methodology text at the bottom of the figure
    - False: hide the bottom footer text

    footer_note:
    - optional custom footer text. When provided, this replaces the
      auto-generated source and methodology footer. ``note`` is ignored
      when ``footer_note`` is set.
    """

    if df is None:
        df = load_data()

    if value_type not in ["count", "proportion"]:
        raise ValueError("value_type must be either 'count' or 'proportion'")

    if isinstance(indicator_ids, str):
        raise ValueError("indicator_ids must be a list or tuple of indicator ids")

    indicator_ids = list(indicator_ids)
    expected_series_count = len(indicator_ids)

    if expected_series_count < 2:
        raise ValueError("indicator_ids must contain at least two indicator ids")

    if labels is not None and len(labels) != expected_series_count:
        raise ValueError(
            f"labels must contain exactly {expected_series_count} labels"
        )

    valid_point_labels = {"all", "endpoints", "latest", "none"}
    if point_labels not in valid_point_labels:
        raise ValueError(
            "point_labels must be one of: "
            + ", ".join(sorted(valid_point_labels))
        )

    if (
        point_label_offsets is not None
        and len(point_label_offsets) != expected_series_count
    ):
        raise ValueError(
            "point_label_offsets must contain exactly "
            f"{expected_series_count} offsets"
        )

    valid_point_label_positions = {"smart", "fixed"}
    if point_label_position not in valid_point_label_positions:
        raise ValueError(
            "point_label_position must be one of: "
            + ", ".join(sorted(valid_point_label_positions))
        )

    valid_legend_positions = {"smart", "inside", "top", "none"}
    if legend_position not in valid_legend_positions:
        raise ValueError(
            "legend_position must be one of: "
            + ", ".join(sorted(valid_legend_positions))
        )

    valid_summary_note_positions = {
        "smart",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "none",
    }
    if summary_note_position not in valid_summary_note_positions:
        raise ValueError(
            "summary_note_position must be one of: "
            + ", ".join(sorted(valid_summary_note_positions))
        )

    valid_comparison_modes = {"values", "index"}
    if comparison_mode not in valid_comparison_modes:
        raise ValueError(
            "comparison_mode must be one of: "
            + ", ".join(sorted(valid_comparison_modes))
        )

    valid_x_axis_alignments = {"period_start", "exact"}
    if x_axis_alignment not in valid_x_axis_alignments:
        raise ValueError(
            "x_axis_alignment must be one of: "
            + ", ".join(sorted(valid_x_axis_alignments))
        )

    paper_sizes = {
        "a4_landscape": (11.69, 8.27),
        "a4_portrait": (8.27, 11.69),
        "custom": None,
    }
    if paper_size not in paper_sizes:
        raise ValueError(
            "paper_size must be one of: " + ", ".join(sorted(paper_sizes))
        )

    if figsize is None:
        if paper_size == "custom":
            raise ValueError("figsize is required when paper_size='custom'")
        figsize = paper_sizes[paper_size]

    if source_ids is None:
        source_ids = [source_id] * expected_series_count
    else:
        source_ids = list(source_ids)

        if len(source_ids) != expected_series_count:
            raise ValueError(
                f"source_ids must contain exactly {expected_series_count} values"
            )

        source_ids = [
            source_id if selected_source_id is None else selected_source_id
            for selected_source_id in source_ids
        ]

    if value_type == "proportion":
        default_y_label = "Percentage of Children and Young People (%)"
        value_formatter = lambda x: f"{x:.1f}%"
    else:
        default_y_label = "Number of Children and Young People (CYP)"
        value_formatter = lambda x: f"{x:,.0f}"

    if comparison_mode == "index":
        default_plot_y_label = "Index (first observation = 100)"
        plot_value_formatter = lambda x: f"{x:,.0f}"
    else:
        default_plot_y_label = default_y_label
        plot_value_formatter = value_formatter

    plot_y_label = y_label if y_label is not None else default_plot_y_label

    if year_filter is not None:
        selected_years = (
            [year_filter] if isinstance(year_filter, str) else list(year_filter)
        )
        selected_year_labels = {str(year).strip() for year in selected_years}
        selected_year_period_keys = {
            _normalise_year_period(year)["period_exact_key"]
            for year in selected_years
        }
    else:
        selected_years = None
        selected_year_labels = set()
        selected_year_period_keys = set()

    def prepare_indicator(indicator_id, selected_source_id):
        filtered = df[df["indicator_id"].astype(str) == str(indicator_id)].copy()

        if selected_source_id is not None:
            filtered = filtered[
                filtered["source_id"].astype(str) == str(selected_source_id)
            ].copy()

        filtered["year"] = filtered["year"].astype(str).str.strip()
        period_parts = _expand_year_period_columns(filtered["year"])
        filtered = pd.concat(
            [
                filtered.reset_index(drop=True),
                period_parts.reset_index(drop=True),
            ],
            axis=1,
        )

        if year_filter is not None:
            filtered = filtered[
                filtered["year"].isin(selected_year_labels)
                | filtered["period_exact_key"].isin(selected_year_period_keys)
            ].copy()

        filters = [f"indicator_id='{indicator_id}'"]

        if selected_source_id is not None:
            filters.append(f"source_id='{selected_source_id}'")

        if year_filter is not None:
            filters.append(f"year_filter={year_filter}")

        if filtered.empty:
            raise ValueError("No rows found for " + " and ".join(filters))

        filtered["value"] = pd.to_numeric(filtered["value"], errors="coerce")
        filtered = filtered.dropna(subset=["value"])

        filtered["sort_year"] = filtered["year"].str.extract(r"(\d{4})")[0]
        filtered["sort_year"] = pd.to_numeric(
            filtered["sort_year"],
            errors="coerce",
        )

        filtered = filtered.sort_values(["sort_year", "year"]).reset_index(drop=True)
        filtered["year_label"] = filtered["year"].map(_format_year_label)

        if filtered.empty:
            raise ValueError(
                f"No valid numeric values found for indicator_id='{indicator_id}'"
            )

        if len(filtered) < 3:
            raise ValueError(
                f"At least three valid observations are needed for summary "
                f"calculations for indicator_id='{indicator_id}'"
            )

        if value_type == "proportion":
            filtered["display_value"] = filtered["value"] * 100
        else:
            filtered["display_value"] = filtered["value"]

        if filtered["sort_year"].notna().all():
            filtered["time"] = filtered["sort_year"]
            time_label = "year"
        else:
            filtered["time"] = range(len(filtered))
            time_label = "time period"

        indicator_name = (
            filtered["indicator_name"].dropna().iloc[0]
            if "indicator_name" in filtered.columns
            and not filtered["indicator_name"].dropna().empty
            else str(indicator_id)
        )

        source_name = (
            filtered["source_name"].dropna().iloc[0]
            if "source_name" in filtered.columns
            and not filtered["source_name"].dropna().empty
            else "Source not available"
        )

        return {
            "indicator_id": str(indicator_id),
            "indicator_name": indicator_name,
            "source_name": source_name,
            "time_label": time_label,
            "data": filtered,
        }

    indicators = [
        prepare_indicator(indicator_id, selected_source_id)
        for indicator_id, selected_source_id in zip(indicator_ids, source_ids)
    ]

    plot_labels = (
        list(labels)
        if labels is not None
        else [indicator["indicator_name"] for indicator in indicators]
    )
    compact_layout = figsize[0] <= 9 or figsize[0] < figsize[1]
    legend_label_width = 34 if compact_layout else 58
    legend_labels = [
        textwrap.fill(label, width=legend_label_width)
        for label in plot_labels
    ]

    all_years = pd.concat(
        [
            indicator["data"][
                [
                    "year",
                    "sort_year",
                    "year_label",
                    "period_start_year",
                    "period_exact_key",
                    "period_label",
                    "period_is_financial",
                ]
            ]
            for indicator in indicators
        ],
        ignore_index=True,
    )
    def x_period_key(row):
        if (
            x_axis_alignment == "period_start"
            and pd.notna(row["period_start_year"])
        ):
            return f"start:{int(row['period_start_year'])}"

        return row["period_exact_key"]

    all_years["x_period_key"] = all_years.apply(x_period_key, axis=1)

    any_financial_years = all_years["period_is_financial"].any()

    if x_axis_alignment == "period_start" and any_financial_years:
        all_years["x_period_label"] = all_years["period_start_year"].map(
            lambda year: (
                f"{int(year)}/{str(int(year) + 1)[-2:]}"
                if pd.notna(year)
                else ""
            )
        )
    else:
        all_years["x_period_label"] = all_years["period_label"]

    all_years = (
        all_years
        .drop_duplicates(subset=["x_period_key"])
        .sort_values(["period_start_year", "x_period_label"], na_position="last")
        .reset_index(drop=True)
    )
    year_positions = {
        year: position for position, year in enumerate(all_years["x_period_key"])
    }

    def calculate_summary(indicator):
        data = indicator["data"]

        start_row = data.iloc[0]
        previous_row = data.iloc[-2]
        end_row = data.iloc[-1]

        start_time = float(start_row["time"])
        end_time = float(end_row["time"])

        start_value = float(start_row["display_value"])
        previous_value = float(previous_row["display_value"])
        end_value = float(end_row["display_value"])

        periods_elapsed = end_time - start_time
        absolute_change = end_value - start_value

        if value_type == "proportion":
            total_percentage_change = absolute_change
        elif start_value == 0:
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
        elif previous_value == 0:
            last_year_percentage_change = np.nan
        else:
            last_year_percentage_change = (
                (end_value - previous_value) / previous_value
            ) * 100

        return {
            "start_year_label": start_row["year_label"],
            "end_year_label": end_row["year_label"],
            "observations": len(data),
            "start_value": start_value,
            "previous_value": previous_value,
            "end_value": end_value,
            "absolute_change": absolute_change,
            "total_percentage_change": total_percentage_change,
            "average_annual_change": average_annual_change,
            "last_year_percentage_change": last_year_percentage_change,
        }

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

    def print_summary(indicator, summary, label):
        print("\n" + "=" * 80)
        print(
            f"Summary calculations for {indicator['indicator_id']}: "
            f"{indicator['indicator_name']}"
        )
        print("=" * 80)
        print(f"Series label: {label}")
        print(
            f"Period: {summary['start_year_label']} to "
            f"{summary['end_year_label']}"
        )
        print(f"Observations: {summary['observations']}")

        print("\nStart-to-end change")
        print("-" * 80)
        print(f"Start value: {value_formatter(summary['start_value'])}")
        print(f"End value: {value_formatter(summary['end_value'])}")
        print(
            f"Absolute change: "
            f"{printed_change_formatter(summary['absolute_change'])}"
        )

        total_change = summary["total_percentage_change"]
        if not np.isnan(total_change):
            if value_type == "proportion":
                print(
                    "Total percentage-point change: "
                    f"{total_change:+,.1f} percentage points"
                )
            else:
                print(f"Total percentage change: {total_change:,.1f}%")
        else:
            print("Total percentage change: not available because start value is zero")

        annual_change = summary["average_annual_change"]
        if not np.isnan(annual_change):
            if value_type == "proportion":
                print(
                    "Average annual percentage-point change: "
                    f"{annual_change:+,.2f} percentage points per "
                    f"{indicator['time_label']}"
                )
            else:
                print(
                    "Average annual change (CAGR-style): "
                    f"{annual_change:,.2f}% per {indicator['time_label']}"
                )
        else:
            print("Average annual change: not available")

        print("\nLatest annual change")
        print("-" * 80)
        print(f"Previous value: {value_formatter(summary['previous_value'])}")
        print(
            "Latest annual "
            f"{'percentage-point' if value_type == 'proportion' else 'percentage'} "
            f"change: {change_formatter(summary['last_year_percentage_change'])}"
        )
        print("=" * 80 + "\n")

    summaries = [calculate_summary(indicator) for indicator in indicators]

    for indicator, summary, label in zip(indicators, summaries, plot_labels):
        print_summary(indicator, summary, label)

    def add_plot_values(indicator):
        data = indicator["data"]

        if comparison_mode == "index":
            baseline_value = float(data["display_value"].iloc[0])

            if baseline_value <= 0:
                raise ValueError(
                    "comparison_mode='index' requires each indicator to have "
                    "a positive first observation"
                )

            data["plot_value"] = (data["display_value"] / baseline_value) * 100
            indicator["plot_baseline"] = baseline_value
        else:
            data["plot_value"] = data["display_value"]
            indicator["plot_baseline"] = None

    for indicator in indicators:
        add_plot_values(indicator)

    all_plot_values = pd.concat(
        [indicator["data"]["plot_value"] for indicator in indicators],
        ignore_index=True,
    )
    computed_y_limits = _get_contextual_y_limits(
        all_plot_values,
        value_type="count" if comparison_mode == "index" else value_type,
        y_axis=y_axis,
        y_limits=y_limits,
    )

    plt.rcParams["font.family"] = "Roboto Slab"

    fig, ax = plt.subplots(figsize=figsize)
    colours = _brand_line_colours(expected_series_count)
    default_point_label_offsets = tuple(
        (0, 10 + 7 * (index // 2))
        if index % 2 == 0
        else (0, -16 - 7 * (index // 2))
        for index in range(expected_series_count)
    )
    fixed_point_label_offsets = (
        point_label_offsets
        or default_point_label_offsets[:expected_series_count]
    )

    def should_label_point(position, point_count):
        if point_labels == "none":
            return False
        if point_labels == "all":
            return True
        if point_labels == "latest":
            return position == point_count - 1
        if point_labels == "endpoints":
            return position in {0, point_count - 1}
        return False

    plot_data = []
    for indicator in indicators:
        data = indicator["data"].copy()
        data["x_period_key"] = data.apply(x_period_key, axis=1)
        data["plot_x"] = data["x_period_key"].map(year_positions)
        plot_data.append(data)

    series_values_by_period = [
        dict(zip(data["x_period_key"], data["plot_value"]))
        for data in plot_data
    ]
    summary_changes = [
        summary["total_percentage_change"] for summary in summaries
    ]
    finite_summary_changes = [
        change if np.isfinite(change) else -np.inf
        for change in summary_changes
    ]
    preferred_series_order = list(np.argsort(finite_summary_changes)[::-1])

    def smart_point_label_position(series_index, row, position, point_count):
        if point_label_offsets is not None or point_label_position == "fixed":
            xytext = fixed_point_label_offsets[series_index]
            return {
                "xytext": xytext,
                "ha": "center",
                "va": "bottom" if xytext[1] >= 0 else "top",
            }

        peer_values = [
            values.get(row["x_period_key"])
            for peer_index, values in enumerate(series_values_by_period)
            if peer_index != series_index
        ]
        peer_values = [
            value for value in peer_values if value is not None and pd.notna(value)
        ]

        if peer_values:
            nearest_peer_distance = min(
                abs(row["plot_value"] - peer_value)
                for peer_value in peer_values
            )

            if nearest_peer_distance < 8:
                close_series = [series_index]
                for peer_index, values in enumerate(series_values_by_period):
                    if peer_index == series_index:
                        continue

                    peer_value = values.get(row["x_period_key"])
                    if (
                        peer_value is not None
                        and pd.notna(peer_value)
                        and abs(row["plot_value"] - peer_value) < 8
                    ):
                        close_series.append(peer_index)

                ranked_close_series = [
                    index
                    for index in preferred_series_order
                    if index in close_series
                ]
                rank = ranked_close_series.index(series_index)
                if len(ranked_close_series) == 2:
                    stacked_offsets = [24, -28]
                    y_offset = stacked_offsets[rank]
                else:
                    above_count = math.ceil(len(ranked_close_series) / 2)

                    if rank < above_count:
                        y_offset = 24 + 16 * (above_count - rank - 1)
                    else:
                        y_offset = -28 - 16 * (rank - above_count)
            else:
                label_above = row["plot_value"] >= np.median(peer_values)
                y_offset = 14 if label_above else -18
        else:
            label_above = series_index == 0
            y_offset = 14 if label_above else -18

        x_offset = 0
        ha = "center"

        if point_count > 1 and position == 0:
            x_offset = 8
            ha = "left"
        elif point_count > 1 and position == point_count - 1:
            x_offset = -8
            ha = "right"

        return {
            "xytext": (x_offset, y_offset),
            "ha": ha,
            "va": "bottom" if y_offset >= 0 else "top",
        }

    line_handles = []

    for series_index, (data, label, colour) in enumerate(zip(
        plot_data,
        legend_labels,
        colours,
    )):
        line_handle = ax.plot(
            data["plot_x"],
            data["plot_value"],
            color=colour,
            linewidth=3.5,
            marker="o",
            markersize=9,
            label=label,
        )[0]
        line_handles.append(line_handle)

        for position, (_, row) in enumerate(data.iterrows()):
            if not should_label_point(position, len(data)):
                continue

            point_label = value_formatter(row["display_value"])
            label_position = smart_point_label_position(
                series_index,
                row,
                position,
                len(data),
            )
            ax.annotate(
                point_label,
                xy=(row["plot_x"], row["plot_value"]),
                xytext=label_position["xytext"],
                textcoords="offset points",
                fontsize=point_label_fontsize,
                fontweight="bold",
                color="black",
                ha=label_position["ha"],
                va=label_position["va"],
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.9,
                ),
            )

    legend_order = sorted(
        range(expected_series_count),
        key=lambda index: plot_data[index]["plot_value"].dropna().iloc[-1],
        reverse=True,
    )
    ordered_line_handles = [line_handles[index] for index in legend_order]
    ordered_legend_labels = [legend_labels[index] for index in legend_order]

    if computed_y_limits is not None:
        ax.set_ylim(computed_y_limits)

    plot_title = (
        title
        if title
        else " and ".join(indicator["indicator_name"] for indicator in indicators)
    )

    resolved_legend_position = legend_position
    if legend_position == "smart":
        resolved_legend_position = "top" if compact_layout else "inside"

    title_fontsize = 17 if compact_layout else 22
    title_pad = 25
    if compact_layout and resolved_legend_position == "top":
        title_pad = 52 + max(0, expected_series_count - 2) * 12
    axis_label_fontsize = 11 if compact_layout else 12
    tick_labelsize = 11 if compact_layout else 16
    footer_fontsize = 7 if compact_layout else 9

    ax.set_title(plot_title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xlabel("Year", fontsize=axis_label_fontsize)
    ax.set_ylabel(plot_y_label, fontsize=axis_label_fontsize, labelpad=16)
    ax.tick_params(axis="both", labelsize=tick_labelsize)

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: plot_value_formatter(x))
    )

    plot_x = np.arange(len(all_years))
    ax.set_xticks(plot_x)
    ax.set_xticklabels(all_years["x_period_label"], rotation=45, ha="right")

    legend_avoid_corners = set()
    legend_top_margin = 1

    if resolved_legend_position == "inside":
        ax.legend(
            ordered_line_handles,
            ordered_legend_labels,
            loc="upper right",
            frameon=False,
            fontsize=11,
        )
        legend_avoid_corners.add("top_right")
    elif resolved_legend_position == "top":
        ax.legend(
            ordered_line_handles,
            ordered_legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            frameon=False,
            fontsize=8 if compact_layout else 11,
            ncol=1 if compact_layout else expected_series_count,
            borderaxespad=0,
        )
        if compact_layout:
            legend_top_margin = max(0.58, 0.80 - expected_series_count * 0.045)
        else:
            legend_top_margin = 0.90

    def overall_change_label(value):
        if np.isnan(value) or value == 0:
            change_label = "change"
        elif value > 0:
            change_label = "increase"
        else:
            change_label = "decrease"

        return f"overall {change_label}: {change_formatter(value)}"

    annotation_lines = [
        overall_change_label(summary["total_percentage_change"])
        for summary in summaries
    ]
    annotation_text = "\n".join(annotation_lines)
    annotation_entries = [
        (colours[index], annotation_lines[index])
        for index in legend_order
    ]

    def add_summary_note(
        ax,
        text,
        entries,
        x_values,
        y_values,
        *,
        fontsize=11,
        position="smart",
        avoid_corners=None,
    ):
        if position == "none":
            return

        avoid_corners = avoid_corners or set()
        candidates = {
            "top_left": (0.02, 0.95, "left", "top"),
            "top_right": (0.98, 0.95, "right", "top"),
            "bottom_left": (0.02, 0.05, "left", "bottom"),
            "bottom_right": (0.98, 0.05, "right", "bottom"),
        }

        if position == "smart":
            candidate_items = [
                item for item in candidates.items() if item[0] not in avoid_corners
            ]
        else:
            candidate_items = [(position, candidates[position])]

        x = pd.Series(x_values).astype(float)
        y = pd.Series(y_values).astype(float)
        valid_points = pd.DataFrame({"x": x, "y": y}).dropna()

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        if x_max == x_min:
            valid_points["x_norm"] = 0.5
        else:
            valid_points["x_norm"] = (
                (valid_points["x"] - x_min) / (x_max - x_min)
            )

        if y_max == y_min:
            valid_points["y_norm"] = 0.5
        else:
            valid_points["y_norm"] = (
                (valid_points["y"] - y_min) / (y_max - y_min)
            )

        max_line_length = max(len(line) for line in text.split("\n"))
        box_width = min(0.52, max(0.30, 0.012 * max_line_length))
        box_height = 0.10 + 0.05 * max(1, text.count("\n") + 1)

        def overlap_score(candidate):
            corner, (x0, y0, ha, va) = candidate

            if ha == "left":
                left, right = x0, x0 + box_width
            else:
                left, right = x0 - box_width, x0

            if va == "top":
                bottom, top = y0 - box_height, y0
            else:
                bottom, top = y0, y0 + box_height

            pad = 0.04
            score = np.sum(
                (valid_points["x_norm"] >= left - pad)
                & (valid_points["x_norm"] <= right + pad)
                & (valid_points["y_norm"] >= bottom - pad)
                & (valid_points["y_norm"] <= top + pad)
            )

            if corner in avoid_corners:
                score += 100

            return score

        _, (x0, y0, ha, va) = min(candidate_items, key=overlap_score)

        rows = []
        for colour, entry_text in entries:
            line_area = DrawingArea(26, 10, 0, 0)
            line_area.add_artist(
                Line2D(
                    [2, 24],
                    [5, 5],
                    color=colour,
                    linewidth=3.5,
                    marker="o",
                    markersize=5,
                )
            )
            rows.append(
                HPacker(
                    children=[
                        line_area,
                        TextArea(
                            entry_text,
                            textprops={
                                "color": "black",
                                "fontsize": fontsize,
                                "fontweight": "bold",
                            },
                        ),
                    ],
                    align="center",
                    pad=0,
                    sep=6,
                )
            )

        note_box = VPacker(
            children=rows,
            align="left",
            pad=0,
            sep=4,
        )
        box_alignment = (
            0 if ha == "left" else 1,
            1 if va == "top" else 0,
        )

        ax.add_artist(
            AnnotationBbox(
                note_box,
                (x0, y0),
                xycoords=ax.transAxes,
                box_alignment=box_alignment,
                frameon=True,
                bboxprops=dict(
                    boxstyle="round,pad=0.35",
                    facecolor="white",
                    edgecolor="#cccccc",
                    alpha=0.9,
                ),
            )
        )

    all_plot_x_values = pd.concat(
        [data["plot_x"] for data in plot_data],
        ignore_index=True,
    )
    add_summary_note(
        ax,
        annotation_text,
        annotation_entries,
        all_plot_x_values,
        all_plot_values,
        fontsize=8 if compact_layout else 11,
        position=summary_note_position,
        avoid_corners=legend_avoid_corners,
    )

    source_entries = [
        (label, indicator["source_name"])
        for label, indicator in zip(plot_labels, indicators)
    ]
    unique_source_names = {source_name for _, source_name in source_entries}

    if footer_note is not None:
        footer_text = textwrap.fill(footer_note.strip(), width=85)
    else:
        footer_lines = []

        if note:
            wrapped_note = textwrap.fill(note, width=85)
            footer_lines.append(f"Note: {wrapped_note}")
            footer_lines.append("")

        if len(unique_source_names) == 1:
            footer_lines.append(f"Source: {indicators[0]['source_name']}")
        else:
            footer_lines.append(
                "Sources: "
                + "; ".join(
                    f"{label}: {source_name}"
                    for label, source_name in source_entries
                )
            )

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

        if x_axis_alignment == "period_start":
            footer_lines.append(
                "X-axis method: periods are aligned by start year, so calendar "
                "years and financial years with the same start year share a tick."
            )

        if comparison_mode == "index":
            footer_lines.append(
                "Comparison method: each line is indexed to its first observation "
                "= 100, so the y-axis shows relative change rather than original "
                "values."
            )

        footer_text = "\n".join(footer_lines)

    if show_footer_note:
        fig.text(
            0.01,
            0.02,
            footer_text,
            ha="left",
            va="bottom",
            fontsize=footer_fontsize,
            color="#444444",
            linespacing=1.4,
        )

    bottom_margin = 0.10 if show_footer_note else 0.03
    plt.tight_layout(rect=[0, bottom_margin, 1, legend_top_margin])

    if save_path:
        save_path = Path(save_path)

        if not save_path.is_absolute():
            save_path = OUTPUT_DIR / save_path

        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)

    plt.show()


def plot_two_indicators(
    indicator_id_1: str,
    indicator_id_2: str,
    indicator_id_3: str | None = None,
    source_id: str | int | None = None,
    source_id_1: str | int | None = None,
    source_id_2: str | int | None = None,
    source_id_3: str | int | None = None,
    **kwargs,
) -> None:
    """Backward-compatible wrapper for ploting_multiple_indicators."""
    indicator_ids = [indicator_id_1, indicator_id_2]
    source_ids = [source_id_1, source_id_2]

    if indicator_id_3 is not None:
        indicator_ids.append(indicator_id_3)
        source_ids.append(source_id_3)

    ploting_multiple_indicators(
        indicator_ids=indicator_ids,
        source_id=source_id,
        source_ids=source_ids,
        **kwargs,
    )

# --- PLOT FUNCTION: CSEW CHILD VICTIMISATION LINE CHART ---

def plot_csew_child_victimisation_trends(
    data: Path | str | pd.DataFrame,
    title: str | None = "Child victims of crime by offence type",
    x_label: str = "Year",
    y_label: str | None = None,
    year_column: str = "year",
    crime_columns: list[str] | tuple[str, ...] | None = None,
    figsize: tuple = (15, 9),
    y_axis: str = "zero",
    y_limits: tuple[float, float] | None = None,
    chart_style: str = "overlaid",
    baseline_year: str | None = None,
    band_padding: float = 20,
    category_label_width: int = 32,
    x_tick_rotation: float = 30,
    colour_map: str = "tab20",
    save_path: Path | str | None = None,
):
    """Plot all CSEW child-victimisation crime types as lines on one chart.

    Each crime type is a separate coloured line, with the table's victim
    counts on the shared y-axis and the years on the x-axis. Set
    ``chart_style="offset_index"`` to set every crime type to 100 in a
    baseline year and place its indexed line in a separate horizontal band.

    In ``offset_index`` mode, the y-axis labels identify crime types and the
    value labels at the final point show ``actual count (change from baseline)``.
    This avoids overlap while retaining comparable percentage change from the
    baseline.
    """

    if isinstance(data, pd.DataFrame):
        plotted_data = data.copy()
    else:
        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"CSEW data file not found: {data_path}")
        plotted_data = pd.read_excel(data_path)

    if year_column not in plotted_data.columns:
        raise ValueError(f"The data must include a '{year_column}' column")

    if crime_columns is None:
        start_column = "Violence"
        end_column = "Criminal Demage to Personal Property"
        if start_column not in plotted_data.columns or end_column not in plotted_data.columns:
            raise ValueError(
                "Could not find the expected CSEW crime columns from "
                f"'{start_column}' through '{end_column}'"
            )

        start_index = plotted_data.columns.get_loc(start_column)
        end_index = plotted_data.columns.get_loc(end_column)
        crime_columns = list(plotted_data.columns[start_index : end_index + 1])
    else:
        crime_columns = list(crime_columns)

    missing_columns = [
        column for column in crime_columns if column not in plotted_data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Crime columns not found in the data: " + ", ".join(missing_columns)
        )

    if not crime_columns:
        raise ValueError("Provide at least one crime column to plot")

    valid_chart_styles = {"overlaid", "offset_index"}
    if chart_style not in valid_chart_styles:
        raise ValueError(
            "chart_style must be one of: " + ", ".join(sorted(valid_chart_styles))
        )

    if band_padding < 0:
        raise ValueError("band_padding cannot be negative")

    if category_label_width < 1:
        raise ValueError("category_label_width must be at least 1")

    plotted_data = plotted_data[[year_column, *crime_columns]].copy()
    plotted_data[year_column] = plotted_data[year_column].astype(str)
    plotted_data["_sort_year"] = pd.to_numeric(
        plotted_data[year_column].str.extract(r"(\d{4})")[0],
        errors="coerce",
    )
    plotted_data = plotted_data.sort_values(
        ["_sort_year", year_column], na_position="last"
    ).reset_index(drop=True)

    for column in crime_columns:
        plotted_data[column] = pd.to_numeric(plotted_data[column], errors="coerce")

    plt.rcParams["font.family"] = "Roboto Slab"
    fig, ax = plt.subplots(figsize=figsize)
    plot_x = np.arange(len(plotted_data))
    colours = plt.get_cmap(colour_map)(np.linspace(0, 1, len(crime_columns)))

    if chart_style == "overlaid":
        all_values = plotted_data[crime_columns].stack().dropna()
        if all_values.empty:
            raise ValueError("No numeric victim-count values found in the crime columns")

        computed_y_limits = _get_contextual_y_limits(
            all_values,
            value_type="count",
            y_axis=y_axis,
            y_limits=y_limits,
        )

        for crime, colour in zip(crime_columns, colours):
            ax.plot(
                plot_x,
                plotted_data[crime],
                color=colour,
                linewidth=2.25,
                marker="o",
                markersize=5,
                label=crime,
            )

        if computed_y_limits is not None:
            ax.set_ylim(computed_y_limits)

        ax.set_ylabel(
            y_label or "Number of child victims",
            fontsize=12,
            labelpad=12,
        )
        ax.ticklabel_format(style="plain", axis="y")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:,.0f}")
        )
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.legend(
            title="Crime type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0, 0.78, 1])
    else:
        baseline_year = baseline_year or plotted_data[year_column].iloc[0]
        baseline_rows = plotted_data[plotted_data[year_column] == str(baseline_year)]
        if baseline_rows.empty:
            available_years = ", ".join(plotted_data[year_column])
            raise ValueError(
                f"Baseline year '{baseline_year}' was not found. "
                f"Available years: {available_years}"
            )

        baseline_values = baseline_rows.iloc[0][crime_columns]
        invalid_baseline_columns = [
            crime for crime in crime_columns
            if pd.isna(baseline_values[crime]) or baseline_values[crime] == 0
        ]
        if invalid_baseline_columns:
            raise ValueError(
                "The baseline year must have non-zero values for every crime type. "
                "Invalid columns: " + ", ".join(invalid_baseline_columns)
            )

        indexed_data = plotted_data[crime_columns].divide(baseline_values).multiply(100)
        index_min = indexed_data.min().min()
        index_max = indexed_data.max().max()
        band_height = (index_max - index_min) + (2 * band_padding)
        if band_height <= 0:
            band_height = 1

        offsets = np.arange(len(crime_columns))[::-1] * band_height
        baseline_positions = offsets + 100

        for crime, offset in zip(crime_columns, offsets):
            indexed_values = indexed_data[crime]
            line_colour = "#ff4d5e" if crime == "Violence without injury" else "#b3b3b3"
            ax.plot(
                plot_x,
                indexed_values + offset,
                color=line_colour,
                linewidth=2.25,
                marker="o",
                markersize=5,
            )

            latest_value = plotted_data[crime].iloc[-1]
            percentage_change = ((latest_value / baseline_values[crime]) - 1) * 100
            label = HPacker(
                children=[
                    TextArea(
                        f"{latest_value:,.0f}",
                        textprops={"color": "#000000", "fontsize": 8},
                    ),
                    TextArea(
                        f" ({percentage_change:+.0f}%)",
                        textprops={"color": "#000000", "fontsize": 8},
                    ),
                ],
                align="center",
                pad=0,
                sep=0,
            )
            ax.add_artist(
                AnnotationBbox(
                    label,
                    (plot_x[-1], indexed_values.iloc[-1] + offset),
                    xybox=(7, 0),
                    xycoords="data",
                    boxcoords="offset points",
                    box_alignment=(0, 0.5),
                    frameon=False,
                )
            )

        ax.set_yticks(baseline_positions)
        ax.set_yticklabels(
            [textwrap.fill(crime, width=category_label_width) for crime in crime_columns]
        )
        ax.set_ylabel(
            y_label or f"Crime type (baseline {baseline_year} = 100)",
            fontsize=12,
            labelpad=12,
        )
        ax.set_xlim(plot_x[0] - 0.1, plot_x[-1] + 0.8)
        fig.tight_layout()

    if title:
        ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_xticks(plot_x)
    ax.set_xticklabels(
        plotted_data[year_column],
        rotation=x_tick_rotation,
        ha="right",
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if chart_style == "offset_index":
        fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        if not save_path.is_absolute():
            save_path = OUTPUT_DIR / save_path
        fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=False)

    plt.show()
    return fig, ax, plotted_data.drop(columns="_sort_year")


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
    filtered["year_label"] = filtered["year"].map(_format_year_label)

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

    year_1 = filtered["year_label"].iloc[0]
    year_2 = filtered["year_label"].iloc[1]
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
    fontsize: int = 10,
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
    box_width = 0.40
    box_height = 0.07 + (0.043 * line_count)

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
            boxstyle="round,pad=0.55",
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
    show_smart_note: bool = True,
    show_footer_note: bool = True,
    show_methodology_label: bool = True,
    show_post_methodology_shading: bool = True,
    show: bool = True,
) -> dict:
    """
    Plot an HBAI time-series graph with a visible methodology-change break.

    Total and average annual change are calculated for the
    pre-methodology-change period. Latest annual change is calculated for the
    post-methodology-change period.
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
    filtered["year_label"] = filtered["year"].map(_format_year_label)

    missing_years = set(plot_years) - set(filtered["year"])
    if missing_years:
        raise ValueError(f"Missing expected HBAI years: {sorted(missing_years)}")

    filtered["x"] = np.arange(len(filtered))
    pre = filtered[filtered["year"].isin(pre_methodology_years)].copy()
    post = filtered[filtered["year"].isin(post_methodology_years)].copy()

    if len(pre) < 3:
        raise ValueError("At least three pre-methodology observations are needed")
    if len(post) < 2:
        raise ValueError("At least two post-methodology observations are needed")

    start_row = pre.iloc[0]
    end_row = pre.iloc[-1]
    post_previous_row = post.iloc[-2]
    post_latest_row = post.iloc[-1]
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

    if post_previous_row["value"] == 0:
        latest_annual_percentage_change = np.nan
    else:
        latest_annual_percentage_change = (
            (post_latest_row["value"] - post_previous_row["value"])
            / post_previous_row["value"]
        ) * 100

    def percentage_formatter(value):
        if np.isnan(value):
            return "n/a"
        return f"{value:+.1f}%"

    print("\n" + "=" * 80)
    print(
        f"Summary calculations for {indicator_id}: "
        f"{geography_id}, split by methodology period"
    )
    print("=" * 80)
    print("\nBefore methodology change calculations")
    print("-" * 80)
    print(f"Period: {start_row['year_label']} to {end_row['year_label']}")
    print(f"Observations: {len(pre)}")
    print(f"Start value: {start_row['value']:.3f}m")
    print(f"End value: {end_row['value']:.3f}m")
    print(f"Absolute change: {absolute_change:.3f}m")
    print(f"Total percentage change: {percentage_formatter(percentage_change)}")
    print(
        f"Average annual change (CAGR-style): "
        f"{percentage_formatter(average_annual_change)} per period"
    )
    print("\nAfter methodology change calculations")
    print("-" * 80)
    print(
        f"Period: {post_previous_row['year_label']} "
        f"to {post_latest_row['year_label']}"
    )
    print(f"Observations: {len(post)}")
    print(f"Previous value: {post_previous_row['value']:.3f}m")
    print(f"Latest value: {post_latest_row['value']:.3f}m")
    print(
        f"Latest annual percentage change: "
        f"{percentage_formatter(latest_annual_percentage_change)}"
    )
    print("=" * 80 + "\n")

    plt.rcParams["font.family"] = "Roboto Slab"
    fig, ax = plt.subplots(figsize=figsize)

    cutoff_x = len(pre_methodology_years) - 0.5
    if show_post_methodology_shading:
        ax.axvspan(
            cutoff_x,
            len(plot_years) - 0.5,
            color="#5a97bb",
            alpha=0.08,
            linewidth=0,
        )
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

    if show_methodology_label:
        ax.text(
            cutoff_x + 0.08,
            ax.get_ylim()[1],
            f"Methodology changed\nfrom {_format_year_label(post_methodology_years[0])}",
            ha="left",
            va="top",
            fontsize=11,
            color="#444444",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=4),
        )

    stats_text = (
        "Before methodology change calculations\n"
        f"({start_row['year_label']} until {end_row['year_label']})\n"
        f"- Total change: {percentage_formatter(percentage_change)}\n"
        f"- Average Annual Change: {percentage_formatter(average_annual_change)}\n"
        "After methodology change calculations\n"
        f"({post_previous_row['year_label']} until {post_latest_row['year_label']})\n"
        "- Latest annual change: "
        f"{percentage_formatter(latest_annual_percentage_change)}"
    )

    if show_smart_note:
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
            fontsize=10,
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
    ax.set_xticklabels(filtered["year_label"], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.1f}m"))

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper left")

    omitted_text = ", ".join(_format_year_label(year) for year in omitted_years)
    footer_text = (
        f"Note: {omitted_text} is omitted because it is not available in the "
        "source file. Values from "
        f"{_format_year_label(post_methodology_years[0])} onward are shown after the methodology "
        "change and should not be treated as directly comparable with the "
        "earlier trend.\n"
        f"Source: {source_name}. Summary method: total percentage change, "
        "CAGR-style average annual change for the previous methodology "
        "period, and latest annual percentage change for the new methodology "
        "period."
    )
    if show_footer_note:
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

    bottom_margin = 0.12 if show_footer_note else 0.03
    plt.tight_layout(rect=[0, bottom_margin, 1, 1])

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
            "last_year_percentage_change": latest_annual_percentage_change,
            "latest_annual_percentage_change": latest_annual_percentage_change,
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
