from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import textwrap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

# =========================================================
# FILE PATHS
# =========================================================
OUTPUT_DIR = Path("graphs")

OUTPUT_DIR.mkdir(exist_ok=True)

# =========================================================
# GLOBAL PLOT STYLE CONSTANTS
# =========================================================
   
COLOR_STEALTH = "#1f4ca5"
COLOR_ACTION = "#dc3912"
COLOR_ADAPTIVE = "#ffa927"

COLOR_NEUTRAL = "#9D9D9D"
COLOR_GRID = "#D9D9D9"
COLOR_TEXT = "#222222"
COLOR_LINE = "black"
COLOR_PIE = "white"

MODE_COLORS = {
    "Stealth": COLOR_STEALTH,
    "Action": COLOR_ACTION,
    "Adaptive": COLOR_ADAPTIVE,
}

MODE_ORDER = ["Stealth", "Action", "Adaptive"]

FIG_WIDTH = 8
FIG_HEIGHT = 5
DPI = 300

BAR_ALPHA = 0.9
LINE_WIDTH = 0.8
LINE_WIDTH_PIE = 0.8
LINE_WIDTH_ERROR = 0.8
MARKER_SIZE = 7
GRID_ALPHA = 0.35
ERROR_CAPSIZE = 5
TITLE_FONT_SIZE = 20
LABEL_FONT_SIZE = 12
FONT_FAMILY = "Times New Roman"
TRANSPARENT_BG = False
BG = 'white'

LEGEND_FONT_SIZE = 10
LEGEND_MARKER_SIZE = 10
LEGEND_LOCATION = "upper left"
LEGEND_BBOX_TO_ANCHOR = (1.02, 0.85)
LEGEND_FRAME = False

plt.rcParams["font.family"] = FONT_FAMILY

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def save_figure(fig, filename: str) -> None:
    """Save and close a matplotlib figure."""
    out_path = OUTPUT_DIR / filename 
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", transparent=TRANSPARENT_BG, facecolor=BG)
    plt.close(fig)
    print(f"Saved: {out_path}")


def mean_and_sem(values):
    """Return mean and standard error of the mean."""
    clean = pd.Series(values).dropna()
    if len(clean) == 0:
        return np.nan, np.nan
    mean_val = clean.mean()
    sem_val = clean.std(ddof=1) / np.sqrt(len(clean)) if len(clean) > 1 else 0.0
    return mean_val, sem_val

def mode_columns(metric_name: str, modes=["Stealth", "Action", "Adaptive"], as_list: bool = False):
    """
    Generate column names in the format {mode}_{metric_name}.
    
    Returns a dictionary mapping modes to column names,
    or a list of column names if as_list=True.
    """
    if as_list:
        l = []
        for mode in modes:
            l.append(f"{mode}_{metric_name}")
        return l
    return {mode: f"{mode}_{metric_name}" for mode in modes}

def wrap_text(text: str, width: int | None = None) -> str:
    """
    Wrap text into multiple lines.
    If width is None or <= 0, return text unchanged.
    """
    if width is None or width <= 0:
        return text
    return "\n".join(textwrap.wrap(str(text), width=width))

# -----------------------------------------------------------------
#                              Legend
# -----------------------------------------------------------------
def add_discrete_legend(
    ax,
    labels: list[str],
    answer_colors: dict[str, str] | None = None,
    answer_order: list[str] | None = None,
    include_zero_answers: bool = False,
    counts: pd.Series | dict | None = None,
):
    """
    Add a consistently styled legend for plots with discrete categories.

    Parameters
    ----------
    ax : matplotlib axis
        Axis to attach the legend to.
    labels : list[str]
        Labels currently present in the plotted data.
    answer_colors : dict[str, str] | None
        Mapping from answer text to color.
    answer_order : list[str] | None
        Optional full answer list in preferred order.
        Useful when some answers have zero responses.
    include_zero_answers : bool
        If True and answer_order is given, include all answers in the legend
        even if some have zero responses.
    counts : pd.Series | dict | None
        Optional counts per answer. If given and include_zero_answers is False,
        answers with zero count can be filtered out cleanly.
    Returns
    -------
    matplotlib.legend.Legend
        The created legend object.
    """
    # Decide which labels should appear in the legend
    if answer_order is not None:
        if include_zero_answers:
            legend_labels = answer_order
        else:
            if counts is None:
                legend_labels = [label for label in answer_order if label in labels]
            else:
                if isinstance(counts, pd.Series):
                    counts_dict = counts.to_dict()
                else:
                    counts_dict = dict(counts)
                legend_labels = [label for label in answer_order if counts_dict.get(label, 0) > 0]
    else:
        legend_labels = labels
    
    # Use provided colors when available, otherwise fall back to global colors
    if answer_colors is None:
        global_palette = [
            COLOR_STEALTH,
            COLOR_ACTION,
            COLOR_ADAPTIVE,
            COLOR_NEUTRAL,
        ]
        legend_colors = {
            label: global_palette[i % len(global_palette)]
            for i, label in enumerate(legend_labels)
        }
    else:
        legend_colors = {
            label: answer_colors.get(label, COLOR_NEUTRAL)
            for label in legend_labels
        }

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=LEGEND_MARKER_SIZE,
            markerfacecolor=legend_colors[label],
            markeredgecolor=legend_colors[label],
            label=label,
        )
        for label in legend_labels
    ]

    legend = ax.legend(
        handles=handles,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=LEGEND_FRAME,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=0.8,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)
        text.set_fontfamily(FONT_FAMILY)

    return legend

# -----------------------------------------------------------------
#                            Pie Plot 
# -----------------------------------------------------------------
def plot_pie(
    df: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    answer_order: list[str] | None = None,
    answer_colors: dict[str, str] | None = None,
    show_counts: bool = True,
    show_percentages: bool = True,
    use_legend: bool = True,
    legend_include_zero_answers: bool = True,
    wrap_label_width: int | None = 30,
    subtitle: str | None = None,
):
    """
    Create a pie chart for a discrete-response question.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    column : str
        Column containing the discrete answers.
    title : str
        Chart title.
    filename : str
        Output graphs/filename.
    answer_order : list[str] | None
        Optional fixed order for answers.
    answer_colors : dict[str, str] | None
        Optional mapping from answer text to color.
        If not provided, colors are assigned from the global color system.
    show_counts : bool
        Whether to include raw counts in labels.
    show_percentages : bool
        Whether to show percentages on slices.
    use_legend : bool
        Whether to show a legend.
    legend_include_zero_answers : bool
        Whether the legend should include answers with zero responses.
    wrap_label_width : int | None
        Character width used to wrap labels.
    subtitle : str | None
        Optional subtitle displayed below the title.
        """
    series = df[column].dropna().astype(str).str.strip()

    if answer_order is None:
        counts = series.value_counts()
    else:
        counts = series.value_counts().reindex(answer_order, fill_value=0)
        counts = counts[counts > 0]

    labels = counts.index.tolist()
    values = counts.values.tolist()

    if answer_colors is None:
        global_palette = [
            COLOR_STEALTH,
            COLOR_ACTION,
            COLOR_ADAPTIVE,
            COLOR_NEUTRAL,
        ]
        colors = [global_palette[i % len(global_palette)] for i in range(len(labels))]
    else:
        colors = [answer_colors.get(label, COLOR_NEUTRAL) for label in labels]
    
    # Wrapped labels
    wrapped_labels = [wrap_text(label, wrap_label_width) for label in labels]
    
    def autopct_func(pct):
        if not show_percentages:
            return ""
        return f"{pct:.1f}%"

    display_labels = []
    for label, value in zip(wrapped_labels, values):
        if show_counts:
            display_labels.append(f"{label}\n(n={value})")
        else:
            display_labels.append(label)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
       
    wedges, texts, autotexts = ax.pie(
        values,
        labels=display_labels,
        colors=colors,
        autopct=autopct_func,
        startangle=90,
        labeldistance=1.2,
        counterclock=False,
        wedgeprops={"edgecolor": COLOR_PIE, "linewidth": LINE_WIDTH_PIE},
        textprops={"color": COLOR_TEXT, 
                   "fontsize": LABEL_FONT_SIZE, 
                   "family": FONT_FAMILY},
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(LABEL_FONT_SIZE)

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=60
    )
    if subtitle:
        fig.text(
            0.4, 0.88,  # x, y position (adjust if needed)
            subtitle,
            ha="center",
            fontsize=TITLE_FONT_SIZE * 0.8,
            color=COLOR_TEXT,
            family=FONT_FAMILY,
            fontstyle="italic",
        )
    ax.axis("equal")
    
    if use_legend:
        add_discrete_legend(
            ax=ax,
            labels=labels,
            answer_colors=answer_colors,
            answer_order=answer_order,
            include_zero_answers=legend_include_zero_answers,
            counts=counts,
        )

    save_figure(fig, filename)

# -----------------------------------------------------------------
#                           Bar Chart
# -----------------------------------------------------------------
def plot_bar(
    df: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    answer_order: list[str] | None = None,
    answer_colors: dict[str, str] | None = None,
    is_multiselect: bool = False,
    separator: str = ";",
    use_legend: bool = False,
    legend_include_zero_answers: bool = False,
    show_bar_labels: bool = True,
    xlabel: str | None = None,
    ylabel: str = "Number of Participants",
    rotate_xticks: int = 0,
    bar_width: float = 0.6,
    wrap_label_width: int | None = 10,
    label_rotation: int = 0,
    vertical_labels: bool = False,
    skewed_labels: bool = False,
    use_colors: bool = False,
    subtitle: str | None = None,
    sub_pos = (5, 21),
):
    """
    Plot a bar chart for either:
    - discrete single-answer data
    - multi-select data separated by a delimiter (default=';')

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    column : str
        Column to analyze.
    title : str
        Plot title.
    filename : str
        Output filename.
    answer_order : list[str] | None
        Optional full answer list in desired order.
    answer_colors : dict[str, str] | None
        Optional label -> color mapping.
    is_multiselect : bool
        If True, split each non-empty cell by separator and count all selections.
    separator : str
        Delimiter for multi-select answers.
    use_legend : bool
        Whether to add the shared legend.
    legend_include_zero_answers : bool
        If True and answer_order is given, include zero-count answers in the legend.
    show_bar_labels : bool
        Whether to draw count labels above bars.
    xlabel : str | None
        Optional x-axis label.
    ylabel : str
        Y-axis label.
    rotate_xticks : int
        Rotation angle for x tick labels.
    wrap_label_width : int | None
        Width used to wrap x-axis labels.
    label_rotation : int
        Custom rotation for labels.
    vertical_labels : bool
        If True, rotate labels vertically (90°).
    skewed_labels : bool
        If True, rotate labels diagonally (45°).
    use_colors : bool
        Whether to use multiple colors when available.
    subtitle : str | None
        Optional subtitle displayed on the plot.
    sub_pos : tuple
        Position of the subtitle text.
    """
    series = df[column].dropna().astype(str).str.strip()
    series = series[series != ""]

    if is_multiselect:
        answers = []
        for cell in series:
            parts = [part.strip() for part in cell.split(separator) if part.strip()]
            answers.extend(parts)
        counts = pd.Series(answers).value_counts()
    else:
        counts = series.value_counts()

    if answer_order is not None:
        counts = counts.reindex(answer_order, fill_value=0)

    plot_counts = counts if answer_order is not None else counts.sort_values(ascending=False)

    labels = plot_counts.index.tolist()
    values = plot_counts.values.tolist()

    # Color logic
    if answer_colors is not None:
        colors = [answer_colors.get(label, COLOR_NEUTRAL) for label in labels]
    else:
        palette_4 = [COLOR_STEALTH, COLOR_ACTION, COLOR_ADAPTIVE, COLOR_NEUTRAL]
        if len(labels) <= 4 and use_colors:
            colors = palette_4[:len(labels)]
        else:
            colors = [COLOR_STEALTH] * len(labels)

    # Rotation mode
    if vertical_labels:
        final_rotation = 90
        ha = "center"
    elif skewed_labels:
        final_rotation = 45
        ha = "right"
    else:
        final_rotation = label_rotation
        ha = "right" if label_rotation else "center"

    # Wrapped labels
    wrapped_labels = [wrap_text(label, wrap_label_width) for label in labels]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG)
    ax.set_facecolor(BG)  

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=colors,
        edgecolor=COLOR_LINE,
        linewidth=LINE_WIDTH,
    )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=50,
    )
    if subtitle:
        ax.text(
            sub_pos[0], sub_pos[1],  # x, y position (adjust if needed)
            subtitle,
            ha="center",
            fontsize=TITLE_FONT_SIZE * 0.8,
            color=COLOR_TEXT,
            family=FONT_FAMILY,
            fontstyle="italic",
        )

    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=LABEL_FONT_SIZE,
            color=COLOR_TEXT,
            fontfamily=FONT_FAMILY,
        )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        wrapped_labels,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        rotation=final_rotation,
        ha=ha,
    )

    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    if show_bar_labels:
        y_max = max(values) if values else 0
        offset = max(0.05 * y_max, 0.1)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=LABEL_FONT_SIZE,
                color=COLOR_TEXT,
                fontfamily=FONT_FAMILY,
            )

    if use_legend:
        add_discrete_legend(
            ax=ax,
            labels=labels,
            answer_colors=answer_colors if answer_colors is not None else {
                label: color for label, color in zip(labels, colors)
            },
            answer_order=answer_order,
            include_zero_answers=legend_include_zero_answers,
            counts=counts,
        )

    save_figure(fig, filename)
    
    
def plot_bar_xy(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    filename: str,
    x_order: list[str] | None = None,
    answer_colors: dict[str, str] | None = None,
    show_bar_labels: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    rotate_xticks: int = 0,
    bar_width: float = 0.6,
    wrap_label_width: int | None = 10,
    label_rotation: int = 0,
    vertical_labels: bool = False,
    skewed_labels: bool = False,
    y_min: float | None = None,
    y_max: float | None = None,
):
    """
    Plot a bar chart from one x-axis column and one y-axis column.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    x_column : str
        Column for x-axis labels.
    y_column : str
        Column for bar heights.
    title : str
        Plot title.
    filename : str
        Output filename.
    x_order : list[str] | None
        Optional full x label list in desired order.
    answer_colors : dict[str, str] | None
        Optional label -> color mapping. If provided, overrides automatic color logic.
    show_bar_labels : bool
        Whether to draw y-value labels above bars.
    xlabel : str | None
        Optional x-axis label.
    ylabel : str | None
        Optional y-axis label. Defaults to y_column if None.
    rotate_xticks : int
        Rotation angle for x tick labels.
    bar_width : float
        Width of bars.
    wrap_label_width : int | None
        Width for label wrapping.
    label_rotation : int
        Custom label rotation if not using vertical/skewed labels.
    vertical_labels : bool
        If True, rotate labels 90 degrees.
    skewed_labels : bool
        If True, rotate labels 45 degrees.
    y_min : float | None
        Optional minimum y-axis limit.
    y_max : float | None
        Optional maximum y-axis limit.
    """
    plot_df = df[[x_column, y_column]].copy()
    plot_df = plot_df.dropna(subset=[x_column, y_column])

    plot_df[x_column] = plot_df[x_column].astype(str).str.strip()
    plot_df = plot_df[plot_df[x_column] != ""]

    # Ensure y-values are numeric
    plot_df[y_column] = pd.to_numeric(plot_df[y_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[y_column])

    if x_order is not None:
        plot_df = (
            plot_df.set_index(x_column)
            .reindex(x_order)
            .reset_index()
            .rename(columns={"index": x_column})
        )
        labels = plot_df[x_column].tolist()
        values = plot_df[y_column].fillna(0).tolist()
    else:
        labels = plot_df[x_column].tolist()
        values = plot_df[y_column].tolist()

    # Color logic
    if answer_colors is not None:
        colors = [answer_colors.get(label, COLOR_NEUTRAL) for label in labels]
    else:
        keyword_map = {
            "action": COLOR_ACTION,
            "stealth": COLOR_STEALTH,
            "adaptive": COLOR_ADAPTIVE,
        }

        labels_lower = [label.strip().lower() for label in labels]
        has_any_special = any(label in keyword_map for label in labels_lower)

        default_color = COLOR_NEUTRAL if has_any_special else COLOR_STEALTH
        colors = [keyword_map.get(label.strip().lower(), default_color) for label in labels]

    # Rotation mode
    if vertical_labels:
        final_rotation = 90
        ha = "center"
    elif skewed_labels:
        final_rotation = 45
        ha = "right"
    else:
        final_rotation = label_rotation if label_rotation is not None else rotate_xticks
        ha = "right" if final_rotation else "center"

    # Wrapped labels
    wrapped_labels = [wrap_text(label, wrap_label_width) for label in labels]

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG)
    ax.set_facecolor(BG)  

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=colors,
        edgecolor=COLOR_LINE,
        linewidth=LINE_WIDTH,
    )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=30,
    )

    if xlabel is not None:
        ax.set_xlabel(
            xlabel,
            fontsize=LABEL_FONT_SIZE,
            color=COLOR_TEXT,
            fontfamily=FONT_FAMILY,
        )

    ax.set_ylabel(
        ylabel if ylabel is not None else y_column,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        wrapped_labels,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        rotation=final_rotation,
        ha=ha,
    )

    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # y-axis limits
    current_y_min, current_y_max = ax.get_ylim()
    final_y_min = y_min if y_min is not None else current_y_min
    final_y_max = y_max if y_max is not None else current_y_max

    # prevent invalid limits
    if final_y_max <= final_y_min:
        final_y_max = final_y_min + 1

    ax.set_ylim(final_y_min, final_y_max)

    if show_bar_labels:
        y_range = final_y_max - final_y_min
        offset = max(0.02 * y_range, 0.1)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=LABEL_FONT_SIZE,
                color=COLOR_TEXT,
                fontfamily=FONT_FAMILY,
            )

    save_figure(fig, filename)

# -----------------------------------------------------------------
#                           Point Plot
# -----------------------------------------------------------------
def plot_mode_points(
    df: pd.DataFrame,
    columns_by_mode: dict[str, str],
    title: str,
    filename: str,
    ylabel: str = "Score",
    xlabel: str = "Mode",
    mode_order: list[str] | None = None,
    show_mean_lines: bool = True,
    show_legend: bool = True,
    point_alpha: float = 0.4,
    point_size: float = MARKER_SIZE * 12,
    x_jitter: float = 0.16,
    y_min=None,
    y_max=None,
):
    """
    Plot participant-level points clustered by mode on the x-axis, with
    horizontal mean lines and an optional legend.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe, one row per participant.
    columns_by_mode : dict[str, str]
        Mapping like:
        {
            "Stealth": "stealth_mean",
            "Action": "action_mean",
            "Adaptive": "adaptive_mean"
        }
    title : str
        Plot title.
    filename : str
        Output filename.
    ylabel : str
        Y-axis label.
    xlabel : str
        X-axis label.
    mode_order : list[str] | None
        Optional display order of modes.
    show_mean_lines : bool
        Whether to draw horizontal mean lines for each mode.
    show_legend : bool
        Whether to show legend entries for points and mean lines.
    point_alpha : float
        Alpha for participant points.
    point_size : float
        Marker size.
    x_jitter : float
        Horizontal spread of participant points within each mode cluster.
    y_min, y_max : float or None
        Optional y-axis limits.
    """
    if mode_order is None:
        mode_order = ["Stealth", "Action", "Adaptive"]

    mode_order = [mode for mode in mode_order if mode in columns_by_mode]
    if not mode_order:
        raise ValueError("No valid modes found in columns_by_mode.")

    missing_cols = [columns_by_mode[m] for m in mode_order if columns_by_mode[m] not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    # Colors
    mode_colors = {
        "Stealth": COLOR_STEALTH,
        "Action": COLOR_ACTION,
        "Adaptive": COLOR_ADAPTIVE,
    }
    colors = {mode: mode_colors.get(mode, COLOR_NEUTRAL) for mode in mode_order}

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, facecolor=BG)
    ax.set_facecolor(BG)  
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    x_positions = np.arange(len(mode_order))
    legend_elements = []
    means = {}

    for i, mode in enumerate(mode_order):
        col = columns_by_mode[mode]
        values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

        if len(values) == 0:
            continue

        means[mode] = float(np.mean(values))

        # symmetric deterministic jitter
        if len(values) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-x_jitter, x_jitter, len(values))

        ax.scatter(
            np.full(len(values), x_positions[i]) + jitter,
            values,
            s=point_size,
            color=colors[mode],
            alpha=point_alpha,
            edgecolors="white",
            linewidths=0.8,
            zorder=3
        )

        # point legend entry
        if show_legend:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    label=mode,
                    markerfacecolor=colors[mode],
                    markeredgecolor="white",
                    markersize=LEGEND_MARKER_SIZE,
                    alpha=point_alpha,
                )
            )

    if show_mean_lines:
        for mode in mode_order:
            if mode not in means:
                continue

            ax.axhline(
                means[mode],
                color=colors[mode],
                linewidth=LINE_WIDTH_ERROR * 3,
                linestyle="--",
                alpha=0.9,
                zorder=1
            )

        if show_legend:
            legend_elements.extend([
                Line2D(
                    [0], [0],
                    color=colors[mode],
                    lw=LINE_WIDTH_ERROR * 3,
                    linestyle="--",
                    label=f"{mode} mean = {means[mode]:.2f}"
                )
                for mode in mode_order if mode in means
            ])

    ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=COLOR_TEXT)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)

    ax.grid(True, axis="y", color=COLOR_GRID, alpha=GRID_ALPHA)
    ax.tick_params(colors=COLOR_TEXT)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(mode_order)

    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max
        )
    

    if show_legend and legend_elements:
        ax.legend(
            handles=legend_elements,
            fontsize=LEGEND_FONT_SIZE,
            loc=LEGEND_LOCATION,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
            frameon=LEGEND_FRAME,
            labelcolor=COLOR_TEXT,
        )

    save_figure(fig, filename)
    
# -----------------------------------------------------------------
#                           Box Chart
# -----------------------------------------------------------------
def plot_mode_box_likert(
    df: pd.DataFrame,
    columns_by_mode: dict[str, str],
    title: str,
    filename: str,
    ylabel: str = "Score",
    show_means: bool = True,
    show_points: bool = True,
    point_alpha: float = 0.35,
    point_size: float = 28,
    box_width: float = 0.55,
    mode_order: list[str] | None = None,
    y_min = -3.2,
    y_max = 3.2,
):
    """
    Plot a box chart comparing Likert responses across Stealth, Action, and Adaptive modes.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    columns_by_mode : dict[str, str]
        Example:
        {
            "Stealth": "Stealth_Enjoyment",
            "Action": "Action_Enjoyment",
            "Adaptive": "Adaptive_Enjoyment"
        }
    title : str
        Plot title.
    filename : str
        Output filename.
    ylabel : str
        Y-axis label.
    show_means : bool
        Whether to draw mean markers.
    show_points : bool
        Whether to overlay individual participant points.
    point_alpha : float
        Transparency of participant points.
    point_size : float
        Size of participant points.
    box_width : float
        Width of each box.
    mode_order : list[str] | None
        Optional override of mode order.
    y_min, y_max : float
        Y-axis limits.
    """
    if mode_order is None:
        mode_order = ["Stealth", "Action", "Adaptive"]

    data = []
    labels = []
    colors = []

    for mode in mode_order:
        col = columns_by_mode[mode]
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        data.append(values)
        labels.append(mode)
        colors.append(MODE_COLORS.get(mode, COLOR_NEUTRAL))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG)
    ax.set_facecolor(BG)  

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    positions = np.arange(1, len(labels) + 1)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showmeans=False,
        showfliers=False,
        medianprops={
            "color": COLOR_LINE,
            "linewidth": 2.0,
        },
        whiskerprops={
            "color": COLOR_LINE,
            "linewidth": LINE_WIDTH,
        },
        capprops={
            "color": COLOR_LINE,
            "linewidth": LINE_WIDTH,
        },
        boxprops={
            "edgecolor": COLOR_LINE,
            "linewidth": LINE_WIDTH,
        },
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    if show_points:
        rng = np.random.default_rng(42)
        for pos, values, color in zip(positions, data, colors):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=point_size,
                alpha=point_alpha,
                color=color,
                edgecolors=COLOR_LINE,
                linewidths=0.5,
                zorder=3,
            )

    if show_means:
        means = [np.mean(values) if len(values) > 0 else np.nan for values in data]
        ax.scatter(
            positions,
            means,
            marker="o",
            s=70,
            color="white",
            edgecolors=COLOR_LINE,
            linewidths=1.2,
            zorder=4,
        )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=30,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(int(y_min), int(y_max)+1, 1))
    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    save_figure(fig, filename)

# -----------------------------------------------------------------
#                           100% rank stack
# -----------------------------------------------------------------
def plot_rank_100_stacked_bar(
    df: pd.DataFrame,
    columns_by_mode: dict[str, str],
    title: str,
    filename: str,
    rank_order: list[str],
    ylabel: str = "Percentage",
    show_segment_labels: bool = True,
    use_legend: bool = True,
):
    """
    Create a 100% stacked bar chart showing the distribution of ranking responses across modes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    columns_by_mode : dict[str, str]
        Mapping of mode names to ranking columns.
    rank_order : list[str]
        Ordered list of rank labels (e.g., lowest to highest).
    ylabel : str
        Y-axis label.
    show_segment_labels : bool
        Whether to display percentage labels inside segments.
    use_legend : bool
        Whether to display a legend.
    """
    mode_order = ["Stealth", "Action", "Adaptive"]

    count_table = pd.DataFrame(0, index=mode_order, columns=rank_order)

    for mode in mode_order:
        col = columns_by_mode[mode]
        counts = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .reindex(rank_order, fill_value=0)
        )
        count_table.loc[mode] = counts.values

    percent_table = count_table.div(count_table.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG)
    ax.set_facecolor(BG)  

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        
    rank_colors = {
        "Lowest enjoyment": COLOR_STEALTH,
        "Middle enjoyment": COLOR_ACTION,
        "Highest enjoyment": COLOR_ADAPTIVE,
    }

    x = np.arange(len(mode_order))
    bottom = np.zeros(len(mode_order))

    for rank in rank_order:
        values = percent_table[rank].values
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            color=rank_colors.get(rank, COLOR_NEUTRAL),
            edgecolor=COLOR_LINE,
            linewidth=LINE_WIDTH,
            label=rank,
            width=0.6,
        )

        if show_segment_labels:
            for bar, value, btm in zip(bars, values, bottom):
                if value > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        btm + value / 2,
                        f"{value:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=LABEL_FONT_SIZE,
                        color="white",
                        fontfamily=FONT_FAMILY,
                    )

        bottom += values

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=30,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        mode_order,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    if use_legend:
        add_discrete_legend(
            ax=ax,
            labels=rank_order,
            answer_colors=rank_colors,
            answer_order=rank_order,
            include_zero_answers=True,
            counts=count_table.sum(axis=0).to_dict(),
        )

    save_figure(fig, filename)

# -----------------------------------------------------------------
#                           Mean rank score
# -----------------------------------------------------------------
def plot_mean_rank_score_bar(
    df: pd.DataFrame,
    columns_by_mode: dict[str, str],
    title: str,
    filename: str,
    rank_score_map: dict[str, float],
    ylabel: str = "Mean Rank Score",
    show_bar_labels: bool = True,
):
    """
    Create a bar chart of mean rank scores across gameplay modes.
    
    Rank labels are converted to numeric scores using rank_score_map.
    """
    mode_order = ["Stealth", "Action", "Adaptive"]

    means = []
    colors = []

    for mode in mode_order:
        col = columns_by_mode[mode]
        scores = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .map(rank_score_map)
        )
        means.append(scores.mean())
        colors.append(MODE_COLORS.get(mode, COLOR_NEUTRAL))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG)
    ax.set_facecolor(BG)  

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    x = np.arange(len(mode_order))
    bars = ax.bar(
        x,
        means,
        color=colors,
        edgecolor=COLOR_LINE,
        linewidth=LINE_WIDTH,
        width=0.6,
    )
    
    ax.set_ylim(1.0, 2.4)

    if show_bar_labels:
        y_max = max(means) if len(means) > 0 else 0
        offset = max(0.03 * y_max, 0.03)

        for bar, value in zip(bars, means):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=LABEL_FONT_SIZE,
                    color=COLOR_TEXT,
                    fontfamily=FONT_FAMILY,
                )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=30,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        mode_order,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    save_figure(fig, filename)
    
# -----------------------------------------------------------------
#                        Mode Percentage Bar
# -----------------------------------------------------------------
def plot_mode_percentage_bar(
    df: pd.DataFrame,
    columns_by_mode: dict[str, str],
    title: str,
    filename: str,
    ylabel: str = "Percentage",
    error_type: str = "sd",
    show_bar_labels: bool = True,
    show_points: bool = False,
    point_alpha: float = 0.35,
    point_size: float = 28,
    mode_order: list[str] | None = None,
    as_percent_axis: bool = True,
    bar_width: float = 0.6,
    y_min: float | None = 0.0,
    y_max: float | None = 1.0,
):
    """
    Plot mean percentage-like values (0 to 1) across Stealth, Action, and Adaptive
    with error bars.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    columns_by_mode : dict[str, str]
        Example:
        {
            "Stealth": "Stealth_SeenRatio",
            "Action": "Action_SeenRatio",
            "Adaptive": "Adaptive_SeenRatio"
        }
    title : str
        Plot title.
    filename : str
        Output filename.
    ylabel : str
        Y-axis label.
    error_type : str
        "sd" for standard deviation
        "sem" for standard error of the mean
    show_bar_labels : bool
        Whether to draw value labels above bars.
    show_points : bool
        Whether to overlay participant-level points.
    point_alpha : float
        Transparency of points.
    point_size : float
        Size of points.
    mode_order : list[str] | None
        Plot order of modes.
    as_percent_axis : bool
        If True, axis labels and bar labels are shown as percentages.
    bar_width : float
        Width of bars.
    y_min, y_max : float | None
        Axis limits.
    """
    if mode_order is None:
        mode_order = ["Stealth", "Action", "Adaptive"]

    means = []
    errors = []
    colors = []
    all_values = []

    for mode in mode_order:
        col = columns_by_mode[mode]
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        all_values.append(values)

        mean_val = values.mean()
        if error_type.lower() == "sem":
            err_val = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        else:
            err_val = values.std(ddof=1) if len(values) > 1 else 0.0

        means.append(mean_val)
        errors.append(err_val)
        colors.append(MODE_COLORS.get(mode, COLOR_NEUTRAL))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG)
    ax.set_facecolor(BG)  

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    x = np.arange(len(mode_order))
    bars = ax.bar(
        x,
        means,
        yerr=errors,
        capsize=6,
        width=bar_width,
        color=colors,
        edgecolor=COLOR_LINE,
        linewidth=LINE_WIDTH,
        error_kw={
            "elinewidth": LINE_WIDTH_ERROR,
            "ecolor": COLOR_LINE,
            "capthick": LINE_WIDTH_ERROR,
        },
    )

    if show_points:
        rng = np.random.default_rng(42)
        for pos, values, color in zip(x, all_values, colors):
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), pos) + jitter,
                values,
                s=point_size,
                alpha=point_alpha,
                color=color,
                edgecolors=COLOR_LINE,
                linewidths=0.5,
                zorder=3,
            )

    if show_bar_labels:
        top_reference = max(
            [(m + e) for m, e in zip(means, errors) if pd.notna(m)],
            default=0
        )
        offset = max(0.03 * top_reference, 0.01)

        for bar, mean_val, err_val in zip(bars, means, errors):
            if pd.notna(mean_val):
                label_text = f"{mean_val * 100:.1f}%" if as_percent_axis else f"{mean_val:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_val + err_val + offset,
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=LABEL_FONT_SIZE,
                    color=COLOR_TEXT,
                    fontfamily=FONT_FAMILY,
                )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=60,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        mode_order,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    if y_min is not None or y_max is not None:
        current_bottom, current_top = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_bottom,
            y_max if y_max is not None else current_top,
        )

    if as_percent_axis:
        ticks = ax.get_yticks()
        ax.set_yticklabels([f"{tick * 100:.0f}%" for tick in ticks])

    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    save_figure(fig, filename)
    
# -----------------------------------------------------------------
#                       Multi-question box chart
# -----------------------------------------------------------------
def plot_likert_multi_question_box(
    df: pd.DataFrame,
    question_metrics: list[str],
    title: str,
    filename: str,
    question_labels: list[str] | None = None,
    inverted_metrics: list[str] | None = None,
    mode_order: list[str] | None = None,
    show_points: bool = True,
    point_alpha: float = 0.25,
    point_size: float = 18,
    box_width: float = 0.22,
    group_spacing: float = 1.5,
    label_rotation: float = 0,
    wrap_label_width: int | None = 10,
    ylabel: str = "Likert Score",
    ):
    """
    Plot grouped box plots for multiple Likert metrics across modes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Combined dataframe.
    question_metrics : list[str]
        Base metric names, e.g. ["Enjoyment", "Immersion", "Challenge", "Frustration", ...]
        Assumes columns are named:
            Stealth_<metric>, Action_<metric>, Adaptive_<metric>
    title : str
        Plot title.
    filename : str
        Output filename.
    question_labels : list[str] | None
        Labels for x-axis. If None, question_metrics are used.
    inverted_metrics : list[str] | None
        Metrics to invert by multiplying by -1, e.g. ["Frustration"]
    mode_order : list[str] | None
        Usually ["Stealth", "Action", "Adaptive"]
    show_points : bool
        Whether to overlay participant-level points.
    box_width : float
        Width of each box.
    group_spacing : float
        Spacing between question groups.
    label_rotation : float
        Rotation of x-axis labels.
    wrap_label_width : int | None
        Width for wrapping labels.
    ylabel : str
        Y-axis label.
    """
    if mode_order is None:
        mode_order = ["Stealth", "Action", "Adaptive"]
    
    if question_labels is None:
        question_labels = question_metrics
    
    if len(question_labels) != len(question_metrics):
        raise ValueError("question_labels must have the same length as question_metrics")
    
    if inverted_metrics is None:
        inverted_metrics = []
    
    fig_width = max(FIG_WIDTH, len(question_metrics) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, FIG_HEIGHT + 1), facecolor=BG)
    ax.set_facecolor(BG)  
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    
    n_modes = len(mode_order)
    group_centers = np.arange(len(question_metrics)) * group_spacing
    offsets = np.linspace(-(n_modes - 1) / 2, (n_modes - 1) / 2, n_modes) * box_width * 1.4
    
    for m_idx, mode in enumerate(mode_order):
        mode_color = MODE_COLORS.get(mode, COLOR_NEUTRAL)
        positions = group_centers + offsets[m_idx]
    
        mode_data = []
        for metric in question_metrics:
            col = f"{mode}_{metric}"
            values = pd.to_numeric(df[col], errors="coerce").dropna()
    
            if metric in inverted_metrics:
                values = values * -1
    
            mode_data.append(values)
    
        box = ax.boxplot(
            mode_data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            medianprops={
                "color": COLOR_LINE,
                "linewidth": 2.0,
            },
            whiskerprops={
                "color": COLOR_LINE,
                "linewidth": LINE_WIDTH_ERROR,
            },
            capprops={
                "color": COLOR_LINE,
                "linewidth": LINE_WIDTH_ERROR,
            },
            boxprops={
                "edgecolor": COLOR_LINE,
                "linewidth": LINE_WIDTH_ERROR/2,
            },
        )
    
        for patch in box["boxes"]:
            patch.set_facecolor(mode_color)
            patch.set_alpha(1.0)
    
        if show_points:
            rng = np.random.default_rng(42 + m_idx)
            for pos, values in zip(positions, mode_data):
                if len(values) == 0:
                    continue
                jitter = rng.uniform(-box_width * 0.18, box_width * 0.18, size=len(values))
                ax.scatter(
                    np.full(len(values), pos) + jitter,
                    values,
                    s=point_size,
                    alpha=point_alpha,
                    color=mode_color,
                    edgecolors=COLOR_LINE,
                    linewidths=0.4,
                    zorder=3,
                )
    
    # Build a consistent legend
    legend_labels = mode_order
    legend_colors = {mode: MODE_COLORS.get(mode, COLOR_NEUTRAL) for mode in mode_order}
    add_discrete_legend(
        ax=ax,
        labels=legend_labels,
        answer_colors=legend_colors,
        answer_order=legend_labels,
        include_zero_answers=True,
        counts={mode: 1 for mode in mode_order},
    )
    
    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=30,
    )
    
    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )
    
    wrapped_labels = [wrap_text(label, wrap_label_width) for label in question_labels]
    
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        wrapped_labels,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        rotation=label_rotation,
        ha="right",
    )
    
    ax.set_ylim(-3.2, 3.2)
    ax.set_yticks(np.arange(-3, 4, 1))
    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)
    
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    
    save_figure(fig, filename)
    
# -----------------------------------------------------------------
#                    Stacked bar chart for weights
# -----------------------------------------------------------------
def plot_mode_weight_stacked_bar(
    df: pd.DataFrame,
    title: str,
    filename: str,
    stealth_metric: str = "Stealth_W_avg",
    action_metric: str = "Action_W_avg",
    mode_order: list[str] | None = None,
    ylabel: str = "Average Weight",
    error_type: str | None = None,   # None, "sem", or "sd"
    show_bar_labels: bool = True,
    show_total_labels: bool = False,
    bar_width: float = 0.6,
):
    """
    Plot stacked bars of average Stealth and Action weights for each mode.
    
    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.
    stealth_metric : str
        Base name for stealth weight columns.
    action_metric : str
        Base name for action weight columns.
    mode_order : list[str] | None
        Order of modes on the x-axis.
    ylabel : str
        Y-axis label.
    error_type : str | None
        None, "sd", or "sem" for error bars.
    show_bar_labels : bool
        Whether to show values inside each segment.
    show_total_labels : bool
        Whether to show total stacked values above bars.
    bar_width : float
        Width of bars.
    """
    if mode_order is None:
        mode_order = ["Stealth", "Action", "Adaptive"]

    stealth_means = []
    action_means = []
    stealth_errors = []
    action_errors = []

    for mode in mode_order:
        stealth_col = f"{mode}_{stealth_metric}"
        action_col = f"{mode}_{action_metric}"

        stealth_vals = pd.to_numeric(df[stealth_col], errors="coerce").dropna()
        action_vals = pd.to_numeric(df[action_col], errors="coerce").dropna()

        stealth_mean = stealth_vals.mean()
        action_mean = action_vals.mean()

        stealth_means.append(stealth_mean)
        action_means.append(action_mean)

        if error_type is None:
            stealth_err = 0.0
            action_err = 0.0
        elif error_type.lower() == "sd":
            stealth_err = stealth_vals.std(ddof=1) if len(stealth_vals) > 1 else 0.0
            action_err = action_vals.std(ddof=1) if len(action_vals) > 1 else 0.0
        else:  # sem
            stealth_err = stealth_vals.std(ddof=1) / np.sqrt(len(stealth_vals)) if len(stealth_vals) > 1 else 0.0
            action_err = action_vals.std(ddof=1) / np.sqrt(len(action_vals)) if len(action_vals) > 1 else 0.0

        stealth_errors.append(stealth_err)
        action_errors.append(action_err)

    stealth_means = np.array(stealth_means)
    action_means = np.array(action_means)
    stealth_errors = np.array(stealth_errors)
    action_errors = np.array(action_errors)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG)
    ax.set_facecolor(BG)  

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    x = np.arange(len(mode_order))

    # Bottom segment: stealth weight
    ax.bar(
        x,
        stealth_means,
        width=bar_width,
        color=COLOR_STEALTH,
        edgecolor=COLOR_LINE,
        linewidth=LINE_WIDTH,
        label="Stealth Weight",
    )

    # Top segment: action weight
    ax.bar(
        x,
        action_means,
        width=bar_width,
        bottom=stealth_means,
        color=COLOR_ACTION,
        edgecolor=COLOR_LINE,
        linewidth=LINE_WIDTH,
        label="Action Weight",
    )

    # Optional error bars
    if error_type is not None:
        # error on stealth segment mean
        ax.errorbar(
            x,
            stealth_means,
            yerr=stealth_errors,
            fmt="none",
            ecolor=COLOR_LINE,
            elinewidth=LINE_WIDTH,
            capsize=5,
            capthick=LINE_WIDTH,
            zorder=4,
        )

        # error on total stack (placed at the top of action segment)
        total_means = stealth_means + action_means
        total_errors = np.sqrt(stealth_errors**2 + action_errors**2)

        ax.errorbar(
            x,
            total_means,
            yerr=total_errors,
            fmt="none",
            ecolor=COLOR_LINE,
            elinewidth=LINE_WIDTH,
            capsize=5,
            capthick=LINE_WIDTH,
            zorder=4,
        )

    if show_bar_labels:
        for i in range(len(mode_order)):
            # label inside stealth segment
            if pd.notna(stealth_means[i]) and stealth_means[i] > 0:
                ax.text(
                    x[i],
                    stealth_means[i] / 2,
                    f"{stealth_means[i]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=LABEL_FONT_SIZE,
                    color="white",
                    fontfamily=FONT_FAMILY,
                )

            # label inside action segment
            if pd.notna(action_means[i]) and action_means[i] > 0:
                ax.text(
                    x[i],
                    stealth_means[i] + action_means[i] / 2,
                    f"{action_means[i]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=LABEL_FONT_SIZE,
                    color="white",
                    fontfamily=FONT_FAMILY,
                )

    if show_total_labels:
        totals = stealth_means + action_means
        for i, total in enumerate(totals):
            ax.text(
                x[i],
                total + 0.02,
                f"{total:.2f}",
                ha="center",
                va="bottom",
                fontsize=LABEL_FONT_SIZE,
                color=COLOR_TEXT,
                fontfamily=FONT_FAMILY,
            )

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=30,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        mode_order,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    add_discrete_legend(
        ax=ax,
        labels=["Stealth Weight", "Action Weight"],
        answer_colors={
            "Stealth Weight": COLOR_STEALTH,
            "Action Weight": COLOR_ACTION,
        },
        answer_order=["Stealth Weight", "Action Weight"],
        include_zero_answers=True,
        counts={"Stealth Weight": 1, "Action Weight": 1},
    )

    save_figure(fig, filename)

# -----------------------------------------------------------------
#                    Aggregated weight progression
# -----------------------------------------------------------------
def plot_raw_weight_progression(
    df: pd.DataFrame,
    mode: str,
    weight_type: str = "Stealth",   # "Stealth" or "Action"
    title: str | None = None,
    filename: str | None = None,
    ylabel: str = "Raw Weight",
    show_percent_axis: bool = True,
    show_participant_lines: bool = True,
    participant_line_alpha: float = 0.22,
    participant_line_width: float = 1.0,
    mean_line_width: float = 3.0,
    show_mean_markers: bool = True,
    y_min: float = 0.0,
    y_max: float = 1.0,
):
    """
    Plot per-participant and mean progression lines for raw weights
    across the 5 levels of one mode.

    Expected columns:
        {mode}_{weight_type}_W_1
        ...
        {mode}_{weight_type}_W_5

    Example:
        mode="Stealth", weight_type="Stealth"
        -> Stealth_Stealth_AW_1 ... Stealth_Stealth_AW_5
    """
    
    mode_all = False
    if mode.lower() == "all":
        show_participant_lines = False
        mode_all = True
    else:    
        level_cols = [f"{mode}_{weight_type}_W_{i}" for i in range(1, 6)]
        levels = np.arange(1, 6)

        # Pull numeric data
        plot_df = df[level_cols].apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), facecolor=BG)
    ax.set_facecolor(BG)  

    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
    

    # Choose main color by weight type
    if mode.lower() == "stealth":
        main_color = COLOR_STEALTH
    elif mode.lower() == "action":
        main_color = COLOR_ACTION
    elif mode.lower() == "adaptive":
        main_color = COLOR_ADAPTIVE
    else:
        main_color = COLOR_NEUTRAL

    # Individual participant lines
    if show_participant_lines:
        for _, row in plot_df.iterrows():
            y = row.values.astype(float)
            if np.all(np.isnan(y)):
                continue

            ax.plot(
                levels,
                y,
                color=main_color,
                alpha=participant_line_alpha,
                linewidth=participant_line_width,
                zorder=1,
            )

   
    
    if mode_all:
        for mode_ in ["Stealth", "Action", "Adaptive"]:
            if mode_.lower() == "stealth":
                main_color = COLOR_STEALTH
            elif mode_.lower() == "action":
                main_color = COLOR_ACTION
            elif mode_.lower() == "adaptive":
                main_color = COLOR_ADAPTIVE
            else:
                main_color = COLOR_NEUTRAL
                
            level_cols = [f"{mode_}_{weight_type}_W_{i}" for i in range(1, 6)]
            levels = np.arange(1, 6)
            
            plot_df = df[level_cols].apply(pd.to_numeric, errors="coerce")
            
            mean_values = plot_df.mean(axis=0, skipna=True).values.astype(float)
                
            ax.plot(
                levels,
                mean_values,
                color=main_color,
                linewidth=mean_line_width,
                marker="o" if show_mean_markers else None,
                markersize=6 if show_mean_markers else 0,
                zorder=3,
            )
    else:   
        # Mean line
        mean_values = plot_df.mean(axis=0, skipna=True).values.astype(float)
        
        ax.plot(
            levels,
            mean_values,
            color=main_color,
            linewidth=mean_line_width,
            marker="o" if show_mean_markers else None,
            markersize=6 if show_mean_markers else 0,
            zorder=3,
        )

    # Title
    if title is None:
        title = f"{mode} Mode - Raw {weight_type} Weight Progression"

    ax.set_title(
        title,
        fontsize=TITLE_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
        pad=30,
    )

    ax.set_xlabel(
        "Level",
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_xticks(levels)
    ax.set_xticklabels(
        [str(i) for i in levels],
        fontsize=LABEL_FONT_SIZE,
        color=COLOR_TEXT,
        fontfamily=FONT_FAMILY,
    )

    ax.set_ylim(y_min, y_max)

    ax.tick_params(axis="y", labelsize=LABEL_FONT_SIZE, colors=COLOR_TEXT)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily(FONT_FAMILY)

    if show_percent_axis:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_LINE)
    ax.spines["bottom"].set_color(COLOR_LINE)

    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    
    legend_elements = []

    # Participant line (only if shown)
    if show_participant_lines:
        legend_elements.append(
            Line2D(
                [0], [0],
                color=main_color,
                lw=participant_line_width,
                alpha=participant_line_alpha,
                label="Participants"
            )
        )
    
    # Mean lines
    if mode_all:
        for mode_, color in [
            ("Stealth", COLOR_STEALTH),
            ("Action", COLOR_ACTION),
            ("Adaptive", COLOR_ADAPTIVE),
        ]:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    color=color,
                    lw=mean_line_width,
                    marker="o" if show_mean_markers else None,
                    label=f"{mode_} Mean"
                )
            )
    else:
        legend_elements.append(
            Line2D(
                [0], [0],
                color=main_color,
                lw=mean_line_width,
                marker="o" if show_mean_markers else None,
                label="Mean"
            )
        )
    
    ax.legend(
        handles=legend_elements,
        fontsize=LABEL_FONT_SIZE,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=LEGEND_FRAME,
        labelcolor=COLOR_TEXT
    )

    if filename is None:
        filename = f"{mode}_{weight_type}_W_Progression.png"

    save_figure(fig, filename)

# -----------------------------------------------------------------
#                    Plot Mixed Model
# -----------------------------------------------------------------
def plot_mixed_model_predictions(
    model_dic,
    title: str,
    filename="mixed_model_plot.png",
    y_label: str = "Predicted Outcome",
    preference_range=(0, 1),
    n_points=100,
    show_raw_points=False,
    jitter_x=0.0,
    point_alpha=0.40,
    point_size=25,
    highlight_mode: str | None = None
):
    """
    Plot predicted outcomes from a fitted mixed-effects model as a function of player preference.
    
    This function generates smooth prediction lines for each gameplay mode based on the
    fitted mixed model, and can optionally overlay raw participant data points.
    
    Parameters
    ----------
    model_dic : dict
        Dictionary containing model outputs with keys:
            - "fit": fitted statsmodels mixed-effects result
            - "mode_order": list or tuple of mode names (e.g., ["Action", "Stealth", "Adaptive"])
            - "long_df": (optional) dataframe with columns ["preference", "outcome", "mode"]
    title : str
        Title of the plot.
    filename : str, default="mixed_model_plot.png"
        Output filename for the saved figure.
    y_label : str, default="Predicted Outcome"
        Label for the y-axis.
    preference_range : tuple, default=(0, 1)
        Range of preference values used to generate predictions.
    n_points : int, default=100
        Number of points used to create smooth prediction lines.
    show_raw_points : bool, default=False
        Whether to overlay raw participant data points.
    jitter_x : float, default=0.0
        Amount of horizontal jitter applied to raw points to reduce overlap.
    point_alpha : float, default=0.40
        Transparency of raw data points.
    point_size : float, default=25
        Size of raw data points.
    highlight_mode : str | None, default=None
        If provided, only plot predictions and raw points for the specified mode.
    
    Returns
    -------
    None
        The function saves the plot to file and does not return a value.
    """
    
    fit_result = model_dic["fit"]
    mode_order = model_dic["mode_order"]
    long_df = model_dic.get("long_df", None)

    # Generate prediction grid
    pref_values = np.linspace(preference_range[0], preference_range[1], n_points)

    rows = []
    for mode in mode_order:
        for p in pref_values:
            rows.append({
                "preference": p,
                "mode": mode
            })

    pred_df = pd.DataFrame(rows)

    # Ensure categorical ordering
    pred_df["mode"] = pd.Categorical(pred_df["mode"], categories=mode_order)

    # Get predictions
    pred_df["predicted"] = fit_result.predict(pred_df)

    # Create plot
    fig, ax = plt.subplots(figsize=(FIG_WIDTH-1, FIG_HEIGHT+1), dpi=DPI, facecolor=BG)
    ax.set_facecolor(BG)  
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    # Raw data overlay
    if show_raw_points and long_df is not None:
        raw_df = long_df.copy()
        raw_df = raw_df.dropna(subset=["preference", "outcome", "mode"])
        raw_df["mode"] = pd.Categorical(raw_df["mode"], categories=mode_order)

        rng = np.random.default_rng(42)

        for mode in mode_order:
            if highlight_mode is not None and mode != highlight_mode:
                continue
            subset = raw_df[raw_df["mode"] == mode].copy()
            if subset.empty:
                continue

            x_jitter = rng.uniform(-jitter_x, jitter_x, size=len(subset))
            x_vals = subset["preference"].to_numpy() + x_jitter

            ax.scatter(
                x_vals,
                subset["outcome"],
                s=point_size,
                alpha=point_alpha,
                color=MODE_COLORS.get(mode, COLOR_LINE),
                edgecolors="none",
                zorder=2
            )

    # Model lines
    for mode in mode_order:
        if highlight_mode is not None and mode != highlight_mode:
            continue
        subset = pred_df[pred_df["mode"] == mode]

        ax.plot(
            subset["preference"],
            subset["predicted"],
            label=mode,
            color=MODE_COLORS.get(mode, COLOR_LINE),
            linewidth=LINE_WIDTH * 2,
            alpha=BAR_ALPHA
        )

    # Styling
    ax.set_xlabel("Preference (0.0 = Action, 1.0 = Stealth)", fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)

    ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=COLOR_TEXT)

    # Grid
    ax.grid(True, color=COLOR_GRID, alpha=GRID_ALPHA)

    # Axis colors
    ax.tick_params(colors=COLOR_TEXT)
    
    # Keep x-axis in sensible bounds
    ax.set_xlim(preference_range[0] - 0.05, preference_range[1] + 0.05)

    # Legend
    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=LEGEND_FRAME,
        labelcolor=COLOR_TEXT
    )

    save_figure(fig, filename)
    
# -----------------------------------------------------------------
#                    difference plot
# -----------------------------------------------------------------
def plot_mode_difference_by_participant(
    df: pd.DataFrame,
    reference_mode_col: str,
    columns_by_mode: dict,
    target_mode: str = "Adaptive",
    subtract: str = "target_minus_reference",
    title: str = "Adaptive vs Reference by Participant",
    y_label: str = "Score Difference",
    filename: str = "mode_difference_by_participant.png",
    sort_values: bool = True,
    show_zero_line: bool = True,
    show_mean_line: bool = True,
    show_median_line: bool = False,
    y_min=None,
    y_max=None, 
    legend_prefix:str = "Prefers",
    legend_mean:str = "",
    point_alpha: float = 0.5,
):
    """
    Plot per-participant differences between a target mode and a participant-specific reference mode.
    
    Each point represents a participant, showing how their score in the target mode
    differs from their reference mode. Points are colored based on the participant’s
    reference mode. Optional lines indicate zero difference, mean difference, and median difference.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe with one row per participant.
    reference_mode_col : str
        Column indicating the reference mode for each participant (e.g., preferred or best mode).
    columns_by_mode : dict
        Mapping from mode name to column name containing that mode’s scores.
    target_mode : str, default="Adaptive"
        Mode to compare against each participant’s reference mode.
    subtract : str, default="target_minus_reference"
        Direction of subtraction:
            - "target_minus_reference": target − reference
            - "reference_minus_target": reference − target
    title : str
        Plot title.
    y_label : str
        Y-axis label.
    filename : str
        Output filename.
    sort_values : bool, default=True
        Whether to sort participants by difference value.
    show_zero_line : bool, default=True
        Whether to draw a horizontal line at y = 0.
    show_mean_line : bool, default=True
        Whether to draw a horizontal line at the mean difference.
    show_median_line : bool, default=False
        Whether to draw a horizontal line at the median difference.
    y_min, y_max : float | None
        Optional y-axis limits.
    legend_prefix : str, default="Prefers"
        Prefix used for labeling participant groups in the legend.
    legend_mean : str, default=""
        Additional text appended to mean/median legend labels.
    point_alpha : float, default=0.5
        Transparency of participant points.
    
    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with added columns:
            - "_reference_score"
            - "_target_score"
            - "_difference"
            - "_participant_plot_id"
    """

    if reference_mode_col not in df.columns:
        raise ValueError(f"'{reference_mode_col}' not found in dataframe")

    if target_mode not in columns_by_mode:
        raise ValueError(f"target_mode '{target_mode}' not found in columns_by_mode")

    missing_cols = [col for col in columns_by_mode.values() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    if subtract not in {"target_minus_reference", "reference_minus_target"}:
        raise ValueError(
            "subtract must be either 'target_minus_reference' or 'reference_minus_target'"
        )

    plot_df = df.copy()

    def get_reference_score(row):
        mode_name = row[reference_mode_col]
        if mode_name not in columns_by_mode:
            return np.nan
        return row[columns_by_mode[mode_name]]

    plot_df["_reference_score"] = plot_df.apply(get_reference_score, axis=1)
    plot_df["_target_score"] = pd.to_numeric(
        plot_df[columns_by_mode[target_mode]], errors="coerce"
    )
    plot_df["_reference_score"] = pd.to_numeric(plot_df["_reference_score"], errors="coerce")

    if subtract == "target_minus_reference":
        plot_df["_difference"] = plot_df["_target_score"] - plot_df["_reference_score"]
    else:
        plot_df["_difference"] = plot_df["_reference_score"] - plot_df["_target_score"]

    plot_df = plot_df.dropna(subset=["_difference"]).copy()

    if sort_values:
        plot_df = plot_df.sort_values("_difference").reset_index(drop=True)
    else:
        plot_df = plot_df.reset_index(drop=True)

    plot_df["_participant_plot_id"] = np.arange(1, len(plot_df) + 1)

    mean_val = plot_df["_difference"].mean()
    median_val = plot_df["_difference"].median()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, facecolor=BG)
    ax.set_facecolor(BG)  
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    point_colors = plot_df[reference_mode_col].map(MODE_COLORS)

    ax.scatter(
        plot_df["_participant_plot_id"],
        plot_df["_difference"],
        s=MARKER_SIZE * 12,
        c=point_colors,
        alpha=point_alpha,
        edgecolors="white",
        linewidths=0.8,
        zorder=3
    )

    if show_zero_line:
        ax.axhline(
            0,
            color=COLOR_LINE,
            linewidth=LINE_WIDTH_ERROR,
            linestyle="--",
            alpha=0.9,
            zorder=1
        )

    if show_mean_line:
        ax.axhline(
            mean_val,
            color=COLOR_ADAPTIVE,
            linewidth=LINE_WIDTH_ERROR * 3,
            linestyle="-",
            alpha=0.9,
            zorder=2,
            label=f"Mean {legend_mean} = {mean_val:.2f}"
        )

    if show_median_line:
        ax.axhline(
            median_val,
            color=COLOR_STEALTH,
            linewidth=LINE_WIDTH_ERROR * 3,
            linestyle=":",
            alpha=0.9,
            zorder=2,
            label=f"Median {legend_mean} = {median_val:.2f}"
        )

    ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=COLOR_TEXT,pad=30)
    ax.set_xlabel("Participant", fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)

    ax.grid(True, axis="y", color=COLOR_GRID, alpha=GRID_ALPHA)
    ax.tick_params(colors=COLOR_TEXT)

    ax.set_xticks(plot_df["_participant_plot_id"])
    
    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
    
        ax.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max
        )
    
    legend_elements = []
    
    for mode in plot_df[reference_mode_col].dropna().unique():
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=f"{legend_prefix} {mode}",
                markerfacecolor=MODE_COLORS.get(mode, COLOR_LINE),
                markersize=LEGEND_MARKER_SIZE,
                alpha=point_alpha
            )
        )
    
    if show_mean_line or show_median_line:
        legend_elements.append(
            Line2D(
                [0], [0],
                color=COLOR_ADAPTIVE,
                lw=LINE_WIDTH_ERROR * 3,
                label=f"Mean {legend_mean} = {mean_val:.2f}"
            )
        )
    ax.legend(
        handles=legend_elements,
        #title="Reference Mode",
        fontsize=LEGEND_FONT_SIZE,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=LEGEND_FRAME,
        labelcolor=COLOR_TEXT
    )

    save_figure(fig, filename)

    return plot_df

# -----------------------------------------------------------------
#                    Single Column Dots plot
# -----------------------------------------------------------------
def plot_single_column_by_participant(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str,
    y_label: str = "Value",
    x_label: str = "Participant",
    sort_values: bool = True,
    ascending: bool = True,
    color_col: str | None = None,
    color_map: dict | None = None,
    default_color: str = COLOR_ADAPTIVE,
    show_mean_line: bool = True,
    show_zero_line: bool = False,
    show_median_line: bool = False,
    zero_line_value: float = 0.0,
    y_min=None,
    y_max=None,
    point_alpha: float = 0.9,
    point_size: float = MARKER_SIZE * 12,
    edgecolor: str = "white",
    edgewidth: float = 0.8,
    mean_line_color: str = COLOR_ADAPTIVE,
    median_line_color: str = COLOR_STEALTH,
    zero_line_color: str = COLOR_LINE,
    mean_line_label: str | None = None,
    median_line_label: str | None = None,
    zero_line_label: str = "Zero reference",
    use_threshold:bool = False,
    dsidedthreshold:bool = False,
    threshold:float = 0.5,
    draw_zone:bool = False,
    draw_zone_alpha:float = 0.1,
    threshold_line:bool = False,
    legend_prefix:str = "Prefers",
    legend_suffix:str = "",
):
    """
    Plot one numeric column as participant-level dots.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe, one row per participant.
    value_col : str
        Numeric column to plot on the y-axis.
    title : str
        Plot title.
    filename : str
        Output filename passed to save_figure.
    y_label : str
        Label for y-axis.
    x_label : str
        Label for x-axis.
    sort_values : bool
        Whether to sort participants by the plotted value.
    ascending : bool
        Sort direction if sort_values=True.
    color_col : str or None
        Optional column used to color points by category.
    color_map : dict or None
        Optional mapping from color_col values to colors.
    default_color : str
        Color used when color_col is None or category missing from color_map.
    show_mean_line : bool
        Whether to draw a horizontal mean line.
    show_zero_line : bool
        Whether to draw a horizontal zero line.
    show_median_line : bool
        Whether to draw a horizontal median line.
    y_min, y_max : float or None
        Optional y-axis limits.
    point_alpha : float
        Point transparency.
    point_size : float
        Point size.
    edgecolor : str
        Point edge color.
    edgewidth : float
        Point edge width.
    mean_line_color : str
        Color of mean line.
    median_line_color : str
        Color of median line.
    zero_line_color : str
        Color of zero reference line.
    mean_line_label : str or None
        Custom mean line label. If None, uses mean value.
    median_line_label : str or None
        Custom median line label. If None, uses median value.
    zero_line_label : str
        Label for zero line.
    use_threshold : bool
        Whether to color points based on threshold logic.
    dsidedthreshold : bool
        If True, use two-sided thresholds (low, middle, high zones).
    threshold : float
        Threshold value for classification.
    draw_zone : bool
        Whether to draw shaded threshold regions.
    draw_zone_alpha : float
        Transparency of shaded zones.
    threshold_line : bool
        Whether to draw threshold boundary lines.
    legend_prefix : str
        Prefix for legend labels.
    legend_suffix : str
        Suffix for legend labels.

    Returns
    -------
    pd.DataFrame
        Plot dataframe with:
        - _plot_value
        - _participant_plot_id
    """

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in dataframe")

    if color_col is not None and color_col not in df.columns:
        raise ValueError(f"'{color_col}' not found in dataframe")

    plot_df = df.copy()
    plot_df["_plot_value"] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["_plot_value"]).copy()

    if plot_df.empty:
        raise ValueError(f"No valid numeric data found in column '{value_col}'")

    if sort_values:
        plot_df = plot_df.sort_values("_plot_value", ascending=ascending).reset_index(drop=True)
    else:
        plot_df = plot_df.reset_index(drop=True)

    plot_df["_participant_plot_id"] = np.arange(1, len(plot_df) + 1)

    mean_val = plot_df["_plot_value"].mean()
    median_val = plot_df["_plot_value"].median()

    if color_col is not None:
        if color_map is None:
            unique_vals = plot_df[color_col].dropna().unique()
            color_map = {val: default_color for val in unique_vals}
        point_colors = plot_df[color_col].map(color_map).fillna(default_color)
    else:
        point_colors = pd.Series(default_color, index=plot_df.index)
        
    if use_threshold:
        if dsidedthreshold:
            point_colors[plot_df["_plot_value"] > (1 -threshold)] = COLOR_STEALTH
            point_colors[plot_df["_plot_value"] <= threshold] = COLOR_ACTION
        else:
            point_colors[plot_df["_plot_value"] >= threshold] = COLOR_STEALTH
            point_colors[plot_df["_plot_value"] < threshold] = COLOR_ACTION

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, facecolor=BG)
    ax.set_facecolor(BG)  
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    ax.scatter(
        plot_df["_participant_plot_id"],
        plot_df["_plot_value"],
        s=point_size,
        c=point_colors,
        alpha=point_alpha,
        edgecolors=edgecolor,
        linewidths=edgewidth,
        zorder=3
    )

    legend_elements = []

    if show_zero_line:
        ax.axhline(
            zero_line_value,
            color=zero_line_color,
            linewidth=LINE_WIDTH_ERROR,
            linestyle="--",
            alpha=0.9,
            zorder=1
        )
        legend_elements.append(
            Line2D(
                [0], [0],
                color=zero_line_color,
                lw=LINE_WIDTH_ERROR,
                linestyle="--",
                label=zero_line_label
            )
        )

    if show_mean_line:
        ax.axhline(
            mean_val,
            color=mean_line_color,
            linewidth=LINE_WIDTH_ERROR * 2,
            linestyle="-",
            alpha=0.9,
            zorder=2
        )
        legend_elements.append(
            Line2D(
                [0], [0],
                color=mean_line_color,
                lw=LINE_WIDTH_ERROR * 2,
                linestyle="-",
                label=mean_line_label if mean_line_label is not None else f"Mean = {mean_val:.2f}"
            )
        )

    if show_median_line:
        ax.axhline(
            median_val,
            color=median_line_color,
            linewidth=LINE_WIDTH_ERROR * 2,
            linestyle=":",
            alpha=0.9,
            zorder=2
        )
        legend_elements.append(
            Line2D(
                [0], [0],
                color=median_line_color,
                lw=LINE_WIDTH_ERROR * 2,
                linestyle=":",
                label=median_line_label if median_line_label is not None else f"Median = {median_val:.2f}"
            )
        )

    if color_col is not None:
        unique_vals = [v for v in plot_df[color_col].dropna().unique()]
        for val in unique_vals:
            legend_elements.insert(
                0,
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    label=str(val),
                    markerfacecolor=color_map.get(val, default_color),
                    markeredgecolor=edgecolor,
                    markersize=LEGEND_MARKER_SIZE,
                )
            )
            
    if use_threshold and draw_zone:
        if dsidedthreshold:
            ax.axhspan(y_min,threshold, color=COLOR_ACTION, alpha = draw_zone_alpha)
            ax.axhspan(threshold,(1 - threshold), color=COLOR_ADAPTIVE, alpha = draw_zone_alpha)
            ax.axhspan((1 - threshold),y_max, color=COLOR_STEALTH, alpha = draw_zone_alpha)
        else:
            ax.axhspan(y_min,threshold, color=COLOR_ACTION, alpha = draw_zone_alpha)
            ax.axhspan(threshold,y_max, color=COLOR_STEALTH, alpha = draw_zone_alpha)
    
    if use_threshold:
        if dsidedthreshold:
            modes = ["Stealth", "Action", "Adaptive"]
            if threshold_line:
                for i in range(2):
                    ax.axhline(
                        abs(i - threshold),
                        color=MODE_COLORS.get(modes[1-i], COLOR_LINE),
                        linewidth=LINE_WIDTH_ERROR * 2,
                        linestyle="-",
                        alpha=0.9,
                        zorder=2
                    )
                    legend_elements.append(
                        Line2D(
                            [0], [0],
                            color=MODE_COLORS.get(modes[1-i], COLOR_LINE),
                            lw=LINE_WIDTH_ERROR * 2,
                            linestyle="-",
                            label= f"{modes[1-i]} Threshold = {threshold:.2f}"
                        )
                    )
        else:
            modes = ["Stealth", "Action"]
            if threshold_line:
                ax.axhline(
                    threshold,
                    color=COLOR_ADAPTIVE,
                    linewidth=LINE_WIDTH_ERROR * 2,
                    linestyle="-",
                    alpha=0.9,
                    zorder=2
                )
                legend_elements.append(
                    Line2D(
                        [0], [0],
                        color=COLOR_ADAPTIVE,
                        lw=LINE_WIDTH_ERROR * 2,
                        linestyle="-",
                        label= f"Threshold = {threshold:.2f}"
                    )
                )
        for mode in modes:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    label=f"{legend_prefix} {mode} {legend_suffix}",
                    markerfacecolor=MODE_COLORS.get(mode, COLOR_LINE),
                    markersize=LEGEND_MARKER_SIZE,
                    alpha=point_alpha
                )
            )
    
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=COLOR_TEXT,pad=30)
    ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)

    ax.grid(True, axis="y", color=COLOR_GRID, alpha=GRID_ALPHA)
    ax.tick_params(colors=COLOR_TEXT)
    ax.set_xticks(plot_df["_participant_plot_id"])

    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max
        )

    if legend_elements:
        ax.legend(
            handles=legend_elements,
            fontsize=LEGEND_FONT_SIZE,
            loc=LEGEND_LOCATION,
            bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
            frameon=LEGEND_FRAME,
            labelcolor=COLOR_TEXT
        )

    save_figure(fig, filename)
    return plot_df

# -----------------------------------------------------------------
#                    Binned boxplots
# -----------------------------------------------------------------
def plot_preference_binned_mode_boxplots(
    df: pd.DataFrame,
    preference_col: str,
    columns_by_mode: dict,
    title: str,
    filename: str,
    y_label: str = "Outcome",
    threshold: float = 0.5,
    lower_label: str = "Action-leaning",
    upper_label: str = "Stealth-leaning",
    mode_order=None,
    show_points: bool = True,
    point_alpha: float = 0.6,
    point_size: float = 40,
    jitter_x: float = 0.06,
    box_alpha: float = 0.45,
    y_min=None,
    y_max=None,
):
    """
    Create 2-bin preference boxplots by mode.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe, one row per participant.
    preference_col : str
        Column with preference score, where 0=Action, 1=Stealth.
    columns_by_mode : dict
        Mapping like:
        {
            "Stealth": "stealth_mean",
            "Action": "action_mean",
            "Adaptive": "adaptive_mean"
        }
    title : str
        Plot title.
    filename : str
        Output filename.
    y_label : str
        Y-axis label.
    threshold : float
        Split point for the 2 bins.
    lower_label : str
        Label for preference <= threshold.
    upper_label : str
        Label for preference > threshold.
    mode_order : list/tuple or None
        Order of modes. Defaults to keys of columns_by_mode.
    show_points : bool
        Whether to overlay participant points.
    point_alpha : float
        Point transparency.
    point_size : float
        Point size.
    jitter_x : float
        Horizontal jitter amount for points.
    box_alpha : float
        Transparency of boxplots.
    y_min, y_max : float or None
        Optional y-axis limits.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe used for plotting.
    """

    if preference_col not in df.columns:
        raise ValueError(f"'{preference_col}' not found in dataframe")

    missing_cols = [col for col in columns_by_mode.values() if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    if mode_order is None:
        mode_order = list(columns_by_mode.keys())

    plot_df = df.copy()
    plot_df[preference_col] = pd.to_numeric(plot_df[preference_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[preference_col]).copy()

    if plot_df.empty:
        raise ValueError("No valid rows left after cleaning preference column")

    # Create 2-bin grouping
    plot_df["_preference_bin"] = np.where(
        plot_df[preference_col] <= threshold,
        lower_label,
        upper_label
    )

    # Build long dataframe
    long_parts = []
    for mode in mode_order:
        if mode not in columns_by_mode:
            raise ValueError(f"Mode '{mode}' not found in columns_by_mode")

        temp = plot_df[[preference_col, "_preference_bin", columns_by_mode[mode]]].copy()
        temp = temp.rename(columns={columns_by_mode[mode]: "outcome"})
        temp["mode"] = mode
        long_parts.append(temp)

    long_df = pd.concat(long_parts, ignore_index=True)
    long_df["outcome"] = pd.to_numeric(long_df["outcome"], errors="coerce")
    long_df = long_df.dropna(subset=["outcome"]).copy()

    if long_df.empty:
        raise ValueError("No valid outcome data available for plotting")

    long_df["mode"] = pd.Categorical(long_df["mode"], categories=mode_order, ordered=True)
    bin_order = [lower_label, upper_label]
    long_df["_preference_bin"] = pd.Categorical(
        long_df["_preference_bin"],
        categories=bin_order,
        ordered=True
    )

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, facecolor=BG)
    ax.set_facecolor(BG) 
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    # Positioning
    group_centers = [1, 2]
    offsets = np.linspace(-0.25, 0.25, num=len(mode_order))
    width = 0.18

    rng = np.random.default_rng(42)

    # Draw boxplots by bin x mode
    for gi, group_label in enumerate(bin_order):
        group_center = group_centers[gi]

        for mi, mode in enumerate(mode_order):
            xpos = group_center + offsets[mi]
            vals = long_df[
                (long_df["_preference_bin"] == group_label) &
                (long_df["mode"] == mode)
            ]["outcome"].to_numpy()

            if len(vals) == 0:
                continue

            ax.boxplot(
                [vals],
                positions=[xpos],
                widths=width,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(
                    facecolor=MODE_COLORS.get(mode, COLOR_NEUTRAL),
                    alpha=box_alpha,
                    color=COLOR_LINE,
                    linewidth=LINE_WIDTH_ERROR / 2
                ),
                medianprops=dict(color=COLOR_LINE, linewidth=LINE_WIDTH_ERROR * 2),
                whiskerprops=dict(color=COLOR_LINE, linewidth=LINE_WIDTH_ERROR),
                capprops=dict(color=COLOR_LINE, linewidth=LINE_WIDTH_ERROR),
            )

            if show_points:
                x_jittered = rng.uniform(xpos - jitter_x, xpos + jitter_x, size=len(vals))
                ax.scatter(
                    x_jittered,
                    vals,
                    s=point_size,
                    color=MODE_COLORS.get(mode, COLOR_NEUTRAL),
                    alpha=point_alpha,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=3
                )

    # Axis labels/ticks
    ax.set_xticks(group_centers)
    ax.set_xticklabels(bin_order, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_xlabel("Player Preference Group", fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=COLOR_TEXT,pad=30)

    # Grid / ticks
    ax.grid(True, axis="y", color=COLOR_GRID, alpha=GRID_ALPHA)
    ax.tick_params(colors=COLOR_TEXT)

    # Optional y limits
    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max
        )

    # Legend for modes
    legend_elements = [
        Patch(
            facecolor=MODE_COLORS.get(mode, COLOR_NEUTRAL),
            edgecolor=COLOR_LINE,
            alpha=box_alpha,
            label=mode
        )
        for mode in mode_order
    ]

    ax.legend(
        handles=legend_elements,
        fontsize=LEGEND_FONT_SIZE,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=LEGEND_FRAME,
        labelcolor=COLOR_TEXT
    )

    save_figure(fig, filename)
    return long_df

# -----------------------------------------------------------------
#                    Dumbbell
# -----------------------------------------------------------------   
def plot_mode_dumbbell(
    df: pd.DataFrame,
    columns_by_mode: dict,
    left_mode: str,
    right_mode: str,
    title: str,
    filename: str,
    y_label: str = "Score",
    x_label: str = "Participant",
    sort_by: str = "difference",
    ascending: bool = True,
    color_col: str | None = None,
    color_map: dict | None = None,
    left_color: str | None = None,
    right_color: str | None = None,
    line_color: str = COLOR_NEUTRAL,
    line_alpha: float = 0.35,
    point_alpha: float = 0.95,
    point_size: float = MARKER_SIZE * 12,
    show_mean_lines: bool = True,
    show_zero_line: bool = False,
    y_min=None,
    y_max=None,
):
    """
    Plot a dumbbell chart comparing two mode columns per participant.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe, one row per participant.
    columns_by_mode : dict
        Mapping like:
        {
            "Stealth": "stealth_mean",
            "Action": "action_mean",
            "Adaptive": "adaptive_mean"
        }
    left_mode : str
        Mode shown as the left endpoint conceptually.
    right_mode : str
        Mode shown as the right endpoint conceptually.
    title : str
        Plot title.
    filename : str
        Output filename.
    y_label : str
        Label for y-axis.
    x_label : str
        Label for x-axis.
    sort_by : str
        One of:
        - "difference"  -> sort by (right - left)
        - "left"        -> sort by left mode value
        - "right"       -> sort by right mode value
        - "none"        -> keep original row order
    ascending : bool
        Sort direction.
    color_col : str or None
        Optional column used to color connecting lines by category.
    color_map : dict or None
        Mapping from values in color_col to colors.
    left_color, right_color : str or None
        Colors for endpoint markers. Defaults to MODE_COLORS if available.
    line_color : str
        Default line color when color_col is None.
    line_alpha : float
        Alpha for connecting lines.
    point_alpha : float
        Alpha for endpoint markers.
    point_size : float
        Endpoint marker size.
    show_mean_lines : bool
        Whether to draw horizontal mean lines for the two modes.
    show_zero_line : bool
        Whether to draw a horizontal zero line.
    y_min, y_max : float or None
        Optional y-axis limits.

    Returns
    -------
    pd.DataFrame
        Dataframe used for plotting with columns:
        - _left_value
        - _right_value
        - _difference
        - _participant_plot_id
    """

    if left_mode not in columns_by_mode:
        raise ValueError(f"left_mode '{left_mode}' not found in columns_by_mode")
    if right_mode not in columns_by_mode:
        raise ValueError(f"right_mode '{right_mode}' not found in columns_by_mode")

    left_col = columns_by_mode[left_mode]
    right_col = columns_by_mode[right_mode]

    missing_cols = [c for c in [left_col, right_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    if color_col is not None and color_col not in df.columns:
        raise ValueError(f"'{color_col}' not found in dataframe")

    plot_df = df.copy()
    plot_df["_left_value"] = pd.to_numeric(plot_df[left_col], errors="coerce")
    plot_df["_right_value"] = pd.to_numeric(plot_df[right_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["_left_value", "_right_value"]).copy()

    if plot_df.empty:
        raise ValueError("No valid rows left after cleaning mode columns")

    plot_df["_difference"] = plot_df["_right_value"] - plot_df["_left_value"]

    if sort_by == "difference":
        plot_df = plot_df.sort_values("_difference", ascending=ascending)
    elif sort_by == "left":
        plot_df = plot_df.sort_values("_left_value", ascending=ascending)
    elif sort_by == "right":
        plot_df = plot_df.sort_values("_right_value", ascending=ascending)
    elif sort_by == "none":
        plot_df = plot_df.copy()
    else:
        raise ValueError("sort_by must be one of: 'difference', 'left', 'right', 'none'")

    plot_df = plot_df.reset_index(drop=True)
    plot_df["_participant_plot_id"] = np.arange(1, len(plot_df) + 1)

    if left_color is None:
        left_color = MODE_COLORS.get(left_mode, COLOR_NEUTRAL)    
    if right_color is None:
        right_color = MODE_COLORS.get(right_mode, COLOR_NEUTRAL)

    left_mean = plot_df["_left_value"].mean()
    right_mean = plot_df["_right_value"].mean()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, facecolor=BG)
    ax.set_facecolor(BG)  
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    # Connecting lines
    for _, row in plot_df.iterrows():
        left_val = row["_left_value"]
        right_val = row["_right_value"]

        if right_val > left_val:
            winner_color = right_color
        elif left_val > right_val:
            winner_color = left_color
        else:
            winner_color = COLOR_NEUTRAL

        ax.plot(
            [row["_participant_plot_id"], row["_participant_plot_id"]],
            [left_val, right_val],
            color=winner_color,
            alpha=line_alpha,
            linewidth=point_size / 8,   # thick band-like line
            solid_capstyle="round",
            zorder=1
        )

    # Endpoint markers
    ax.scatter(
        plot_df["_participant_plot_id"],
        plot_df["_left_value"],
        s=point_size,
        color=left_color,
        alpha=point_alpha,
        edgecolors=BG,
        linewidths=0.8,
        zorder=3
    )

    ax.scatter(
        plot_df["_participant_plot_id"],
        plot_df["_right_value"],
        s=point_size,
        color=right_color,
        alpha=point_alpha,
        edgecolors=BG,
        linewidths=0.8,
        zorder=3
    )

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=left_mode,
            markerfacecolor=left_color,
            markeredgecolor=BG,
            markersize=LEGEND_MARKER_SIZE
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=right_mode,
            markerfacecolor=right_color,
            markeredgecolor=BG,
            markersize=LEGEND_MARKER_SIZE
        ),
    ]

    if color_col is not None:
        # Add category legend entries for line colors
        unique_vals = list(plot_df[color_col].dropna().unique())
        for val in unique_vals:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    color=color_map.get(val, line_color),
                    lw=LINE_WIDTH_ERROR * 1.5,
                    label=str(val)
                )
            )

    if show_zero_line:
        ax.axhline(
            0,
            color=COLOR_LINE,
            linewidth=LINE_WIDTH_ERROR,
            linestyle="--",
            alpha=0.9,
            zorder=0
        )
        legend_elements.append(
            Line2D(
                [0], [0],
                color=COLOR_LINE,
                lw=LINE_WIDTH_ERROR,
                linestyle="--",
                label="Zero reference"
            )
        )

    if show_mean_lines:
        ax.axhline(
            left_mean,
            color=left_color,
            linewidth=LINE_WIDTH_ERROR * 3,
            linestyle="--",
            alpha=0.9,
            zorder=0
        )
        ax.axhline(
            right_mean,
            color=right_color,
            linewidth=LINE_WIDTH_ERROR * 3,
            linestyle="--",
            alpha=0.9,
            zorder=0
        )
        legend_elements.extend([
            Line2D(
                [0], [0],
                color=left_color,
                lw=LINE_WIDTH_ERROR * 3,
                linestyle="--",
                label=f"{left_mode} mean = {left_mean:.2f}"
            ),
            Line2D(
                [0], [0],
                color=right_color,
                lw=LINE_WIDTH_ERROR * 3,
                linestyle="--",
                label=f"{right_mode} mean = {right_mean:.2f}"
            ),
        ])

    ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=COLOR_TEXT)
    ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)

    ax.grid(True, axis="y", color=COLOR_GRID, alpha=GRID_ALPHA)
    ax.tick_params(colors=COLOR_TEXT)
    ax.set_xticks(plot_df["_participant_plot_id"])

    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max
        )

    ax.legend(
        handles=legend_elements,
        fontsize=LEGEND_FONT_SIZE,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=LEGEND_FRAME,
        labelcolor=COLOR_TEXT
    )
    
    for spine in ax.spines.values():
        spine.set_color(COLOR_TEXT)

    save_figure(fig, filename)
    return plot_df

def plot_mode_dumbbell_preference(
    df: pd.DataFrame,
    columns_by_mode: dict,
    left_mode: str,
    right_mode: str,
    title: str,
    filename: str,
    preferred_mode_col: str,
    y_label: str = "Score",
    x_label: str = "Participant",
    sort_by: str = "difference",
    ascending: bool = True,
    color_col: str | None = None,
    color_map: dict | None = None,
    left_color: str | None = None,
    right_color: str | None = None,
    line_color: str = COLOR_NEUTRAL,
    line_alpha: float = 0.8,
    point_alpha: float = 0.95,
    point_size: float = MARKER_SIZE * 12,
    show_mean_lines: bool = True,
    show_zero_line: bool = False,
    y_min=None,
    y_max=None,
):
    """
    Plot a dumbbell chart comparing two mode columns per participant.

    This version also allows:
    - left_mode/right_mode = "preferred" or "unpreferred"

    preferred_mode_col must contain the preferred mode name per row
    (for example "Action" or "Stealth").

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe, one row per participant.
    columns_by_mode : dict
        Mapping like:
        {
            "Stealth": "stealth_mean",
            "Action": "action_mean",
            "Adaptive": "adaptive_mean"
        }
    left_mode : str
        Mode shown as the left endpoint conceptually.
        Can also be "preferred" or "unpreferred".
    right_mode : str
        Mode shown as the right endpoint conceptually.
        Can also be "preferred" or "unpreferred".
    title : str
        Plot title.
    filename : str
        Output filename.
    preferred_mode_col : str
        Column containing each participant's preferred mode name.
    y_label : str
        Label for y-axis.
    x_label : str
        Label for x-axis.
    sort_by : str
        One of:
        - "difference"  -> sort by (right - left)
        - "left"        -> sort by left mode value
        - "right"       -> sort by right mode value
        - "none"        -> keep original row order
    ascending : bool
        Sort direction.
    color_col : str or None
        Optional column used to color connecting lines by category.
    color_map : dict or None
        Mapping from values in color_col to colors.
    left_color, right_color : str or None
        Optional fixed colors for endpoint markers.
        If left/right are "preferred"/"unpreferred", these default to neutral
        unless explicitly provided.
    line_color : str
        Default line color when color_col is None.
    line_alpha : float
        Alpha for connecting lines when color_col is used.
    point_alpha : float
        Alpha for endpoint markers.
    point_size : float
        Endpoint marker size.
    show_mean_lines : bool
        Whether to draw horizontal mean lines for the two modes.
    show_zero_line : bool
        Whether to draw a horizontal zero line.
    y_min, y_max : float or None
        Optional y-axis limits.

    Returns
    -------
    pd.DataFrame
        Dataframe used for plotting with columns:
        - _left_value
        - _right_value
        - _difference
        - _participant_plot_id
        - _resolved_left_mode
        - _resolved_right_mode
    """

    def _normalize_mode_name(mode_name: str) -> str:
        return str(mode_name).strip().lower()

    def _find_canonical_mode(mode_name: str, columns_by_mode: dict) -> str:
        normalized = _normalize_mode_name(mode_name)
        for key in columns_by_mode.keys():
            if _normalize_mode_name(key) == normalized:
                return key
        raise ValueError(f"Mode '{mode_name}' not found in columns_by_mode")

    def _opposite_mode(mode_name: str) -> str:
        normalized = _normalize_mode_name(mode_name)

        # Explicit Action/Stealth handling as requested
        if normalized == "action":
            return _find_canonical_mode("stealth", columns_by_mode)
        if normalized == "stealth":
            return _find_canonical_mode("action", columns_by_mode)

        raise ValueError(
            f"Cannot infer unpreferred mode for '{mode_name}'. "
            "This function expects preferred_mode_col to contain Action or Stealth."
        )

    def _resolve_mode(mode_spec: str, preferred_value: str) -> str:
        mode_spec_norm = _normalize_mode_name(mode_spec)

        if mode_spec_norm == "preferred":
            return _find_canonical_mode(preferred_value, columns_by_mode)
        elif mode_spec_norm == "unpreferred":
            preferred_canonical = _find_canonical_mode(preferred_value, columns_by_mode)
            return _opposite_mode(preferred_canonical)
        else:
            return _find_canonical_mode(mode_spec, columns_by_mode)

    # Validate preferred column
    if preferred_mode_col not in df.columns:
        raise ValueError(f"'{preferred_mode_col}' not found in dataframe")

    # Validate any explicit modes
    special_modes = {"preferred", "unpreferred"}
    if _normalize_mode_name(left_mode) not in special_modes:
        _find_canonical_mode(left_mode, columns_by_mode)
    if _normalize_mode_name(right_mode) not in special_modes:
        _find_canonical_mode(right_mode, columns_by_mode)

    if color_col is not None and color_col not in df.columns:
        raise ValueError(f"'{color_col}' not found in dataframe")

    plot_df = df.copy()

    # Resolve row-wise left/right modes and values
    resolved_left_modes = []
    resolved_right_modes = []
    left_values = []
    right_values = []

    for _, row in plot_df.iterrows():
        preferred_value = row[preferred_mode_col]

        resolved_left = _resolve_mode(left_mode, preferred_value)
        resolved_right = _resolve_mode(right_mode, preferred_value)

        left_col = columns_by_mode[resolved_left]
        right_col = columns_by_mode[resolved_right]

        if left_col not in plot_df.columns or right_col not in plot_df.columns:
            missing_cols = [c for c in [left_col, right_col] if c not in plot_df.columns]
            raise ValueError(f"Missing columns in dataframe: {missing_cols}")

        resolved_left_modes.append(resolved_left)
        resolved_right_modes.append(resolved_right)
        left_values.append(pd.to_numeric(row[left_col], errors="coerce"))
        right_values.append(pd.to_numeric(row[right_col], errors="coerce"))

    plot_df["_resolved_left_mode"] = resolved_left_modes
    plot_df["_resolved_right_mode"] = resolved_right_modes
    plot_df["_left_value"] = left_values
    plot_df["_right_value"] = right_values

    plot_df = plot_df.dropna(subset=["_left_value", "_right_value"]).copy()

    if plot_df.empty:
        raise ValueError("No valid rows left after cleaning mode columns")

    plot_df["_difference"] = plot_df["_right_value"] - plot_df["_left_value"]

    if sort_by == "difference":
        plot_df = plot_df.sort_values("_difference", ascending=ascending)
    elif sort_by == "left":
        plot_df = plot_df.sort_values("_left_value", ascending=ascending)
    elif sort_by == "right":
        plot_df = plot_df.sort_values("_right_value", ascending=ascending)
    elif sort_by == "none":
        plot_df = plot_df.copy()
    else:
        raise ValueError("sort_by must be one of: 'difference', 'left', 'right', 'none'")

    plot_df = plot_df.reset_index(drop=True)
    plot_df["_participant_plot_id"] = np.arange(1, len(plot_df) + 1)

    # Fixed endpoint colors if requested, otherwise use defaults
    if left_color is None:
        if _normalize_mode_name(left_mode) in special_modes:
            left_color = COLOR_NEUTRAL
        else:
            left_color = MODE_COLORS.get(_find_canonical_mode(left_mode, columns_by_mode), COLOR_NEUTRAL)

    if right_color is None:
        if _normalize_mode_name(right_mode) in special_modes:
            right_color = COLOR_NEUTRAL
        else:
            right_color = MODE_COLORS.get(_find_canonical_mode(right_mode, columns_by_mode), COLOR_NEUTRAL)

    # Optional category-based line colors, same behavior as original
    if color_col is not None:
        if color_map is None:
            unique_vals = plot_df[color_col].dropna().unique()
            color_map = {val: line_color for val in unique_vals}
        category_line_colors = plot_df[color_col].map(color_map).fillna(line_color)
    else:
        category_line_colors = pd.Series(line_color, index=plot_df.index)

    left_mean = plot_df["_left_value"].mean()
    right_mean = plot_df["_right_value"].mean()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI, facecolor=BG)
    ax.set_facecolor(BG)  
    
    if TRANSPARENT_BG:
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

    # Connecting lines colored by the higher endpoint's resolved mode color
    for idx, row in plot_df.iterrows():
        left_val = row["_left_value"]
        right_val = row["_right_value"]

        left_mode_resolved = row["_resolved_left_mode"]
        right_mode_resolved = row["_resolved_right_mode"]

        left_mode_color = MODE_COLORS.get(left_mode_resolved, left_color)
        right_mode_color = MODE_COLORS.get(right_mode_resolved, right_color)

        if right_val > left_val:
            winner_color = right_mode_color
        elif left_val > right_val:
            winner_color = left_mode_color
        else:
            winner_color = COLOR_NEUTRAL

        # Preserve original optional category coloring behavior only if desired
        # by blending choice: category_line_colors is only used when explicit category
        # coloring is requested; otherwise winner_color is used.
        actual_line_color = category_line_colors.loc[idx] if color_col is not None else winner_color
        actual_line_alpha = line_alpha if color_col is not None else 0.35

        ax.plot(
            [row["_participant_plot_id"], row["_participant_plot_id"]],
            [left_val, right_val],
            color=actual_line_color,
            alpha=actual_line_alpha,
            linewidth=point_size / 8,
            solid_capstyle="round",
            zorder=1
        )

    # Endpoint markers
    if _normalize_mode_name(left_mode) in special_modes:
        left_point_colors = plot_df["_resolved_left_mode"].map(lambda m: MODE_COLORS.get(m, left_color))
    else:
        left_point_colors = pd.Series(left_color, index=plot_df.index)

    if _normalize_mode_name(right_mode) in special_modes:
        right_point_colors = plot_df["_resolved_right_mode"].map(lambda m: MODE_COLORS.get(m, right_color))
    else:
        right_point_colors = pd.Series(right_color, index=plot_df.index)

    ax.scatter(
        plot_df["_participant_plot_id"],
        plot_df["_left_value"],
        s=point_size,
        c=left_point_colors,
        alpha=point_alpha,
        edgecolors=BG,
        linewidths=0.8,
        zorder=3
    )

    ax.scatter(
        plot_df["_participant_plot_id"],
        plot_df["_right_value"],
        s=point_size,
        c=right_point_colors,
        alpha=point_alpha,
        edgecolors=BG,
        linewidths=0.8,
        zorder=3
    )

    legend_elements = []

    special_modes = {"preferred", "unpreferred"}
    
    if left_mode.lower() not in special_modes:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=left_mode,
                markerfacecolor=left_color,
                markeredgecolor=BG,
                markersize=LEGEND_MARKER_SIZE
            )
        )
    
    if right_mode.lower() not in special_modes:
        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=right_mode,
                markerfacecolor=right_color,
                markeredgecolor=BG,
                markersize=LEGEND_MARKER_SIZE
            )
        )

    # Add preference legend when using preferred/unpreferred
    uses_preference_modes = (
        _normalize_mode_name(left_mode) in special_modes
        or _normalize_mode_name(right_mode) in special_modes
    )

    if uses_preference_modes:
        try:
            action_key = _find_canonical_mode("action", columns_by_mode)
            stealth_key = _find_canonical_mode("stealth", columns_by_mode)

            legend_elements.extend([
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    label="Action-leaning participant",
                    markerfacecolor=MODE_COLORS.get(action_key, COLOR_NEUTRAL),
                    markeredgecolor=BG,
                    markersize=LEGEND_MARKER_SIZE
                ),
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    label="Stealth-leaning participant",
                    markerfacecolor=MODE_COLORS.get(stealth_key, COLOR_NEUTRAL),
                    markeredgecolor=BG,
                    markersize=LEGEND_MARKER_SIZE
                ),
            ])
        except ValueError:
            pass

    if color_col is not None:
        unique_vals = list(plot_df[color_col].dropna().unique())
        for val in unique_vals:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    color=color_map.get(val, line_color),
                    lw=LINE_WIDTH_ERROR * 1.5,
                    label=str(val)
                )
            )

    if show_zero_line:
        ax.axhline(
            0,
            color=COLOR_LINE,
            linewidth=LINE_WIDTH_ERROR,
            linestyle="--",
            alpha=0.9,
            zorder=0
        )
        legend_elements.append(
            Line2D(
                [0], [0],
                color=COLOR_LINE,
                lw=LINE_WIDTH_ERROR,
                linestyle="--",
                label="Zero reference"
            )
        )

    if show_mean_lines:
        ax.axhline(
            left_mean,
            color=left_color,
            linewidth=LINE_WIDTH_ERROR * 3,
            linestyle="--",
            alpha=0.9,
            zorder=0
        )
        ax.axhline(
            right_mean,
            color=right_color,
            linewidth=LINE_WIDTH_ERROR * 3,
            linestyle="--",
            alpha=0.9,
            zorder=0
        )
        legend_elements.extend([
            Line2D(
                [0], [0],
                color=left_color,
                lw=LINE_WIDTH_ERROR * 3,
                linestyle="--",
                label=f"{left_mode} mean = {left_mean:.2f}"
            ),
            Line2D(
                [0], [0],
                color=right_color,
                lw=LINE_WIDTH_ERROR * 3,
                linestyle="--",
                label=f"{right_mode} mean = {right_mean:.2f}"
            ),
        ])

    ax.set_title(title, fontsize=TITLE_FONT_SIZE, color=COLOR_TEXT)
    ax.set_xlabel(x_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)
    ax.set_ylabel(y_label, fontsize=LABEL_FONT_SIZE, color=COLOR_TEXT)

    ax.grid(True, axis="y", color=COLOR_GRID, alpha=GRID_ALPHA)
    ax.tick_params(colors=COLOR_TEXT)
    ax.set_xticks(plot_df["_participant_plot_id"])

    if y_min is not None or y_max is not None:
        current_min, current_max = ax.get_ylim()
        ax.set_ylim(
            y_min if y_min is not None else current_min,
            y_max if y_max is not None else current_max
        )

    ax.legend(
        handles=legend_elements,
        fontsize=LEGEND_FONT_SIZE,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX_TO_ANCHOR,
        frameon=LEGEND_FRAME,
        labelcolor=COLOR_TEXT
    )
    
    for spine in ax.spines.values():
        spine.set_color(COLOR_TEXT)
    

    save_figure(fig, filename)
    return plot_df