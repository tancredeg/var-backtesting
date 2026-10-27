"""Figures of the study (matplotlib, static PNG for the README).

Visual conventions: one quiet surface, hairline solid grid, thin marks, categorical hues in
a fixed order per model (colour follows the model in every figure), Basel zones in the
reserved status colours and always named in a legend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from var_es.forecast import ForecastResult, var_col

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LOSS_GRAY = "#b9b8b1"

MODEL_COLORS: dict[str, str] = {
    "Historical": "#2a78d6",
    "Normal": "#eb6834",
    "Student-t": "#1baf7a",
    "FHS-GARCH": "#eda100",
    "Monte Carlo": "#e87ba4",
}
ZONE_COLORS: dict[str, str] = {"green": "#0ca30c", "yellow": "#fab219", "red": "#d03b3b"}
ZONE_WASH: dict[str, str] = {"green": "#e6f4e6", "yellow": "#fdf0d2", "red": "#f8dede"}


def apply_style() -> None:
    """Global rcParams shared by every figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def _pct_axis(ax: Axes) -> None:
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))


def plot_var_paths(
    results: Mapping[str, ForecastResult],
    models: Sequence[str],
    level: float,
    zooms: Mapping[str, tuple[str, str]],
) -> Figure:
    """Daily loss vs VaR of selected models: full sample on top, crisis zooms below."""
    apply_style()
    fig = plt.figure(figsize=(12, 6.6), layout="constrained")
    grid = fig.add_gridspec(2, len(zooms), height_ratios=[1.15, 1])
    top = fig.add_subplot(grid[0, :])
    axes = [top] + [fig.add_subplot(grid[1, k]) for k in range(len(zooms))]
    windows = [None, *zooms.values()]
    titles = [
        f"One-day {100 * level:g} % VaR vs realised loss, full out-of-sample period",
        *zooms.keys(),
    ]
    base = results[models[0]].frame
    for ax, win, title in zip(axes, windows, titles, strict=True):
        f = base if win is None else base.loc[win[0] : win[1]]
        loss = f["loss"].clip(lower=0)
        ax.vlines(
            f.index,
            0,
            loss,
            color=LOSS_GRAY,
            lw=0.6 if win is None else 1.4,
            label="Daily loss (gains not shown)",
        )
        for m in models:
            g = results[m].frame.loc[f.index]
            ax.plot(
                g.index,
                g[var_col(level)],
                color=MODEL_COLORS[m],
                lw=1.1 if win is None else 1.8,
                label=f"{m} VaR",
            )
            hit = g["loss"] > g[var_col(level)]
            ax.scatter(
                g.index[hit],
                g["loss"][hit],
                s=10 if win is None else 26,
                color=MODEL_COLORS[m],
                edgecolor=SURFACE,
                linewidth=0.8,
                zorder=3,
            )
        ax.set_title(title, fontsize=10 if win is None else 9.5)
        _pct_axis(ax)
        ax.set_ylim(0, None)
        if win is None:
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.set_ylabel("Loss, % of portfolio value")
        else:
            ax.xaxis.set_major_locator(
                mdates.MonthLocator(
                    bymonth=None,
                    interval=1 if (pd.Timestamp(win[1]) - pd.Timestamp(win[0])).days < 200 else 3,
                )
            )
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    handles, labels = top.get_legend_handles_labels()
    handles.append(Line2D([], [], ls="", marker="o", color=INK_2, markersize=4))
    labels.append("Exception (loss > VaR), coloured by model")
    top.legend(handles, labels, loc="upper right", ncol=2)
    return fig


def plot_traffic_light(
    zones: Mapping[str, pd.DataFrame],
    zooms: Mapping[str, tuple[str, str]],
    level: float,
    bounds: tuple[int, int],
) -> Figure:
    """Rolling 250-day Basel zone per model (strips) and exception counts in crisis zooms."""
    apply_style()
    names = list(zones)
    fig = plt.figure(figsize=(12, 6.8), layout="constrained")
    grid = fig.add_gridspec(2, len(zooms), height_ratios=[1, 1.05])
    top = fig.add_subplot(grid[0, :])
    code = {"green": 0, "yellow": 1, "red": 2}
    cmap = mpl.colors.ListedColormap([ZONE_COLORS[z] for z in ("green", "yellow", "red")])
    idx = zones[names[0]].index
    mat = np.array([[code[z] for z in zones[n].reindex(idx)["zone"]] for n in names])
    x = mdates.date2num(pd.DatetimeIndex(idx).to_pydatetime())
    top.pcolormesh(
        np.append(x, x[-1] + 1),
        np.arange(len(names) + 1) - 0.5,
        mat,
        cmap=cmap,
        vmin=0,
        vmax=2,
        shading="flat",
        rasterized=True,
    )
    for k in range(1, len(names)):
        top.axhline(k - 0.5, color=SURFACE, lw=3)
    top.set_yticks(range(len(names)), names)
    top.invert_yaxis()
    top.grid(False)
    top.tick_params(axis="y", length=0)
    top.xaxis.set_major_locator(mdates.YearLocator(2))
    top.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for start, end in zooms.values():
        top.axvspan(
            float(mdates.date2num(pd.Timestamp(start))),
            float(mdates.date2num(pd.Timestamp(end))),
            ymin=0,
            ymax=1,
            fill=False,
            edgecolor=INK,
            lw=1.0,
        )
    top.set_title(
        f"Basel traffic light, rolling 250-day count of {100 * level:g} % VaR "
        "exceptions (black frames: zooms below)"
    )
    green_max, yellow_max = bounds
    legend = [
        Patch(color=ZONE_COLORS["green"], label=f"Green (0-{green_max})"),
        Patch(color=ZONE_COLORS["yellow"], label=f"Yellow ({green_max + 1}-{yellow_max})"),
        Patch(color=ZONE_COLORS["red"], label=f"Red (>= {yellow_max + 1})"),
    ]
    top.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)

    ymax = max(int(z["count"].max()) for z in zones.values()) + 2
    for k, (title, (start, end)) in enumerate(zooms.items()):
        ax = fig.add_subplot(grid[1, k])
        ax.axhspan(-0.5, green_max + 0.5, color=ZONE_WASH["green"], lw=0)
        ax.axhspan(green_max + 0.5, yellow_max + 0.5, color=ZONE_WASH["yellow"], lw=0)
        ax.axhspan(yellow_max + 0.5, ymax, color=ZONE_WASH["red"], lw=0)
        for rank, n in enumerate(names):
            z = zones[n].loc[start:end]
            # earlier series drawn wider underneath: identical counts remain visible as stripes
            ax.step(
                z.index,
                z["count"],
                where="post",
                color=MODEL_COLORS[n],
                lw=3.4 - 0.45 * rank,
                label=n,
            )
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("Exceptions in last 250 days" if k == 0 else "")
        span = (pd.Timestamp(end) - pd.Timestamp(start)).days
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1 if span < 200 else 3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        if k == len(zooms) - 1:
            ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_clustering(table: pd.DataFrame, levels: Sequence[float], horizon: int) -> Figure:
    """Share of exceptions arriving within ``horizon`` days of the previous one."""
    apply_style()
    fig, axes = plt.subplots(1, len(levels), figsize=(11, 3.4), layout="constrained", sharex=True)
    for ax, lv in zip(np.atleast_1d(axes), levels, strict=True):
        t = table[np.isclose(table["level"], lv)]
        y = np.arange(len(t))
        ax.barh(y, t["share_within"], height=0.42, color=[MODEL_COLORS[m] for m in t["model"]])
        iid = float(t["iid_share"].iloc[0])
        ax.axvline(iid, color=INK, lw=1.2)
        ax.text(
            iid + 0.005, -0.78, f"i.i.d. benchmark {iid:.0%}", color=INK_2, fontsize=8, va="center"
        )
        for yi, v in zip(y, t["share_within"], strict=True):
            ax.text(v + 0.01, yi, f"{v:.0%}", va="center", color=INK_2, fontsize=8.5)
        ax.set_yticks(y, t["model"])
        ax.set_ylim(len(t) - 0.5, -1.1)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_xlim(0, 0.75)
        ax.set_title(f"{100 * lv:g} % VaR: exceptions within {horizon} days of the previous one")
    return fig
