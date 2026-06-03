#!/usr/bin/env python
"""Draw an HA/PB2 tanglegram with only HA V and PB2 V states highlighted."""

from __future__ import annotations

import argparse
import math
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


ID_RE = re.compile(r"EPI_ISL_\d+", re.IGNORECASE)
TIP_RE = re.compile(r"(?<=[(,])('[^']+'|\"[^\"]+\"|[^():,;]+)(?=:)")

# =========================
# User-editable settings
# =========================
#
# Change these values directly if you want a different color scheme or if your
# metadata column names differ. In your current metadata file the HA column is
# named HA_197; if you later rename it to HA_198, change HA_STATE_COLUMN below.
ID_COLUMN = "Isolate ID"
HA_STATE_COLUMN = "HA_197"
PB2_STATE_COLUMN = "PB2_627"
HA_CLADE_COLUMN = "HA_clade"
HIGHLIGHT_STATE = "V"

# Tree branch/tip colors.
# Current default: all tree branches are black and tip circles are hidden.
# Set COLOR_V_BRANCHES = True if you later want terminal V branches colored.
COLOR_V_BRANCHES = False
SHOW_TIP_POINTS = False
TREE_BRANCH_COLOR = "#111111"
HA_V_COLOR = "#E69F00"
PB2_V_COLOR = "#009E73"
NON_V_COLOR = "#D8DCE2"
TREE_BASE_COLOR = "#D8DCE2"

# HA clade branch colors. "others" remains black.
COLOR_HA_BY_CLADE = True
OTHERS_CLADE_NAMES = {"", "other", "others", "unknown", "nan", "none"}
HA_CLADE_COLORS = {
    "B4.6": "#8EC7D2",
    "B4.7.1": "#F2C78D",
    "B4.7.2": "#B9D9A6",
    "B4.7.3": "#C7B6D9",
    "B4.7.4": "#F0B6B6",
}
HA_CLADE_PALETTE = [
    "#8EC7D2",
    "#F2C78D",
    "#B9D9A6",
    "#C7B6D9",
    "#F0B6B6",
    "#B7C9E2",
    "#D8C49A",
    "#A9D6C2",
    "#D9B8CE",
    "#C8D6A4",
]

# Tree layout.
MIDPOINT_ROOT = True
SORT_DESCENDING = True
UNTANGLE_BY_PARTNER = False
FLIP_HA_VERTICAL = False

# Tanglegram link colors.
LINK_BOTH_V_COLOR = "#0072B2"
LINK_HA_ONLY_V_COLOR = HA_V_COLOR
LINK_PB2_ONLY_V_COLOR = PB2_V_COLOR
LINK_NEITHER_V_COLOR = "#111111"

# Tanglegram link opacity. Increase alpha to make that link class more visible.
LINK_BOTH_V_ALPHA = 0.68
LINK_HA_ONLY_V_ALPHA = 0.54
LINK_PB2_ONLY_V_ALPHA = 0.54
LINK_NEITHER_V_ALPHA = 0.14

TEXT_COLOR = "#22262A"
SHOW_SCALE_BARS = True
SCALE_BAR_FRACTION = 0.18
SCALE_BAR_COLOR = "#111111"
SCALE_BAR_WIDTH = 1.6
SCALE_BAR_LABEL = "subs/site"


@dataclass(frozen=True)
class PlotConfig:
    ha_tree: Path
    pb2_tree: Path
    metadata: Path
    output: Path
    id_column: str = ID_COLUMN
    ha_column: str = HA_STATE_COLUMN
    pb2_column: str = PB2_STATE_COLUMN
    ha_clade_column: str = HA_CLADE_COLUMN
    highlight_state: str = HIGHLIGHT_STATE
    width: float = 10.8
    height: float | None = None
    max_height: float = 11.0
    dpi: int = 300
    tip_size: float = 9.0
    branch_width: float = 1.25
    show_tip_points: bool = SHOW_TIP_POINTS
    midpoint_root: bool = MIDPOINT_ROOT
    sort_descending: bool = SORT_DESCENDING
    untangle_by_partner: bool = UNTANGLE_BY_PARTNER
    flip_ha_vertical: bool = FLIP_HA_VERTICAL
    link_width: float = 0.50
    show_labels: bool = False
    label_every: int = 1
    label_font_size: float = 4.0
    untangle_iterations: int = 0
    show_scale_bars: bool = SHOW_SCALE_BARS


def normalize_tip(label: object) -> str:
    """Map tree labels or metadata names to the shared GISAID isolate ID."""
    text = str(label).strip().strip("'\"")
    match = ID_RE.search(text)
    if match:
        return match.group(0).upper()
    return text.split("|", 1)[0].strip()


def normalize_state(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip().upper()


def normalize_clade(value: object) -> str:
    if value is None:
        return "others"
    if isinstance(value, float) and math.isnan(value):
        return "others"
    clade = str(value).strip()
    if clade.lower() in OTHERS_CLADE_NAMES:
        return "others"
    return clade


class NewickNode:
    def __init__(self, name: str = "", length: float = 0.0, children: list["NewickNode"] | None = None):
        self.name = name
        self.length = length
        self.children = children or []


def strip_newick_comments(text: str) -> str:
    return re.sub(r"\[[^\[\]]*\]", "", text)


def parse_newick(text: str) -> NewickNode:
    text = strip_newick_comments(text.strip())
    if text.endswith(";"):
        text = text[:-1]

    def parse_subtree(pos: int) -> tuple[NewickNode, int]:
        children: list[NewickNode] = []
        name = ""
        if text[pos] == "(":
            pos += 1
            while True:
                child, pos = parse_subtree(pos)
                children.append(child)
                if pos >= len(text):
                    raise ValueError("Unexpected end of Newick while parsing children.")
                if text[pos] == ",":
                    pos += 1
                    continue
                if text[pos] == ")":
                    pos += 1
                    break

        start = pos
        while pos < len(text) and text[pos] not in ":,()":
            pos += 1
        name = text[start:pos].strip().strip("'\"")

        length = 0.0
        if pos < len(text) and text[pos] == ":":
            pos += 1
            start = pos
            while pos < len(text) and text[pos] not in ",()":
                pos += 1
            token = text[start:pos].strip()
            if token:
                length = float(token)
        return NewickNode(name=name, length=length, children=children), pos

    root, pos = parse_subtree(0)
    if pos != len(text):
        raise ValueError(f"Could not parse full Newick string near: {text[pos:pos + 30]}")
    return root


def build_adjacency(root: NewickNode) -> tuple[dict[int, list[tuple[int, float]]], dict[int, str], list[int]]:
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    names: dict[int, str] = {}
    leaves: list[int] = []

    def visit(node: NewickNode, parent_id: int | None = None) -> int:
        node_id = id(node)
        names[node_id] = node.name
        if parent_id is not None:
            length = max(float(node.length), 0.0)
            adjacency[parent_id].append((node_id, length))
            adjacency[node_id].append((parent_id, length))
        if not node.children:
            leaves.append(node_id)
        for child in node.children:
            visit(child, node_id)
        return node_id

    visit(root)
    return adjacency, names, leaves


def farthest_leaf(
    start: int,
    adjacency: dict[int, list[tuple[int, float]]],
    leaves: set[int],
) -> tuple[int, dict[int, int | None], dict[int, float]]:
    parent: dict[int, int | None] = {start: None}
    distance: dict[int, float] = {start: 0.0}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor, length in adjacency[node]:
            if neighbor == parent.get(node):
                continue
            parent[neighbor] = node
            distance[neighbor] = distance[node] + length
            stack.append(neighbor)
    farthest = max(leaves, key=lambda leaf: distance.get(leaf, -1.0))
    return farthest, parent, distance


def midpoint_root_newick(text: str) -> str:
    root = parse_newick(text)
    adjacency, names, leaves = build_adjacency(root)
    if len(leaves) < 2:
        return text

    leaf_set = set(leaves)
    first, _, _ = farthest_leaf(leaves[0], adjacency, leaf_set)
    second, parent, distance = farthest_leaf(first, adjacency, leaf_set)
    total = distance[second]
    if total <= 0:
        return text

    path = [second]
    while path[-1] != first:
        previous = parent[path[-1]]
        if previous is None:
            return text
        path.append(previous)
    path.reverse()

    midpoint = total / 2.0
    walked = 0.0
    root_id: int
    for left, right in zip(path, path[1:]):
        edge_length = next(length for neighbor, length in adjacency[left] if neighbor == right)
        if walked + edge_length > midpoint:
            left_length = midpoint - walked
            right_length = edge_length - left_length
            root_id = -1
            names[root_id] = ""
            adjacency[left] = [(neighbor, length) for neighbor, length in adjacency[left] if neighbor != right]
            adjacency[right] = [(neighbor, length) for neighbor, length in adjacency[right] if neighbor != left]
            adjacency[root_id].append((left, left_length))
            adjacency[root_id].append((right, right_length))
            adjacency[left].append((root_id, left_length))
            adjacency[right].append((root_id, right_length))
            break
        if math.isclose(walked + edge_length, midpoint):
            root_id = right
            break
        walked += edge_length
    else:
        root_id = path[-1]

    def format_length(length: float) -> str:
        return f"{length:.10g}"

    def emit(node: int, parent_node: int | None = None, length_to_parent: float | None = None) -> str:
        children = [(neighbor, length) for neighbor, length in adjacency[node] if neighbor != parent_node]
        if children:
            label = "(" + ",".join(emit(neighbor, node, length) for neighbor, length in children) + ")"
        else:
            label = names[node] or f"node_{abs(node)}"
        if length_to_parent is None:
            return label
        return f"{label}:{format_length(length_to_parent)}"

    return emit(root_id) + ";"


def read_metadata(config: PlotConfig) -> pd.DataFrame:
    usecols = [config.id_column, config.ha_column, config.pb2_column]
    if COLOR_HA_BY_CLADE:
        usecols.append(config.ha_clade_column)
    frame = pd.read_excel(config.metadata, usecols=usecols)
    frame = frame.dropna(subset=[config.id_column]).copy()
    frame["match_id"] = frame[config.id_column].map(normalize_tip)
    frame[config.ha_column] = frame[config.ha_column].map(normalize_state)
    frame[config.pb2_column] = frame[config.pb2_column].map(normalize_state)
    if COLOR_HA_BY_CLADE:
        frame[config.ha_clade_column] = frame[config.ha_clade_column].map(normalize_clade)

    duplicated = frame["match_id"].duplicated(keep=False)
    if duplicated.any():
        examples = ", ".join(frame.loc[duplicated, "match_id"].head(6))
        raise ValueError(f"metadata contains duplicated isolate IDs after normalization: {examples}")
    return frame.set_index("match_id", drop=False)


def normalized_newick_file(tree_path: Path, keep_ids: set[str] | None = None, midpoint_root: bool = True) -> Path:
    """Write a temporary Newick where leaf names are normalized to isolate IDs."""
    text = tree_path.read_text(encoding="utf-8", errors="ignore")

    seen: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        raw = match.group(1)
        tip = normalize_tip(raw)
        if keep_ids is not None and tip not in keep_ids:
            return raw
        if tip in seen:
            raise ValueError(f"duplicated tip after normalization in {tree_path}: {tip}")
        seen.add(tip)
        return tip

    normalized = TIP_RE.sub(replace, text)
    if midpoint_root:
        normalized = midpoint_root_newick(normalized)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".nwk", delete=False, encoding="utf-8")
    with tmp:
        tmp.write(normalized)
    return Path(tmp.name)


def load_baltic_tree(tree_path: Path, midpoint_root: bool):
    try:
        import baltic as bt
    except ImportError:
        pixi_site_packages = Path(__file__).resolve().parents[1] / ".pixi" / "envs" / "default" / "Lib" / "site-packages"
        if pixi_site_packages.exists():
            sys.path.append(str(pixi_site_packages))
            try:
                import baltic as bt
            except ImportError as exc:
                raise SystemExit(
                    "The 'baltic' package is not installed in this Python environment. "
                    "Create the environment with: conda env create -f environment.yml"
                ) from exc
        else:
            raise SystemExit(
                "The 'baltic' package is not installed in this Python environment. "
                "Create the environment with: conda env create -f environment.yml"
            )

    normalized_path = normalized_newick_file(tree_path, midpoint_root=midpoint_root)
    return bt.loadNewick(str(normalized_path), absoluteTime=False)


def external_map(tree) -> dict[str, object]:
    leaves = tree.getExternal()
    return {normalize_tip(leaf.name): leaf for leaf in leaves}


def state_color(state: str, highlight_state: str, v_color: str) -> str:
    return v_color if state == highlight_state else NON_V_COLOR


def build_ha_clade_colors(metadata: pd.DataFrame, config: PlotConfig) -> dict[str, str]:
    clades = sorted(
        clade
        for clade in metadata[config.ha_clade_column].dropna().unique()
        if normalize_clade(clade) != "others"
    )
    colors: dict[str, str] = {}
    for index, clade in enumerate(clades):
        colors[clade] = HA_CLADE_COLORS.get(clade, HA_CLADE_PALETTE[index % len(HA_CLADE_PALETTE)])
    return colors


def ha_clade_branch_color(metadata: pd.DataFrame, config: PlotConfig, clade_colors: dict[str, str]) -> Callable[[object], str]:
    def color(obj: object) -> str:
        if getattr(obj, "branchType", "") == "leaf":
            tip_ids = [normalize_tip(getattr(obj, "name", ""))]
        else:
            tip_ids = [normalize_tip(leaf) for leaf in (getattr(obj, "leaves", []) or [])]

        clades = {
            metadata[config.ha_clade_column].get(tip_id, "others")
            for tip_id in tip_ids
        }
        if "others" not in clades and len(clades) == 1:
            return clade_colors.get(next(iter(clades)), TREE_BRANCH_COLOR)
        return TREE_BRANCH_COLOR

    return color


def branch_color(column: str, metadata: pd.DataFrame, highlight_state: str, v_color: str) -> Callable[[object], str]:
    def color(obj: object) -> str:
        if not COLOR_V_BRANCHES:
            return TREE_BRANCH_COLOR

        if getattr(obj, "branchType", "") == "leaf":
            name = getattr(obj, "name", "")
            state = metadata[column].get(normalize_tip(name), "")
            return state_color(state, highlight_state, v_color)

        return TREE_BRANCH_COLOR

    return color


def link_color(ha_state: str, pb2_state: str, highlight_state: str) -> str:
    ha_v = ha_state == highlight_state
    pb2_v = pb2_state == highlight_state
    if ha_v and pb2_v:
        return LINK_BOTH_V_COLOR
    if ha_v:
        return LINK_HA_ONLY_V_COLOR
    if pb2_v:
        return LINK_PB2_ONLY_V_COLOR
    return LINK_NEITHER_V_COLOR


def link_alpha(ha_state: str, pb2_state: str, highlight_state: str) -> float:
    ha_v = ha_state == highlight_state
    pb2_v = pb2_state == highlight_state
    if ha_v and pb2_v:
        return LINK_BOTH_V_ALPHA
    if ha_v:
        return LINK_HA_ONLY_V_ALPHA
    if pb2_v:
        return LINK_PB2_ONLY_V_ALPHA
    return LINK_NEITHER_V_ALPHA


def prune_to_common_tips(tree, common_ids: set[str]):
    keep = [leaf for leaf in tree.getExternal() if normalize_tip(leaf.name) in common_ids]
    return tree.reduceTree(keep)


def sort_tree_by_partner(tree, partner_y: dict[str, float], descending: bool = False) -> None:
    """Order child branches by the partner tree's mean y position when baltic supports it."""
    def score(branch: object) -> float:
        leaves = getattr(branch, "leaves", []) or []
        values = [partner_y[normalize_tip(leaf)] for leaf in leaves if normalize_tip(leaf) in partner_y]
        if values:
            return sum(values) / len(values)
        return getattr(branch, "y", 0.0)

    try:
        tree.sortBranches(sort_function=score, descending=descending)
    except TypeError:
        try:
            tree.sortBranches(sortFunction=score, descending=descending)
        except TypeError:
            tree.sortBranches(descending=descending)


def sort_tree_plain(tree, descending: bool) -> None:
    try:
        tree.sortBranches(descending=descending)
    except TypeError:
        tree.sortBranches()
    tree.drawTree()


def untangle_trees(left, right, iterations: int, descending: bool) -> None:
    for _ in range(max(0, iterations)):
        left.drawTree()
        right.drawTree()
        right_y = {normalize_tip(leaf.name): leaf.y for leaf in right.getExternal()}
        sort_tree_by_partner(left, right_y, descending=descending)
        left.drawTree()
        left_y = {normalize_tip(leaf.name): leaf.y for leaf in left.getExternal()}
        sort_tree_by_partner(right, left_y, descending=descending)
    left.drawTree()
    right.drawTree()


def tree_extent(tree) -> tuple[float, float]:
    leaves = tree.getExternal()
    y_values = [leaf.y for leaf in leaves]
    return min(y_values), max(y_values)


def nice_scale_length(tree_height: float, fraction: float = SCALE_BAR_FRACTION) -> float:
    target = max(tree_height * fraction, 1e-12)
    exponent = math.floor(math.log10(target))
    unit = 10 ** exponent
    base = target / unit
    if base >= 5:
        nice = 5
    elif base >= 2:
        nice = 2
    else:
        nice = 1
    return nice * unit


def draw_scale_bar(ax, tree, side: str, y_min: float, pad: float) -> None:
    scale = nice_scale_length(max(tree.treeHeight, 1e-12))
    y = y_min - pad * 0.56
    tick = pad * 0.09
    if side == "left":
        x0 = tree.treeHeight * 0.06
        x1 = x0 + scale
    else:
        x0 = -tree.treeHeight * 0.06
        x1 = x0 - scale
    x_mid = (x0 + x1) / 2

    ax.plot([x0, x1], [y, y], color=SCALE_BAR_COLOR, lw=SCALE_BAR_WIDTH, solid_capstyle="butt", clip_on=False)
    ax.plot([x0, x0], [y - tick, y + tick], color=SCALE_BAR_COLOR, lw=SCALE_BAR_WIDTH, clip_on=False)
    ax.plot([x1, x1], [y - tick, y + tick], color=SCALE_BAR_COLOR, lw=SCALE_BAR_WIDTH, clip_on=False)
    ax.text(
        x_mid,
        y - pad * 0.24,
        f"{scale:.3g} {SCALE_BAR_LABEL}",
        ha="center",
        va="top",
        fontsize=8.5,
        color=TEXT_COLOR,
        family="Arial",
        clip_on=False,
    )


def flip_y(y: float, y_min: float, y_max: float) -> float:
    return y_min + y_max - y


def plot_tree(
    tree,
    ax,
    side: str,
    color_func: Callable[[object], str],
    branch_width: float,
    tip_size: float,
    show_tip_points: bool,
) -> None:
    x_sign = 1 if side == "left" else -1

    def x_attr(obj: object) -> float:
        return x_sign * getattr(obj, "height", 0.0)

    tree.plotTree(ax, x_attr=x_attr, colour=color_func, width=branch_width)
    if show_tip_points:
        tree.plotPoints(
            ax,
            x_attr=x_attr,
            target=lambda obj: getattr(obj, "branchType", "") == "leaf",
            size=tip_size,
            colour=color_func,
            zorder=4,
        )


def draw_labels(ax, tree, side: str, every: int, font_size: float) -> None:
    if every <= 0:
        every = 1
    leaves = sorted(tree.getExternal(), key=lambda leaf: leaf.y)
    x_sign = 1 if side == "left" else -1
    align = "left" if side == "left" else "right"
    x_pad = 0.01 * max(tree.treeHeight, 1.0)
    for index, leaf in enumerate(leaves):
        if index % every:
            continue
        ax.text(
            x_sign * leaf.height + x_sign * x_pad,
            leaf.y,
            normalize_tip(leaf.name),
            ha=align,
            va="center",
            fontsize=font_size,
            color=TEXT_COLOR,
        )


def add_legend(fig, config: PlotConfig, ha_clade_colors: dict[str, str]) -> None:
    def legend_line(color: str, label: str, lw: float = 3.0) -> Line2D:
        return Line2D([0], [0], color=color, lw=lw, label=label, solid_capstyle="round")

    handles = [legend_line(TREE_BRANCH_COLOR, "tree branches", lw=3.6)]
    if COLOR_HA_BY_CLADE and ha_clade_colors:
        handles.extend(
            legend_line(color, f"HA clade {clade}", lw=3.4)
            for clade, color in ha_clade_colors.items()
        )
    if COLOR_V_BRANCHES:
        handles.extend(
            [
                legend_line(HA_V_COLOR, f"HA {config.ha_column}={config.highlight_state}", lw=3.4),
                legend_line(PB2_V_COLOR, f"PB2 {config.pb2_column}={config.highlight_state}", lw=3.4),
                legend_line(NON_V_COLOR, "all other branches/tips", lw=3.4),
            ]
        )
    handles.extend(
        [
            legend_line(LINK_BOTH_V_COLOR, "link both V", lw=3.6),
            legend_line(LINK_HA_ONLY_V_COLOR, "link HA only V", lw=3.6),
            legend_line(LINK_PB2_ONLY_V_COLOR, "link PB2 only V", lw=3.6),
            legend_line(LINK_NEITHER_V_COLOR, "link neither V", lw=3.6),
        ]
    )
    fig.legend(
        handles=handles,
        loc="center right",
        ncol=1,
        frameon=False,
        prop={"family": "Arial", "size": 8.5},
        handlelength=1.25,
        handletextpad=0.75,
        labelspacing=0.45,
        borderaxespad=0.0,
        bbox_to_anchor=(0.985, 0.5),
    )


def save_figure_outputs(fig, output: Path) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    written = [output]
    fig.savefig(output, bbox_inches="tight")

    if output.suffix.lower() != ".png":
        png_path = output.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight")
        written.append(png_path)

    if output.suffix.lower() not in {".tif", ".tiff"}:
        tiff_path = output.with_suffix(".tiff")
        fig.savefig(tiff_path, bbox_inches="tight")
        written.append(tiff_path)

    return written


def draw_tanglegram(config: PlotConfig) -> dict[str, object]:
    metadata = read_metadata(config)
    ha_clade_colors = build_ha_clade_colors(metadata, config) if COLOR_HA_BY_CLADE else {}
    left = load_baltic_tree(config.ha_tree, midpoint_root=config.midpoint_root)
    right = load_baltic_tree(config.pb2_tree, midpoint_root=config.midpoint_root)

    left_ids = set(external_map(left))
    right_ids = set(external_map(right))
    meta_ids = set(metadata.index)
    common_ids = left_ids & right_ids & meta_ids
    if not common_ids:
        raise ValueError("No shared isolate IDs found among HA tree, PB2 tree, and metadata.")

    missing_left = len((right_ids & meta_ids) - left_ids)
    missing_right = len((left_ids & meta_ids) - right_ids)
    missing_meta = len((left_ids & right_ids) - meta_ids)

    left = prune_to_common_tips(left, common_ids)
    right = prune_to_common_tips(right, common_ids)
    if config.untangle_by_partner and config.untangle_iterations > 0:
        untangle_trees(left, right, config.untangle_iterations, descending=config.sort_descending)
    else:
        sort_tree_plain(left, descending=config.sort_descending)
        sort_tree_plain(right, descending=config.sort_descending)

    left_map = external_map(left)
    right_map = external_map(right)
    ordered_ids = sorted(common_ids, key=lambda tip_id: left_map[tip_id].y)
    n_tips = len(ordered_ids)

    height = config.height or min(config.max_height, max(7.5, n_tips / 62))
    fig = plt.figure(figsize=(config.width, height), dpi=config.dpi)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.80, 1.12], wspace=0.0)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_mid = fig.add_subplot(grid[0, 1])
    ax_right = fig.add_subplot(grid[0, 2])

    for ax in (ax_left, ax_mid, ax_right):
        ax.set_facecolor("white")
        ax.axis("off")

    ha_color_func = (
        ha_clade_branch_color(metadata, config, ha_clade_colors)
        if COLOR_HA_BY_CLADE
        else branch_color(config.ha_column, metadata, config.highlight_state, HA_V_COLOR)
    )
    plot_tree(
        left,
        ax_left,
        "left",
        ha_color_func,
        config.branch_width,
        config.tip_size,
        config.show_tip_points,
    )
    plot_tree(
        right,
        ax_right,
        "right",
        branch_color(config.pb2_column, metadata, config.highlight_state, PB2_V_COLOR),
        config.branch_width,
        config.tip_size,
        config.show_tip_points,
    )

    y_min_left, y_max_left = tree_extent(left)
    y_min_right, y_max_right = tree_extent(right)
    y_min = min(y_min_left, y_min_right)
    y_max = max(y_max_left, y_max_right)
    pad = max(2.0, (y_max - y_min) * 0.012)

    if config.flip_ha_vertical:
        ax_left.set_ylim(y_max + pad, y_min - pad)
    else:
        ax_left.set_ylim(y_min - pad, y_max + pad)
    ax_right.set_ylim(y_min - pad, y_max + pad)
    ax_mid.set_ylim(y_min - pad, y_max + pad)
    ax_left.set_xlim(-left.treeHeight * 0.03, left.treeHeight * 1.06)
    ax_right.set_xlim(-right.treeHeight * 1.06, right.treeHeight * 0.03)
    ax_mid.set_xlim(0, 1)
    if config.show_scale_bars:
        draw_scale_bar(ax_left, left, "left", y_min, pad)
        draw_scale_bar(ax_right, right, "right", y_min, pad)

    for tip_id in ordered_ids:
        ha_state = metadata.at[tip_id, config.ha_column]
        pb2_state = metadata.at[tip_id, config.pb2_column]
        ha_y = left_map[tip_id].y
        if config.flip_ha_vertical:
            ha_y = flip_y(ha_y, y_min, y_max)
        ax_mid.plot(
            [0.02, 0.98],
            [ha_y, right_map[tip_id].y],
            color=link_color(ha_state, pb2_state, config.highlight_state),
            alpha=link_alpha(ha_state, pb2_state, config.highlight_state),
            lw=config.link_width,
            solid_capstyle="round",
            zorder=1 if ha_state != config.highlight_state and pb2_state != config.highlight_state else 2,
        )

    ax_left.text(0.0, 1.01, "HA tree", transform=ax_left.transAxes, ha="left", va="bottom", fontsize=13, fontweight="bold")
    ax_right.text(1.0, 1.01, "PB2 tree", transform=ax_right.transAxes, ha="right", va="bottom", fontsize=13, fontweight="bold")
    ax_mid.text(0.5, 1.01, f"{n_tips:,} paired isolates", transform=ax_mid.transAxes, ha="center", va="bottom", fontsize=10, color="#5B6168")

    ha_v = int((metadata.loc[list(common_ids), config.ha_column] == config.highlight_state).sum())
    pb2_v = int((metadata.loc[list(common_ids), config.pb2_column] == config.highlight_state).sum())
    both_v = int(((metadata.loc[list(common_ids), config.ha_column] == config.highlight_state) & (metadata.loc[list(common_ids), config.pb2_column] == config.highlight_state)).sum())
    ha_only_v = int(((metadata.loc[list(common_ids), config.ha_column] == config.highlight_state) & (metadata.loc[list(common_ids), config.pb2_column] != config.highlight_state)).sum())
    pb2_only_v = int(((metadata.loc[list(common_ids), config.ha_column] != config.highlight_state) & (metadata.loc[list(common_ids), config.pb2_column] == config.highlight_state)).sum())
    if config.show_labels:
        draw_labels(ax_left, left, "left", config.label_every, config.label_font_size)
        draw_labels(ax_right, right, "right", config.label_every, config.label_font_size)

    add_legend(fig, config, ha_clade_colors)
    fig.subplots_adjust(left=0.045, right=0.825, top=0.955, bottom=0.045)

    written_files = save_figure_outputs(fig, config.output)
    plt.close(fig)

    return {
        "paired_tips": n_tips,
        "ha_v": ha_v,
        "pb2_v": pb2_v,
        "both_v": both_v,
        "ha_only_v": ha_only_v,
        "pb2_only_v": pb2_only_v,
        "missing_from_ha_tree": missing_left,
        "missing_from_pb2_tree": missing_right,
        "missing_from_metadata": missing_meta,
        "written_files": written_files,
    }


def existing_default(path: str) -> Path:
    return Path(path)


def parse_args(argv: Iterable[str] | None = None) -> PlotConfig:
    parser = argparse.ArgumentParser(
        description="Draw a publication-style HA/PB2 tanglegram, highlighting HA_197=V and PB2_627=V."
    )
    parser.add_argument("--ha-tree", type=existing_default, default=Path("H9N2_HA_mafft.fas.treefile"))
    parser.add_argument("--pb2-tree", type=existing_default, default=Path("H9N2_PB2_mafft.fas.treefile"))
    parser.add_argument("--metadata", type=existing_default, default=Path("metadata_tangle.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("figures/HA197_PB2_627_tanglegram.pdf"))
    parser.add_argument("--id-column", default=ID_COLUMN)
    parser.add_argument("--ha-column", default=HA_STATE_COLUMN)
    parser.add_argument("--pb2-column", default=PB2_STATE_COLUMN)
    parser.add_argument("--ha-clade-column", default=HA_CLADE_COLUMN)
    parser.add_argument("--highlight-state", default=HIGHLIGHT_STATE)
    parser.add_argument("--width", type=float, default=10.8)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--max-height", type=float, default=11.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--tip-size", type=float, default=9.0)
    parser.add_argument("--branch-width", type=float, default=1.25)
    parser.add_argument("--link-width", type=float, default=0.50)
    parser.add_argument("--show-tip-points", action="store_true", default=SHOW_TIP_POINTS)
    parser.add_argument("--no-midpoint-root", action="store_false", dest="midpoint_root", default=MIDPOINT_ROOT)
    parser.add_argument("--ascending", action="store_false", dest="sort_descending", default=SORT_DESCENDING)
    parser.add_argument("--flip-ha-vertical", action="store_true", default=FLIP_HA_VERTICAL)
    parser.add_argument("--show-labels", action="store_true")
    parser.add_argument("--label-every", type=int, default=1)
    parser.add_argument("--label-font-size", type=float, default=4.0)
    parser.add_argument("--untangle-iterations", type=int, default=0)
    parser.add_argument("--no-scale-bars", action="store_false", dest="show_scale_bars", default=SHOW_SCALE_BARS)
    args = parser.parse_args(argv)

    return PlotConfig(
        ha_tree=args.ha_tree,
        pb2_tree=args.pb2_tree,
        metadata=args.metadata,
        output=args.output,
        id_column=args.id_column,
        ha_column=args.ha_column,
        pb2_column=args.pb2_column,
        ha_clade_column=args.ha_clade_column,
        highlight_state=normalize_state(args.highlight_state) or "V",
        width=args.width,
        height=args.height,
        max_height=args.max_height,
        dpi=args.dpi,
        tip_size=args.tip_size,
        branch_width=args.branch_width,
        show_tip_points=args.show_tip_points,
        midpoint_root=args.midpoint_root,
        sort_descending=args.sort_descending,
        untangle_by_partner=UNTANGLE_BY_PARTNER,
        flip_ha_vertical=args.flip_ha_vertical,
        link_width=args.link_width,
        show_labels=args.show_labels,
        label_every=args.label_every,
        label_font_size=args.label_font_size,
        untangle_iterations=args.untangle_iterations,
        show_scale_bars=args.show_scale_bars,
    )


def validate_inputs(config: PlotConfig) -> None:
    for label, path in [("HA tree", config.ha_tree), ("PB2 tree", config.pb2_tree), ("metadata", config.metadata)]:
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")


def main(argv: Iterable[str] | None = None) -> int:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    config = parse_args(argv)
    try:
        validate_inputs(config)
        stats = draw_tanglegram(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for written_file in stats["written_files"]:
        print(f"Wrote {written_file}")
    print(
        f"Paired tips: {stats['paired_tips']} | "
        f"{config.ha_column} {config.highlight_state}: {stats['ha_v']} | "
        f"{config.pb2_column} {config.highlight_state}: {stats['pb2_v']} | "
        f"both: {stats['both_v']} | "
        f"HA only: {stats['ha_only_v']} | PB2 only: {stats['pb2_only_v']}"
    )
    if any(stats[key] for key in ("missing_from_ha_tree", "missing_from_pb2_tree", "missing_from_metadata")):
        print(
            "Dropped unmatched tips - missing from HA tree: {missing_from_ha_tree}, "
            "missing from PB2 tree: {missing_from_pb2_tree}, missing metadata: {missing_from_metadata}".format(
                **stats
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
