"""
Neural Network Training Animation
==================================
Animates a NetworkX graph of your SimpleMMLModel across training epochs.

Architecture:  TF inputs -> (in_weights) -> hidden nodes -> (out_weights) -> prediction
Edge colours:  red = strong positive, blue = strong negative, grey = near zero
Edge thickness: scales with |weight|

Usage:
    python nn_training_animation.py \
        --in_weights  data/in_weights_OPALIN.csv \
        --out_weights data/out_weights_OPALIN.csv \
        --top_n 30 \
        --output opalin_training.gif

Requirements:
    pip install matplotlib networkx pandas numpy
"""

import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from pathlib import Path


# ── Colour helpers ─────────────────────────────────────────────────────────────

def weight_to_colour(w: float, vmax: float = 1.0) -> str:
    """Map a scalar weight to a hex colour string.
    Positive → red (#E24B4A), negative → blue (#378ADD), zero → light grey.
    NetworkX requires hex strings or named colours — RGBA tuples are silently ignored.
    """
    t = np.clip(w / (vmax + 1e-8), -1, 1)
    if t >= 0:
        # interpolate grey (#B5B5B5) → red (#E24B4A)
        r = int(round(0xB5 + (0xE2 - 0xB5) * t))
        g = int(round(0xB5 + (0x4B - 0xB5) * t))
        b = int(round(0xB5 + (0x4A - 0xB5) * t))
    else:
        # interpolate grey (#B5B5B5) → blue (#378ADD)
        s = -t
        r = int(round(0xB5 + (0x37 - 0xB5) * s))
        g = int(round(0xB5 + (0x8A - 0xB5) * s))
        b = int(round(0xB5 + (0xDD - 0xB5) * s))
    return f"#{r:02X}{g:02X}{b:02X}"


def weights_to_colours(weights: np.ndarray, vmax: float) -> list:
    return [weight_to_colour(w, vmax) for w in weights]


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(tf_names: list[str], top_idx: np.ndarray,
                in_w: np.ndarray, out_w: np.ndarray) -> nx.DiGraph:
    """Build a DiGraph for the top-N TFs at one epoch."""
    G = nx.DiGraph()

    # Node layers
    G.add_node("OUT", layer=2, label="OPALIN")
    for i in top_idx:
        name = tf_names[i]
        G.add_node(f"H_{i}", layer=1, label="")          # hidden node
        G.add_node(f"TF_{i}", layer=0, label=name)        # input node
        G.add_edge(f"TF_{i}", f"H_{i}",  weight=in_w[i])
        G.add_edge(f"H_{i}", "OUT",       weight=out_w[i])

    return G


def multipartite_pos(G: nx.DiGraph, layer_attr: str = "layer",
                     v_spread: float = 1.0) -> dict:
    """Assign (x, y) positions using layer attribute for x, evenly spaced y."""
    layers: dict[int, list] = {}
    for node, data in G.nodes(data=True):
        l = data.get(layer_attr, 0)
        layers.setdefault(l, []).append(node)

    pos = {}
    x_map = {0: 0.0, 1: 1.0, 2: 2.0}
    for l, nodes in layers.items():
        n = len(nodes)
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * v_spread
        for node, y in zip(nodes, ys):
            pos[node] = (x_map.get(l, l), y)
    return pos


# ── Main animation function ────────────────────────────────────────────────────

def animate(in_weights_path: str, out_weights_path: str,
            top_n: int = 30, output: str = "training_animation.gif",
            fps: int = 4, dpi: int = 120) -> None:

    # ── Load data ──────────────────────────────────────────────────────────────
    in_df  = pd.read_csv(in_weights_path,  index_col=0)
    out_df = pd.read_csv(out_weights_path, index_col=0)

    # Columns are epoch indices as strings; coerce to int and sort
    in_df.columns  = in_df.columns.astype(int)
    out_df.columns = out_df.columns.astype(int)
    epochs = sorted(in_df.columns.tolist())

    # Row index = TF names (set when saving from model weight rows)
    tf_names = list(in_df.index.astype(str))
    n_epochs = len(epochs)

    # Global colour scale: max |weight| across all epochs
    vmax_in  = in_df.abs().values.max()
    vmax_out = out_df.abs().values.max()
    vmax     = max(vmax_in, vmax_out)

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [3, 1]},
        facecolor="#0f0f0f"
    )
    ax_graph, ax_bar = axes
    for ax in axes:
        ax.set_facecolor("#0f0f0f")

    fig.suptitle("", color="white", fontsize=14, y=0.98)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color="#E24B4A", linewidth=2, label="Positive weight"),
        Line2D([0], [0], color="#378ADD", linewidth=2, label="Negative weight"),
        Line2D([0], [0], color="#aaaaaa", linewidth=1, label="Near zero"),
    ]
    ax_graph.legend(handles=legend_elements, loc="upper left",
                    framealpha=0.2, labelcolor="white", fontsize=8)

    # ── Animation update function ──────────────────────────────────────────────
    def update(frame_idx: int):
        epoch = epochs[frame_idx]
        in_w  = in_df[epoch].values.flatten()
        out_w = out_df[epoch].values.flatten()

        # Combined importance score: product of |in| and |out|
        importance = np.abs(in_w) * np.abs(out_w)
        top_idx    = np.argsort(importance)[-top_n:][::-1]

        # ── Graph panel ───────────────────────────────────────────────────────
        ax_graph.clear()
        ax_graph.set_facecolor("#0f0f0f")
        ax_graph.set_title(
            f"Epoch {epoch} / {epochs[-1]}   —   Top {top_n} TFs by |in_w × out_w|",
            color="white", fontsize=10, pad=8
        )
        ax_graph.axis("off")

        G   = build_graph(tf_names, top_idx, in_w, out_w)
        pos = multipartite_pos(G, v_spread=0.9)

        # Separate input→hidden and hidden→output edges
        in_edges  = [(u, v) for u, v, d in G.edges(data=True) if v.startswith("H_")]
        out_edges = [(u, v) for u, v, d in G.edges(data=True) if v == "OUT"]

        def edge_style(edges, weight_arr, vmax_val):
            ws      = [G[u][v]["weight"] for u, v in edges]
            colours = weights_to_colours(ws, vmax_val)
            widths  = [max(0.4, abs(w) / vmax_val * 3) for w in ws]
            return colours, widths

        in_cols,  in_wids  = edge_style(in_edges,  in_w,  vmax_in)
        out_cols, out_wids = edge_style(out_edges, out_w, vmax_out)

        nx.draw_networkx_edges(G, pos, edgelist=in_edges,
                               edge_color=in_cols,  width=in_wids,
                               ax=ax_graph, arrows=False, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edgelist=out_edges,
                               edge_color=out_cols, width=out_wids,
                               arrows=True, arrowstyle="-|>", arrowsize=10,
                               ax=ax_graph, alpha=0.9)

        # Node colours by layer
        node_colours = []
        node_sizes   = []
        for node in G.nodes():
            if node.startswith("TF_"):
                node_colours.append("#2a6496")
                node_sizes.append(120)
            elif node.startswith("H_"):
                node_colours.append("#1a6b3a")
                node_sizes.append(80)
            else:  # OUT
                node_colours.append("#8B2020")
                node_sizes.append(300)

        nx.draw_networkx_nodes(G, pos, node_color=node_colours,
                               node_size=node_sizes, ax=ax_graph)

        # Labels: only TF inputs and output node
        labels = {n: d["label"] for n, d in G.nodes(data=True) if d["label"]}
        nx.draw_networkx_labels(G, pos, labels=labels,
                                font_size=6, font_color="white", ax=ax_graph)

        # ── Bar chart panel: top TF importances ───────────────────────────────
        ax_bar.clear()
        ax_bar.set_facecolor("#0f0f0f")
        ax_bar.set_title("TF importance\n|in_w × out_w|",
                          color="white", fontsize=9)

        top_names = [tf_names[i] for i in top_idx]
        top_imp   = importance[top_idx]
        bar_cols  = [weight_to_colour(in_w[i], vmax_in if vmax_in > 0 else 1.0) for i in top_idx]

        bars = ax_bar.barh(range(top_n), top_imp[::-1], color=bar_cols[::-1])
        ax_bar.set_yticks(range(top_n))
        ax_bar.set_yticklabels(top_names[::-1], fontsize=6, color="white")
        ax_bar.tick_params(colors="white")
        ax_bar.spines[:].set_color("#333333")
        ax_bar.xaxis.label.set_color("white")
        ax_bar.set_xlabel("|in_w × out_w|", color="white", fontsize=8)

        fig.suptitle(
            f"SimpleMMLModel — OPALIN   Epoch {epoch}/{epochs[-1]}",
            color="white", fontsize=13
        )

    # ── Build & save ──────────────────────────────────────────────────────────
    ani = animation.FuncAnimation(
        fig, update, frames=n_epochs, interval=1000 // fps, repeat=True
    )

    out_path = Path(output)
    if out_path.suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)

    print(f"Saving {n_epochs} frames to {output} …")
    ani.save(str(out_path), writer=writer, dpi=dpi)
    print(f"Saved → {output}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate MML model training")
    parser.add_argument("--in_weights",  required=True, help="Path to in_weights CSV")
    parser.add_argument("--out_weights", required=True, help="Path to out_weights CSV")
    parser.add_argument("--top_n",  type=int, default=30,
                        help="Number of top TFs to display (default: 30)")
    parser.add_argument("--output", default="opalin_training.gif",
                        help="Output file (.gif or .mp4)")
    parser.add_argument("--fps",    type=int, default=4,
                        help="Frames per second (default: 4)")
    parser.add_argument("--dpi",    type=int, default=120,
                        help="Output resolution (default: 120)")
    args = parser.parse_args()

    animate(
        in_weights_path  = args.in_weights,
        out_weights_path = args.out_weights,
        top_n            = args.top_n,
        output           = args.output,
        fps              = args.fps,
        dpi              = args.dpi,
    )
