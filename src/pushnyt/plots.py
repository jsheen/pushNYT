"""Plots saved alongside the training outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display inside the container
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

# Feature groups from features.py, colored in fixed order
GROUPS = {
    "Similarity": ("#2a78d6", {"sim_text", "sim_entity", "subcat_affinity"}),
    "Slot engagement": ("#eb6834", {"slot", "slot_prior_impr", "slot_prior_clicks",
                                    "slot_engaged_rate", "slot_click_share"}),
    "Complementarity": ("#1baf7a", {"complement_score"}),
    "Push fatigue": ("#eda100", {"slot_skipped_72h", "slot_skip_rate_72h", "impr_last_24h",
                                 "hours_since_last_impr", "hours_since_last_slot_impr"}),
    "Embedding retrieval": ("#e87ba4", {"emb_sim_mean", "emb_sim_max", "emb_sim_top5", "emb_n_close",
                                        "emb_retrieval_rank"}),
    "Article recency": ("#8a5cd1", {"article_age_hours"}),
    "User depth": ("#7a6f5f", {"user_history_len", "user_prior_impr"}),
}
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plot_importance(importance: pd.Series, path: Path) -> None:
    """Horizontal bar chart of XGBoost gain per feature, colored by feature group."""
    color = {f: c for c, feats in GROUPS.values() for f in feats}
    imp = importance.sort_values()

    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.3 * len(imp)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(imp))
    ax.barh(y, imp.to_numpy(), height=0.6, color=[color.get(f, TEXT_MUTED) for f in imp.index])
    for i, v in enumerate(imp.to_numpy()):
        ax.text(v + imp.max() * 0.012, i, f"{v:.1f}", va="center", color=TEXT_MUTED, fontsize=9)

    ax.set_yticks(list(y), imp.index, color=TEXT, fontfamily="monospace", fontsize=9)
    ax.set_xlabel("Average gain per split", color=TEXT_MUTED)
    ax.set_xlim(0, imp.max() * 1.12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=TEXT_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("pushNYT XGBoost click model: feature importance", loc="left",
                 color=TEXT, fontsize=13, fontweight="bold", pad=28)
    ax.text(0, 1.035, "Gain = average loss reduction each time the feature is used in a split",
            transform=ax.transAxes, color=TEXT_MUTED, fontsize=9)
    ax.legend(handles=[Patch(color=c, label=g) for g, (c, _) in GROUPS.items()],
              loc="lower right", frameon=False, labelcolor=TEXT)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
