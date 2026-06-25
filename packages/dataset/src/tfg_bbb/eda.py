from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FEATURE_COLS = [
    "mw",
    "hydrophobic_ratio_pct",
    "pi",
    "net_charge_ph7",
    "total_charge",
    "mean_hydrophobicity",
    "aliphatic_index",
    "boman_index",
    "aromaticity",
    "instability_index",
    "gravy",
]

COMPARISON_FEATURES = ["length", "gravy", "net_charge_ph7", "boman_index", "instability_index"]


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path.resolve())


def dataset_overview_table(df: pd.DataFrame, *, name: str = "dataset") -> pd.DataFrame:
    pos = int((df["bbb_label"] == 1).sum())
    neg = int((df["bbb_label"] == 0).sum())
    return pd.DataFrame(
        [
            {
                "dataset": name,
                "rows": len(df),
                "bbb_positive": pos,
                "bbb_negative": neg,
                "positive_ratio": round(pos / len(df), 4) if len(df) else 0.0,
                "length_median": float(df["length"].median()) if len(df) else float("nan"),
                "length_min": int(df["length"].min()) if len(df) else 0,
                "length_max": int(df["length"].max()) if len(df) else 0,
                "unique_sequences": int(df["sequence"].nunique()) if "sequence" in df.columns else len(df),
                "unique_clusters": int(df["cluster_id"].nunique()) if "cluster_id" in df.columns else float("nan"),
            }
        ]
    )


def fold_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, group in df.groupby("fold_id", sort=True):
        rows.append(
            {
                "fold_id": int(fold_id),
                "rows": len(group),
                "bbb_positive": int((group["bbb_label"] == 1).sum()),
                "bbb_negative": int((group["bbb_label"] == 0).sum()),
                "positive_ratio": round(float(group["bbb_label"].mean()), 4),
                "unique_clusters": int(group["cluster_id"].nunique()),
                "external_test_rows": int(group.get("external_test", pd.Series([0] * len(group))).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def cluster_leakage_table(df: pd.DataFrame) -> pd.DataFrame:
    """Clusters assigned to more than one CV fold (should be empty on non-holdout rows)."""
    cv = df[(df.get("external_test", 0) == 0) & (df["fold_id"] >= 0)].copy()
    if cv.empty:
        return pd.DataFrame(columns=["cluster_id", "n_folds", "fold_ids"])

    grouped = (
        cv.groupby("cluster_id")["fold_id"]
        .agg(n_folds="nunique", fold_ids=lambda s: sorted(set(int(x) for x in s)))
        .reset_index()
    )
    leaked = grouped[grouped["n_folds"] > 1].sort_values("n_folds", ascending=False)
    return leaked.reset_index(drop=True)


def _plot_class_balance(df: pd.DataFrame, out_dir: Path, *, title: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="bbb_label", hue="split", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("BBB label")
    return _save(fig, out_dir / filename)


def _plot_length_distribution(df: pd.DataFrame, out_dir: Path, *, title: str, filename: str) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="length", hue="bbb_label", kde=True, bins=25, element="step", ax=ax)
    ax.set_title(title)
    return _save(fig, out_dir / filename)


def _plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path, *, title: str, filename: str) -> str:
    corr = df[FEATURE_COLS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
    ax.set_title(title)
    return _save(fig, out_dir / filename)


def _plot_fold_class_balance(df: pd.DataFrame, out_dir: Path) -> str:
    cv = df[(df.get("external_test", 0) == 0) & (df["fold_id"] >= 0)].copy()
    cv["fold_id"] = cv["fold_id"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=cv, x="fold_id", hue="bbb_label", ax=ax)
    ax.set_title("CV fold class balance (non-holdout)")
    ax.set_xlabel("fold_id (0 = validation fold)")
    return _save(fig, out_dir / "fold_class_balance.png")


def _plot_fold_sizes(df: pd.DataFrame, out_dir: Path) -> str:
    sizes = fold_overview_table(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=sizes, x="fold_id", y="rows", hue="fold_id", dodge=False, ax=ax, legend=False)
    ax.set_title("Rows per fold_id")
    return _save(fig, out_dir / "fold_sizes.png")


def _plot_holdout_overview(df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="split", hue="external_test", ax=ax)
    ax.set_title("Source split vs external holdout flag")
    return _save(fig, out_dir / "holdout_overview.png")


def _plot_source_composition(df: pd.DataFrame, out_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="source_db", hue="bbb_label", ax=ax)
    ax.set_title("Source database composition")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, out_dir / "source_composition.png")


def run_gold_eda(df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    """EDA for the curated gold dataset, including CV fold diagnostics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        _plot_class_balance(df, out_dir, title="BBB class balance by split", filename="class_balance.png"),
        _plot_length_distribution(df, out_dir, title="Length distribution", filename="length_distribution.png"),
        _plot_correlation_heatmap(df, out_dir, title="Spearman correlation (features)", filename="correlation_heatmap.png"),
        _plot_fold_class_balance(df, out_dir),
        _plot_fold_sizes(df, out_dir),
        _plot_holdout_overview(df, out_dir),
        _plot_source_composition(df, out_dir),
    ]
    overview = dataset_overview_table(df, name="gold")
    folds = fold_overview_table(df)
    leakage = cluster_leakage_table(df)
    overview.to_csv(out_dir / "overview.csv", index=False)
    folds.to_csv(out_dir / "fold_overview.csv", index=False)
    leakage.to_csv(out_dir / "cluster_leakage.csv", index=False)
    return {
        "overview": overview,
        "fold_table": folds,
        "cluster_leakage": leakage,
        "figures": figures,
        "rows": float(len(df)),
        "bbb_positive_ratio": float(df["bbb_label"].mean()),
        "length_median": float(df["length"].median()),
    }


def run_augmentation_eda(
    gold_df: pd.DataFrame,
    combined_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, object]:
    """Compare gold vs augmented dataset: counts, folds, lengths, and key features."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    overview = pd.concat(
        [
            dataset_overview_table(gold_df, name="gold"),
            dataset_overview_table(aug_df, name="augmented_extra"),
            dataset_overview_table(combined_df, name="gold_plus_augmented"),
        ],
        ignore_index=True,
    )

    train_gold = gold_df[(gold_df.get("external_test", 0) == 0) & (gold_df["fold_id"] != 0)]
    train_combined = combined_df[(combined_df.get("external_test", 0) == 0) & (combined_df["fold_id"] != 0)]
    train_overview = pd.concat(
        [
            dataset_overview_table(train_gold, name="gold_train_eligible"),
            dataset_overview_table(train_combined, name="combined_train_eligible"),
        ],
        ignore_index=True,
    )

    fold_compare = fold_overview_table(gold_df)[["fold_id", "rows", "bbb_positive", "positive_ratio"]].rename(
        columns={
            "rows": "gold_rows",
            "bbb_positive": "gold_pos",
            "positive_ratio": "gold_pos_ratio",
        }
    )
    combined_folds = fold_overview_table(combined_df)[["fold_id", "rows", "bbb_positive", "positive_ratio"]].rename(
        columns={
            "rows": "combined_rows",
            "bbb_positive": "combined_pos",
            "positive_ratio": "combined_pos_ratio",
        }
    )
    fold_compare = fold_compare.merge(combined_folds, on="fold_id", how="outer")
    fold_compare["delta_rows"] = fold_compare["combined_rows"] - fold_compare["gold_rows"]

    # Row counts pre/post
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = overview.set_index("dataset")["rows"]
    counts.plot(kind="bar", ax=ax, color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title("Dataset size before / after augmentation")
    ax.set_ylabel("rows")
    ax.tick_params(axis="x", rotation=15)
    figures.append(_save(fig, out_dir / "row_counts_pre_post.png"))

    # Class balance on train-eligible rows
    fig, ax = plt.subplots(figsize=(6, 4))
    balance = train_overview.set_index("dataset")[["bbb_positive", "bbb_negative"]]
    balance.plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"])
    ax.set_title("Class balance (train-eligible: fold_id != 0)")
    ax.set_ylabel("peptides")
    ax.tick_params(axis="x", rotation=10)
    figures.append(_save(fig, out_dir / "class_balance_train_eligible.png"))

    # Length distributions
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(data=gold_df, x="length", label="gold", fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(data=aug_df, x="length", label="augmented", fill=True, alpha=0.35, ax=ax)
    ax.set_title("Length distribution: gold vs augmented")
    ax.legend()
    figures.append(_save(fig, out_dir / "length_pre_post.png"))

    # Feature shift on augmented rows
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.ravel()
    for ax, col in zip(axes, COMPARISON_FEATURES, strict=False):
        if col not in gold_df.columns:
            ax.axis("off")
            continue
        sns.kdeplot(data=train_gold, x=col, label="gold train", fill=True, alpha=0.3, ax=ax)
        sns.kdeplot(data=aug_df, x=col, label="augmented", fill=True, alpha=0.3, ax=ax)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Feature distributions: gold train vs augmented", y=1.02)
    fig.tight_layout()
    figures.append(_save(fig, out_dir / "feature_shift_pre_post.png"))

    # Augmentations per parent
    if not aug_df.empty and "parent_peptide_id" in aug_df.columns:
        per_parent = aug_df.groupby("parent_peptide_id").size()
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(per_parent, bins=range(1, int(per_parent.max()) + 2), ax=ax)
        ax.set_title("Augmented peptides generated per parent")
        ax.set_xlabel("n augmented / parent")
        figures.append(_save(fig, out_dir / "aug_per_parent.png"))

        parent_lengths = gold_df.set_index("peptide_id")["length"]
        aug_with_parent = aug_df.copy()
        aug_with_parent["parent_length"] = aug_with_parent["parent_peptide_id"].map(parent_lengths)
        aug_with_parent["length_delta"] = (aug_with_parent["length"] - aug_with_parent["parent_length"]).abs()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(aug_with_parent["length_delta"], bins=range(0, 6), ax=ax)
        ax.set_title("Absolute length change vs parent")
        ax.set_xlabel("|len_aug - len_parent|")
        figures.append(_save(fig, out_dir / "length_delta_aug.png"))

    # Fold row deltas
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=fold_compare, x="fold_id", y="delta_rows", ax=ax, color="#8172b3")
    ax.set_title("Added rows per fold after augmentation")
    ax.axhline(0, color="black", linewidth=0.8)
    figures.append(_save(fig, out_dir / "fold_row_delta.png"))

    overview.to_csv(out_dir / "overview_pre_post.csv", index=False)
    train_overview.to_csv(out_dir / "train_eligible_overview.csv", index=False)
    fold_compare.to_csv(out_dir / "fold_compare.csv", index=False)

    return {
        "overview": overview,
        "train_overview": train_overview,
        "fold_compare": fold_compare,
        "figures": figures,
    }


def run_eda(df: pd.DataFrame, out_dir: Path) -> dict[str, float]:
    """Backward-compatible wrapper around `run_gold_eda`."""
    result = run_gold_eda(df, out_dir)
    return {
        "rows": float(result["rows"]),
        "bbb_positive_ratio": float(result["bbb_positive_ratio"]),
        "length_median": float(result["length_median"]),
    }


def show_eda(result: dict[str, object], title: str, *, max_figures: int = 8) -> None:
    """Render EDA tables and saved figures in a Jupyter notebook."""
    from IPython.display import Image, display

    print(f"\n=== {title} ===")
    for key in ("overview", "fold_table", "cluster_leakage", "train_overview", "fold_compare"):
        value = result.get(key)
        if isinstance(value, pd.DataFrame) and not value.empty:
            display(value)
    leakage = result.get("cluster_leakage")
    if isinstance(leakage, pd.DataFrame) and leakage.empty:
        print("Cluster leakage across folds: none detected (good).")
    for fig_path in result.get("figures", [])[:max_figures]:
        display(Image(filename=fig_path, width=700))
