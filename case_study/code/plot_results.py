import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
from annotations import operon_membership

STRAINS = ["MG1655", "TW09308", "TW11588", "IAI1"]
GENES = ["flhD", "flhC", "fliA", "fliD", "cheZ", "flgK", "flgL", "motB"]


def export(name):
    content = json.loads((DATA / "regulondb" / (name + ".json")).read_text())["content"]
    frame = pd.read_csv(io.StringIO(content), sep="\t", dtype=str).fillna("")
    frame.columns = [column.split(")", 1)[-1].strip() for column in frame.columns]
    return frame.map(str.strip)


def run():
    products = export("GeneProductSet")
    id_to_gene = dict(zip(products.geneId, products.bnumber))
    name_to_gene = dict(zip(products.geneName, products.bnumber))
    tf_names = set(export("TFSet").name)
    edge_rows = []
    for row in export("NetworkRegulatorGene").itertuples():
        if row.confidenceLevel in ["C", "S"] and row.regulatorName in tf_names:
            edge_rows.append({"regulator": row.regulatorName, "gene_id": id_to_gene.get(row.regulatedGeneId)})
    for row in export("NetworkSigmaGene").itertuples():
        if row.confidenceLevel in ["C", "S"]:
            edge_rows.append({"regulator": row.sigmaName, "gene_id": name_to_gene.get(row.regulatedGeneName)})
    edges = pd.DataFrame(edge_rows).dropna().drop_duplicates()
    reference = pd.read_csv(OUT / "DE_MG1655_TW09308.csv").set_index("gene_id")
    operons = operon_membership(products.bnumber[products.bnumber != ""].unique())
    ranking_rows = []
    for regulator, group in edges.groupby("regulator"):
        subset = reference.loc[reference.index.intersection(group.gene_id)]
        target_operons = pd.Series(subset.index.map(lambda gene: operons.get(gene, gene)), index=subset.index)
        if len(subset) < 10 or target_operons.nunique() < 3:
            continue
        opposite = subset.loc[subset.opposite_DE]
        ranking_rows.append({"regulator": regulator, "tested_targets": len(subset), "target_operons": target_operons.nunique(),
                             "opposite_DE": int(subset.opposite_DE.sum()), "opposite_fraction": subset.opposite_DE.mean(),
                             "operons_with_opposite_gene": target_operons.loc[opposite.index].nunique(),
                             "opposite_names": ";".join(opposite.gene_name)})
    ranking = pd.DataFrame(ranking_rows).sort_values(["opposite_fraction", "tested_targets"], ascending=False)
    ranking["rank"] = ranking.opposite_fraction.rank(method="min", ascending=False).astype(int)
    ranking["operon_fraction"] = ranking.operons_with_opposite_gene / ranking.target_operons
    ranking["operon_rank"] = ranking.operon_fraction.rank(method="min", ascending=False).astype(int)
    ranking.to_csv(OUT / "reference_regulon_screen.csv", index=False)

    effects = pd.read_csv(OUT / "selected_pairwise_results.csv")
    display_rows = []
    reference_effects = effects.loc[effects.relative_strain == "TW09308"]
    for row in reference_effects.itertuples():
        display_rows.append({"strain": "MG1655", "gene": row.gene_name, "log2FC": row.MG1655_log2FoldChange,
                             "lfcSE": row.MG1655_lfcSE, "padj": row.MG1655_padj})
    for row in effects.itertuples():
        display_rows.append({"strain": row.relative_strain, "gene": row.gene_name, "log2FC": row.relative_log2FoldChange,
                             "lfcSE": row.relative_lfcSE, "padj": row.relative_padj})
    display = pd.DataFrame(display_rows)
    display.to_csv(OUT / "displayed_effects.csv", index=False)
    ratio_changes = pd.read_csv(OUT / "RNA_ratio_changes.csv")
    ratio_points = pd.read_csv(OUT / "replicate_RNA_ratios.csv")
    strong_points = ratio_points.loc[ratio_points.ratio.isin(["fliD/fliA", "cheZ/fliA"])].groupby(["strain", "sample", "state"]).log2_raw_count_ratio.mean().reset_index()
    sensitivity_rows = []
    for strain, group in strong_points.groupby("strain"):
        change = group.loc[group.state == "starvation", "log2_raw_count_ratio"].mean() - group.loc[group.state == "chemostat", "log2_raw_count_ratio"].mean()
        sensitivity_rows.append({"strain": strain, "log2_change": change, "fold_change": 2**change,
                                 "definition": "Geometric mean of fliD and cheZ divided by fliA; excludes flgKL's weak-labelled sigma edges."})
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "strong_edge_output_sensitivity.csv", index=False)
    combined_audit = pd.read_csv(OUT / "selected_sequence_audits.csv")
    combined_audit.to_csv(OUT / "all_selected_sequence_audits.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    figure, axis = plt.subplots(figsize=(10, 3.5), layout="constrained")
    positions = {"FlhDC": (0.14, 0.45), "FliA": (0.46, 0.45), "fliD": (0.85, 0.74),
                 "cheZ": (0.85, 0.43), "flgKL": (0.85, 0.12)}
    arrows = [((0.24, 0.45), (0.36, 0.45), False, 0),
              ((0.19, 0.57), (0.82, 0.82), False, -0.1),
              ((0.55, 0.51), (0.81, 0.74), False, 0),
              ((0.55, 0.44), (0.81, 0.43), False, 0),
              ((0.55, 0.38), (0.785, 0.15), True, 0)]
    for source, target, weak, curvature in arrows:
        axis.annotate("", xy=target, xytext=source,
                      arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 1.4,
                                  "linestyle": "--" if weak else "-", "shrinkA": 0, "shrinkB": 0,
                                  "connectionstyle": f"arc3,rad={curvature}"})
    labels = {"FlhDC": "FlhDC complex\nRNAs: flhD, flhC", "FliA": "FliA / sigma28\nRNA: fliA",
              "fliD": "fliD", "cheZ": "cheZ", "flgKL": "flgK, flgL"}
    for node, position in positions.items():
        axis.text(*position, labels[node], ha="center", va="center", fontsize=11,
                  bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#777777"})
    axis.set(xlim=(0, 1), ylim=(-0.04, 1.1))
    axis.axis("off")
    figure.suptitle("Selected MG1655 reference links used to organize the RNA comparisons")
    figure.supxlabel("All arrows denote positive regulation. Solid: confirmed/strong RegulonDB edges; dashed: weak-labelled sigma annotation.\nAutoregulation and other inputs are omitted. Edges and protein activities were not measured in the relatives.", fontsize=9)
    figure.savefig(FIG / "00_reference_hierarchy.png")
    figure.savefig(FIG / "00_reference_hierarchy.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained", gridspec_kw={"width_ratios": [1.05, 1]})
    highlight = reference.FliA_target
    axes[0].scatter(reference.loc[~highlight, "MG1655_log2FoldChange"], reference.loc[~highlight, "relative_log2FoldChange"],
                    s=8, color="#bcbcbc", alpha=0.55, label="Other tested genes")
    axes[0].scatter(reference.loc[highlight, "MG1655_log2FoldChange"], reference.loc[highlight, "relative_log2FoldChange"],
                    s=30, color="#2166ac", label="Annotated FliA targets", zorder=3)
    extent = 1 + max(reference.MG1655_log2FoldChange.abs().max(), reference.relative_log2FoldChange.abs().max())
    axes[0].plot([-extent, extent], [-extent, extent], color="gray", linewidth=0.7, linestyle="--")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(0, color="gray", linewidth=0.5)
    for name in ["flhD", "flhC", "fliA"]:
        row = reference.loc[reference.gene_name == name].iloc[0]
        offset = (6, -13) if name == "flhD" else (6, 6)
        axes[0].annotate(name, (row.MG1655_log2FoldChange, row.relative_log2FoldChange), xytext=offset, textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="MG1655 log2(starvation / chemostat)", ylabel="TW09308 log2(starvation / chemostat)",
                title=f"Gene responses with reference-network annotations\n{len(reference):,} tested ortholog pairs")
    axes[0].legend(fontsize=8, loc="lower right")
    shown = ranking.head(10).iloc[::-1]
    axes[1].barh(np.arange(len(shown)), 100 * shown.opposite_fraction,
                 color=["#2166ac" if name == "sigma28" else "#999999" for name in shown.regulator])
    axes[1].set_yticks(np.arange(len(shown)), ["FliA / sigma28" if name == "sigma28" else name for name in shown.regulator])
    for position, row in enumerate(shown.itertuples()):
        axes[1].text(100 * row.opposite_fraction + 0.8, position, f"{row.opposite_DE}/{row.tested_targets}", va="center", fontsize=9)
    axes[1].set(xlabel="Targets with substantial opposite DE responses (%)", ylabel="Reference regulator",
                xlim=(0, 63), title=f"Retrospective screen of {len(ranking)} eligible target sets\n≥10 tested genes; ≥3 annotated operons")
    figure.suptitle("1 | Genome-wide context and reference target sets")
    figure.supxlabel("GSE77325; four-hour starvation versus chemostat. Descriptive ranks; overlapping regulons and operon dependence preclude independent-gene inference.", fontsize=8)
    figure.savefig(FIG / "01_network_localization.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(13, 5.2), layout="constrained", sharey=True, sharex=True)
    for axis, strain in zip(axes, STRAINS):
        subset = display.loc[display.strain == strain].set_index("gene")
        for position, gene in enumerate(GENES):
            if gene not in subset.index:
                axis.text(0, position, "Excluded by quality/count filters", ha="center", va="center", fontsize=7)
                continue
            row = subset.loc[gene]
            axis.errorbar(row.log2FC, position, xerr=1.96 * row.lfcSE, fmt="o", color="#2166ac", capsize=2,
                          markerfacecolor="#2166ac" if row.padj < 0.05 else "white", markersize=5)
        axis.axvline(0, color="gray", linewidth=0.6)
        axis.set(title=strain, xlabel="log2(starvation / chemostat)", xlim=(-4.5, 3.4))
        axis.set_yticks(range(len(GENES)), GENES)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Selected genes in the reference hierarchy")
    figure.suptitle("2 | Gene responses to the same transition across four isolates")
    figure.supxlabel("Points: unshrunk estimates; bars: nominal 95% Wald intervals; filled: within-strain FDR <0.05. Two biological replicates per cell. Inspect counts before interpreting weak responses.", fontsize=8)
    figure.savefig(FIG / "02_gene_responses.png")
    plt.close(figure)

    ratios = ["fliA/flhD", "fliA/flhC", "selected_outputs/fliA"]
    titles = ["fliA relative to flhD RNA", "fliA relative to flhC RNA", "Selected output RNAs relative to fliA"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 5.4), layout="constrained")
    for axis, ratio, title in zip(axes, ratios, titles):
        subset = ratio_points.loc[ratio_points.ratio == ratio]
        for position, strain in enumerate(STRAINS):
            strain_points = subset.loc[subset.strain == strain]
            baseline = strain_points.loc[strain_points.state == "chemostat", "log2_raw_count_ratio"].mean()
            for state, offset, color, marker in [("chemostat", -0.12, "#777777", "o"), ("starvation", 0.12, "#2166ac", "^")]:
                values = strain_points.loc[strain_points.state == state, "log2_raw_count_ratio"].to_numpy() - baseline
                axis.scatter(position + offset + np.array([-0.025, 0.025]), values, color=color, marker=marker,
                             label=state if position == 0 else None, s=38, zorder=3)
                axis.plot([position + offset - 0.055, position + offset + 0.055], [values.mean()] * 2, color=color, linewidth=1)
        axis.axhline(0, color="gray", linewidth=0.6)
        axis.set_xticks(range(4), STRAINS, rotation=25)
        axis.set(title=title, xlabel="Isolate", ylabel="log2 RNA ratio minus its chemostat mean")
    axes[0].legend(fontsize=8)
    figure.suptitle("3 | RNA-ratio changes at successive positions in the reference hierarchy")
    figure.supxlabel("Original counts; no pseudocount or size-factor normalization. Every replicate is shown; no pairing assumed. Output summary weights flgKL, fliD and cheZ operons equally.", fontsize=8)
    figure.savefig(FIG / "03_normalization_free_RNA_ratios.png")
    plt.close(figure)

    summary = {
        "reference_screen_regulons": len(ranking),
        "FliA_screen": ranking.loc[ranking.regulator == "sigma28"].iloc[0].to_dict(),
        "fixed_gene_sequence_audit": {strain: {"checked": int((combined_audit.strain == strain).sum()),
                                                "supported": int(combined_audit.loc[combined_audit.strain == strain, "sequence_supported_pair"].sum())}
                                      for strain in STRAINS[1:]},
        "RNA_ratio_changes": ratio_changes.loc[ratio_changes.ratio.isin(ratios)].to_dict("records"),
        "strong_edge_output_sensitivity": sensitivity_rows,
        "interpretation_limits": "Exploratory summaries with two biological replicates per cell. No protein activity, motility assay, raw-read remapping or population-level replication. Inspect low counts and mapping quality. Gene-set ranks are retrospective and descriptive."
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=int) + "\n")
    print(ranking.head(12).to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2, default=int), flush=True)


if __name__ == "__main__":
    run()
