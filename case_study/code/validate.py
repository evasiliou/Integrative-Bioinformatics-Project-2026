import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
DATA = ROOT / "data"

metadata = pd.read_csv(OUT / "sample_qc.csv", index_col=0)
assert len(metadata) == 16 and metadata.index.is_unique
assert metadata.groupby(["strain", "state"]).size().eq(2).all()
assert metadata.accession.nunique() == 16
assert metadata.size_factor.gt(0).all()
for strain, group in metadata.groupby("strain"):
    assert np.isclose(np.exp(np.log(group.size_factor).mean()), 1)

frozen = pd.read_csv(OUT / "frozen_selected_genes.csv")
assert frozen.gene_name.tolist() == ["flhD", "flhC", "fliA", "fliD", "cheZ", "flgK", "flgL", "motB"]
displayed = pd.read_csv(OUT / "displayed_effects.csv")
assert len(displayed) > 0 and not displayed.duplicated(["strain", "gene"]).any()
for strain in ["TW09308", "TW11588", "IAI1"]:
    frame = pd.read_csv(OUT / f"DE_MG1655_{strain}.csv")
    assert len(frame) > 0 and frame.gene_id.is_unique
    assert np.allclose(frame.relative_log2FoldChange - frame.MG1655_log2FoldChange, frame.interaction_log2FoldChange)
    expected = (frame.MG1655_padj < 0.05) & (frame.relative_padj < 0.05) & (frame.MG1655_log2FoldChange.abs() >= 1) & (frame.relative_log2FoldChange.abs() >= 1) & (frame.MG1655_log2FoldChange * frame.relative_log2FoldChange < 0)
    assert expected.eq(frame.opposite_DE).all()

audit = pd.read_csv(OUT / "all_selected_sequence_audits.csv")
assert len(audit) == 24 and not audit.duplicated(["strain", "gene_name"]).any()
failed = audit.loc[~audit.sequence_supported_pair]
for row in failed.itertuples():
    assert not ((displayed.strain == row.strain) & (displayed.gene == row.gene_name)).any()
new_audit = pd.read_csv(OUT / "selected_sequence_audits.csv")
assert new_audit.reference_coverage.dropna().between(0, 1).all()
assert new_audit.relative_coverage.dropna().between(0, 1).all()

points = pd.read_csv(OUT / "selected_replicate_values.csv")
ratios = pd.read_csv(OUT / "replicate_RNA_ratios.csv")
changes = pd.read_csv(OUT / "RNA_ratio_changes.csv")
assert len(points) == 128 and len(ratios) == 112 and len(changes) == 28
assert points.raw_count.gt(0).all()
assert np.isfinite(ratios.log2_raw_count_ratio).all()
counts = points.pivot(index="sample", columns="gene", values="raw_count")
normalized = points.pivot(index="sample", columns="gene", values="normalized_count")
for row in ratios.itertuples():
    if row.ratio == "selected_outputs/fliA":
        values = np.log2(counts.loc[row.sample])
        expected = ((values.flgK + values.flgL) / 2 + values.fliD + values.cheZ) / 3 - values.fliA
    else:
        numerator, denominator = row.ratio.split("/")
        expected = np.log2(counts.loc[row.sample, numerator] / counts.loc[row.sample, denominator])
        normalized_ratio = np.log2(normalized.loc[row.sample, numerator] / normalized.loc[row.sample, denominator])
        assert np.isclose(expected, normalized_ratio)
    assert np.isclose(expected, row.log2_raw_count_ratio)
for row in changes.itertuples():
    subset = ratios.loc[(ratios.strain == row.strain) & (ratios.ratio == row.ratio)]
    chemo = subset.loc[subset.state == "chemostat", "log2_raw_count_ratio"].to_numpy()
    starvation = subset.loc[subset.state == "starvation", "log2_raw_count_ratio"].to_numpy()
    possible = (starvation[:, None] - chemo[None, :]).ravel()
    assert np.isclose(row.change_log2_ratio, starvation.mean() - chemo.mean())
    assert np.isclose(row.fold_change_in_ratio, 2**row.change_log2_ratio)
    assert np.isclose(row.minimum_observed_pair_change, possible.min())
    assert np.isclose(row.maximum_observed_pair_change, possible.max())

ranking = pd.read_csv(OUT / "reference_regulon_screen.csv").set_index("regulator")
assert ranking.index.is_unique
assert ranking.tested_targets.ge(10).all() and ranking.target_operons.ge(3).all()
assert np.allclose(ranking.opposite_fraction, ranking.opposite_DE / ranking.tested_targets)
assert ranking["rank"].eq(ranking.opposite_fraction.rank(method="min", ascending=False)).all()
mapping = pd.read_csv(DATA / "expression/GSE77325_panGeneMap.txt.gz", sep="\t", index_col=0).fillna("").map(str.strip)
reference = pd.read_csv(OUT / "ortholog_audit_all.csv", index_col=0)
locus_context = []
for strain in ["MG1655", "TW09308", "TW11588", "IAI1"]:
    for genes in [("flhD", "flhC"), ("flgK", "flgL")]:
        coordinates = []
        for gene in genes:
            identifier = reference.index[reference.gene_name.eq(gene)].item()
            matched = re.fullmatch(r"([^:]+):(c?)(\d+)-(\d+)", mapping.loc[identifier, strain])
            source, reverse, first, second = matched.groups()
            lower, upper = sorted([int(first), int(second)])
            coordinates.append((source, reverse, lower, upper))
        assert coordinates[0][0] == coordinates[1][0]
        ordered = sorted(coordinates, key=lambda coordinate: coordinate[2])
        locus_context.append({"strain": strain, "genes": "/".join(genes),
                              "intergenic_bases": ordered[1][2] - ordered[0][3] - 1,
                              "same_strand": coordinates[0][1] == coordinates[1][1],
                              "boundary": "Author coordinate context, not independent transcript-unit validation."})
pd.DataFrame(locus_context).to_csv(OUT / "selected_locus_context.csv", index=False)
result = {"status": "passed", "samples": 16, "selected_gene_replicate_values": len(points),
          "RNA_ratio_values_checked": len(ratios), "ratio_summaries_checked": len(changes),
          "selected_sequence_pairs_checked": len(audit), "excluded_sequence_pairs": len(failed),
          "checks": "Sample uniqueness, mapping eligibility, interaction algebra, DE call definitions, sequence-audit exclusions, alignment coverage, RNA-ratio reconstruction and normalization invariance, summary ranks."}
(OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
