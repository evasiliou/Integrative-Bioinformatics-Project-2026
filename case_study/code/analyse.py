import hashlib
import importlib.metadata
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
STRAINS = ["MG1655", "TW09308", "TW11588", "IAI1"]
SELECTED = ["flhD", "flhC", "fliA", "fliD", "cheZ", "flgK", "flgL", "motB"]


def coordinate(value):
    matched = re.fullmatch(r"([^:]+):(c?)(\d+)-(\d+)", str(value).strip())
    if matched is None:
        return None
    accession, reverse, first, second = matched.groups()
    return accession, min(int(first), int(second)), max(int(first), int(second)), -1 if reverse else 1


def median_factors(frame):
    positive = frame.loc[:, (frame > 0).all(axis=0)]
    assert positive.shape[1] > 100
    values = np.log(positive.to_numpy(dtype=float))
    factors = np.exp(np.median(values - values.mean(axis=0), axis=1))
    return pd.Series(factors / np.exp(np.log(factors).mean()), index=frame.index)


class FixedFactorDataSet(DeseqDataSet):
    def fit_size_factors(self, fit_type=None, control_genes=None):
        super().fit_size_factors(fit_type=fit_type, control_genes=control_genes)
        self.obs["size_factors"] = self.fixed_factors.loc[self.obs_names].to_numpy()
        self.layers["normed_counts"] = self.X / self.obs["size_factors"].to_numpy()[:, None]
        self.var["_normed_means"] = self.layers["normed_counts"].mean(axis=0)


def test(dataset, contrast):
    statistics = DeseqStats(dataset, contrast=np.asarray(contrast, dtype=float), alpha=0.05,
                            independent_filter=False, n_cpus=4, quiet=True)
    statistics.summary()
    result = statistics.results_df.copy()
    result["ci_low"] = result.log2FoldChange - 1.96 * result.lfcSE
    result["ci_high"] = result.log2FoldChange + 1.96 * result.lfcSE
    return result


def run():
    counts = pd.read_csv(DATA / "expression/GSE77325_raw_count.txt.gz", sep="\t", index_col=0)
    metadata = pd.read_csv(OUT / "sample_audit.csv", index_col=0)
    metadata = metadata.loc[metadata.state.isin(["chemostat", "starvation"])].copy()
    author_mapping = pd.read_csv(DATA / "expression/GSE77325_panGeneMap.txt.gz", sep="\t", index_col=0).fillna("").map(str.strip)
    audit = pd.read_csv(OUT / "ortholog_audit_all.csv", index_col=0)
    valid = audit.exact_MG1655_CDS_matches.eq(1) & audit.has_translation.fillna(False).astype(bool) & ~audit.pseudogene.fillna(False).astype(bool)
    audit = audit.loc[valid].copy()
    audit = audit.loc[~audit.gene_id.duplicated(keep=False)]
    for strain in STRAINS:
        parsed = audit[strain].map(coordinate)
        audit[strain + "_length"] = parsed.map(lambda value: value[2] - value[1] + 1 if value else np.nan)
        frequencies = author_mapping.loc[author_mapping[strain] != "-", strain].value_counts()
        audit[strain + "_unique"] = audit[strain].map(frequencies).eq(1)
    for strain in STRAINS:
        audit[strain + "_ratio"] = audit[strain + "_length"] / audit.MG1655_length
        audit[strain + "_eligible"] = (audit[strain + "_unique"] & audit.MG1655_unique
                                          & audit[strain + "_ratio"].between(0.8, 1.25))
    audit.to_csv(OUT / "mapping_audit.csv")
    selected = audit.loc[audit.gene_name.isin(SELECTED)].set_index("gene_name", drop=False).loc[SELECTED]
    assert len(selected) == 8
    selected.to_csv(OUT / "frozen_selected_genes.csv", index=False)

    size_factors = pd.Series(index=metadata.index, dtype=float)
    normalization_rows = []
    for strain in STRAINS:
        samples = metadata.index[metadata.strain == strain]
        universe = audit.index[audit[strain + "_eligible"]]
        frame = counts.loc[universe, samples].T
        size_factors.loc[samples] = median_factors(frame)
        normalization_rows.append({"strain": strain, "normalisation_genes": len(universe),
                                   "positive_genes": int((frame > 0).all(axis=0).sum())})
    metadata["size_factor"] = size_factors
    metadata.to_csv(OUT / "sample_qc.csv")
    pd.DataFrame(normalization_rows).to_csv(OUT / "normalisation_universes.csv", index=False)

    annotation = json.loads((DATA / "regulondb/NetworkSigmaGene.json").read_text())["content"]
    sigma = pd.read_csv(io.StringIO(annotation), sep="\t", dtype=str).fillna("")
    sigma.columns = [column.split(")", 1)[-1].strip() for column in sigma.columns]
    sigma = sigma.map(str.strip)
    target_names = set(sigma.loc[(sigma.sigmaName == "sigma28") & sigma.confidenceLevel.isin(["C", "S"]) & (sigma.function == "+"), "regulatedGeneName"])
    product_content = json.loads((DATA / "regulondb/GeneProductSet.json").read_text())["content"]
    products = pd.read_csv(io.StringIO(product_content), sep="\t", dtype=str).fillna("")
    products.columns = [column.split(")", 1)[-1].strip() for column in products.columns]
    products = products.map(str.strip)
    target_ids = set(products.loc[products.geneName.isin(target_names), "bnumber"])
    assert len(target_ids) >= 22
    selected_rows, summaries = [], []
    for strain in STRAINS[1:]:
        pair_metadata = metadata.loc[metadata.strain.isin(["MG1655", strain])].copy()
        genes = audit.index[audit.MG1655_eligible & audit[strain + "_eligible"]]
        frame = counts.loc[genes, pair_metadata.index].T
        keep = (frame >= 10).sum(axis=0) >= 2
        for member in ["MG1655", strain]:
            keep &= frame.loc[pair_metadata.strain == member].sum(axis=0) >= 20
        frame = frame.loc[:, keep].astype(int)
        design = pd.DataFrame({"Intercept": 1.0, "relative": (pair_metadata.strain == strain).astype(float),
                               "starvation": (pair_metadata.state == "starvation").astype(float)}, index=pair_metadata.index)
        design["interaction"] = design.relative * design.starvation
        assert np.linalg.matrix_rank(design) == 4
        dataset = FixedFactorDataSet(counts=frame, metadata=pair_metadata, design=design, refit_cooks=False, n_cpus=4, quiet=True)
        dataset.fixed_factors = size_factors.loc[pair_metadata.index]
        dataset.deseq2()
        result = audit.loc[frame.columns, ["gene_id", "gene_name", "product"]].copy()
        for label, vector in {"MG1655": [0, 0, 1, 0], "relative": [0, 0, 1, 1], "interaction": [0, 0, 0, 1]}.items():
            result = result.join(test(dataset, vector).add_prefix(label + "_"))
        result["relative_strain"] = strain
        result["MG1655_DE"] = (result.MG1655_padj < 0.05) & (result.MG1655_log2FoldChange.abs() >= 1)
        result["relative_DE"] = (result.relative_padj < 0.05) & (result.relative_log2FoldChange.abs() >= 1)
        result["opposite_DE"] = result.MG1655_DE & result.relative_DE & (result.MG1655_log2FoldChange * result.relative_log2FoldChange < 0)
        result["FliA_target"] = result.gene_id.isin(target_ids)
        result.to_csv(OUT / f"DE_MG1655_{strain}.csv", index_label="pan_gene")
        selected_rows.append(result.loc[result.gene_name.isin(SELECTED)].reset_index(names="pan_gene"))
        target = result.loc[result.FliA_target]
        background = result.loc[~result.FliA_target]
        summaries.append({"relative": strain, "tested_genes": len(result), "FliA_targets": len(target),
                          "FliA_opposite_DE": int(target.opposite_DE.sum()), "other_genes": len(background),
                          "other_opposite_DE": int(background.opposite_DE.sum()),
                          "selected_genes_tested": int(result.gene_name.isin(SELECTED).sum())})
        print(strain, "model complete", summaries[-1], flush=True)
    pd.concat(selected_rows).to_csv(OUT / "selected_pairwise_results.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "FliA_response_summary.csv", index=False)

    names_to_pan = {name: audit.index[audit.gene_name == name][0] for name in SELECTED}
    point_rows = []
    for strain in STRAINS:
        samples = metadata.index[metadata.strain == strain]
        for name in SELECTED:
            identifier = names_to_pan[name]
            normalized = counts.loc[identifier, samples] / size_factors.loc[samples]
            baseline = normalized.loc[metadata.loc[samples, "state"] == "chemostat"].mean()
            for sample in samples:
                point_rows.append({"strain": strain, "sample": sample, "state": metadata.loc[sample, "state"],
                                   "replicate": metadata.loc[sample, "replicate"], "gene": name,
                                   "raw_count": int(counts.loc[identifier, sample]), "normalized_count": normalized[sample],
                                   "log2_vs_chemostat_mean": np.log2(normalized[sample] / baseline) if normalized[sample] > 0 and baseline > 0 else np.nan,
                                   "mapping_length_eligible": bool(audit.loc[identifier, strain + "_eligible"])})
    pd.DataFrame(point_rows).to_csv(OUT / "selected_replicate_values.csv", index=False)

    ratio_rows, ratio_changes = [], []
    for strain in STRAINS:
        samples = metadata.index[metadata.strain == strain]
        for numerator, denominator in [("fliA", "flhD"), ("fliA", "flhC"), ("flgK", "fliA"), ("flgL", "fliA"), ("fliD", "fliA"), ("cheZ", "fliA")]:
            values = np.log2(counts.loc[names_to_pan[numerator], samples] / counts.loc[names_to_pan[denominator], samples])
            assert np.isfinite(values).all()
            for sample, value in values.items():
                ratio_rows.append({"strain": strain, "sample": sample, "state": metadata.loc[sample, "state"],
                                   "replicate": metadata.loc[sample, "replicate"], "ratio": numerator + "/" + denominator,
                                   "log2_raw_count_ratio": value})
            chemo = values.loc[metadata.loc[samples, "state"] == "chemostat"].to_numpy()
            starvation = values.loc[metadata.loc[samples, "state"] == "starvation"].to_numpy()
            possible_changes = (starvation[:, None] - chemo[None, :]).ravel()
            change = starvation.mean() - chemo.mean()
            ratio_changes.append({"strain": strain, "ratio": numerator + "/" + denominator,
                                  "change_log2_ratio": change, "fold_change_in_ratio": 2**change,
                                  "minimum_observed_pair_change": possible_changes.min(),
                                  "maximum_observed_pair_change": possible_changes.max(),
                                  "all_observed_pairings_positive": bool((possible_changes > 0).all()),
                                  "all_observed_pairings_negative": bool((possible_changes < 0).all())})
        log_counts = np.log2(counts.loc[[names_to_pan[name] for name in ["flgK", "flgL", "fliD", "cheZ", "fliA"]], samples])
        log_counts.index = ["flgK", "flgL", "fliD", "cheZ", "fliA"]
        values = ((log_counts.loc["flgK"] + log_counts.loc["flgL"]) / 2 + log_counts.loc["fliD"] + log_counts.loc["cheZ"]) / 3 - log_counts.loc["fliA"]
        for sample, value in values.items():
            ratio_rows.append({"strain": strain, "sample": sample, "state": metadata.loc[sample, "state"],
                               "replicate": metadata.loc[sample, "replicate"], "ratio": "selected_outputs/fliA",
                               "log2_raw_count_ratio": value})
        chemo = values.loc[metadata.loc[samples, "state"] == "chemostat"].to_numpy()
        starvation = values.loc[metadata.loc[samples, "state"] == "starvation"].to_numpy()
        possible_changes = (starvation[:, None] - chemo[None, :]).ravel()
        change = starvation.mean() - chemo.mean()
        ratio_changes.append({"strain": strain, "ratio": "selected_outputs/fliA", "change_log2_ratio": change,
                              "fold_change_in_ratio": 2**change, "minimum_observed_pair_change": possible_changes.min(),
                              "maximum_observed_pair_change": possible_changes.max(),
                              "all_observed_pairings_positive": bool((possible_changes > 0).all()),
                              "all_observed_pairings_negative": bool((possible_changes < 0).all())})
    pd.DataFrame(ratio_rows).to_csv(OUT / "replicate_RNA_ratios.csv", index=False)
    pd.DataFrame(ratio_changes).to_csv(OUT / "RNA_ratio_changes.csv", index=False)
    methods = {
        "fixed_gene_set": SELECTED, "samples": "GSE77325 chemostat and four-hour starvation; four isolates; two biological replicates per cell.",
        "normalisation": "Median-of-ratios separately within each strain using uniquely mapped MG1655-matched coding rows with CDS length ratio 0.8-1.25; factors centered within strain. Same factors in all pairwise fits.",
        "DE": "Three separate eight-sample strain + condition + interaction NB fits; common gene-wise dispersion within each pair; FDR correction over each full tested genome-wide universe. No independent filtering; Cook filtering retained; no outlier replacement. All primary mapping lengths must lie in 0.8-1.25 of MG1655.",
        "filter": "At least 10 reads in two of the eight samples and at least 20 total reads in each strain's four samples.",
        "RNA_ratios": "Per-sample log2 ratios of original integer counts, no pseudocount. Change is difference of mean log ratios. Within-strain gene lengths and global size factors cancel in this change. Observed-pair ranges are descriptive and are NOT confidence intervals. Pairing of biological replicates is unverified and not assumed.",
        "selected_output_ratio": "Geometric mean of flgK/flgL (one operon weight), fliD and cheZ (one weight each), divided by fliA. Uses the four selected output genes with supported sequence matches in all strains. This is a selected-gene RNA summary, not a measure of FliA protein activity.",
        "selection": "Genes selected from prior exploratory work for a focused case study. RNA-ratio comparisons are exploratory follow-up summaries. FliA target annotation fixed from RegulonDB confirmed/strong sigma28 edges.",
        "limits": "RNA ratios do not measure regulator activity or promoter strength. Two replicates limit calibration, one isolate per lineage limits generality, and existing processed counts cannot validate read coverage or remapping. Sequence audit is separate.",
        "software": {package: importlib.metadata.version(package) for package in ["pydeseq2", "pandas", "numpy", "biopython"]},
        "inputs": {name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for name, path in {
            "counts": "data/expression/GSE77325_raw_count.txt.gz", "mapping": "data/expression/GSE77325_panGeneMap.txt.gz",
            "gene_audit": "results/ortholog_audit_all.csv", "samples": "results/sample_audit.csv"}.items()}}
    (OUT / "methods.json").write_text(json.dumps(methods, indent=2) + "\n")
    print(pd.DataFrame(ratio_changes).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
