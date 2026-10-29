"""Audit author-defined ortholog pairs, exact MG1655 coordinates and sample IDs."""
from collections import defaultdict
from pathlib import Path
import gzip
import json
import re
import shutil

from Bio import SeqIO
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/expression"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)


def coordinate(value):
    match = re.fullmatch(r"([^:]+):(c?)(\d+)-(\d+)", str(value).strip())
    if match is None:
        return None
    accession, reverse, first, second = match.groups()
    return accession, min(int(first), int(second)), max(int(first), int(second)), -1 if reverse else 1


def metadata():
    samples = []
    current = None
    for line in gzip.open(DATA / "GSE77325_family.soft.gz", "rt", encoding="utf-8"):
        if line.startswith("^SAMPLE = "):
            current = {"accession": line.split(" = ", 1)[1].strip()}
            samples.append(current)
        elif current is not None and line.startswith("!Sample_title = "):
            title = line.split(" = ", 1)[1].strip()
            match = re.fullmatch(r"(\w+) transcriptome from (chemostat|batch|starvation) growth_(\d+)", title)
            assert match, title
            strain, state, replicate = match.groups()
            tag = {"chemostat": "CC", "batch": "b", "starvation": "_starv_"}[state]
            current.update(sample=strain + tag + replicate, strain=strain, state=state,
                           replicate=int(replicate), title=title)
        elif current is not None and line.startswith("!Sample_relation = "):
            current.setdefault("relations", []).append(line.split(" = ", 1)[1].strip())
    for sample in samples:
        sample["relations"] = " | ".join(sample.get("relations", []))
    frame = pd.DataFrame(samples).set_index("sample")
    assert len(frame) == 24 and frame.index.is_unique
    assert frame.groupby(["strain", "state"]).size().eq(2).all()
    return frame


def main():
    counts = pd.read_csv(DATA / "GSE77325_raw_count.txt.gz", sep="\t", index_col=0)
    mapping_all = pd.read_csv(DATA / "GSE77325_panGeneMap.txt.gz", sep="\t", index_col=0)
    mapping_all = mapping_all.map(lambda value: str(value).strip())
    assert counts.index.is_unique and mapping_all.index.is_unique
    assert counts.index.isin(mapping_all.index).all()
    assert (counts.values >= 0).all() and (counts.values % 1 == 0).all()
    sample_table = metadata().loc[counts.columns]
    sample_table["counted_reads"] = counts.sum(axis=0)
    sample_table["genes_with_10_reads"] = (counts >= 10).sum(axis=0)
    sample_table["pilot_primary"] = sample_table.strain.isin(["MG1655", "TW09308"]) & sample_table.state.isin(["chemostat", "starvation"])
    sample_table.to_csv(OUT / "sample_audit.csv")

    genome = SeqIO.read(ROOT / "data/genomes/NC_000913.2.gb", "genbank")
    assert genome.id == "NC_000913.2"
    features = defaultdict(list)
    for feature in genome.features:
        if feature.type == "CDS" and len(feature.location.parts) == 1:
            key = (genome.id, int(feature.location.start) + 1,
                   int(feature.location.end), feature.location.strand)
            features[key].append({
                "gene_id": feature.qualifiers.get("locus_tag", [""])[0],
                "gene_name": feature.qualifiers.get("gene", [""])[0],
                "product": feature.qualifiers.get("product", [""])[0],
                "pseudogene": "pseudo" in feature.qualifiers or "pseudogene" in feature.qualifiers,
                "has_translation": bool(feature.qualifiers.get("translation")),
            })

    use_counts = {strain: mapping_all.loc[mapping_all[strain] != "-", strain].value_counts()
                  for strain in ["MG1655", "TW09308"]}
    audit = mapping_all.loc[counts.index].copy()
    records = []
    for pan_gene, row in audit.iterrows():
        mg_coord, tw_coord = coordinate(row.MG1655), coordinate(row.TW09308)
        found = features.get(mg_coord, [])
        record = {"pan_gene": pan_gene, "pair_present": mg_coord is not None and tw_coord is not None,
                  "unique_MG1655_site": use_counts["MG1655"].get(row.MG1655, 0) == 1,
                  "unique_TW09308_site": use_counts["TW09308"].get(row.TW09308, 0) == 1,
                  "exact_MG1655_CDS_matches": len(found)}
        if len(found) == 1:
            record.update(found[0])
        if mg_coord and tw_coord:
            record["MG1655_cds_length"] = mg_coord[2] - mg_coord[1] + 1
            record["TW09308_cds_length"] = tw_coord[2] - tw_coord[1] + 1
            record["length_ratio"] = record["TW09308_cds_length"] / record["MG1655_cds_length"]
        records.append(record)
    audit = audit.join(pd.DataFrame(records).set_index("pan_gene"))
    audit["included_pair"] = (audit.pair_present & audit.unique_MG1655_site & audit.unique_TW09308_site
                               & audit.exact_MG1655_CDS_matches.eq(1)
                               & audit.has_translation.fillna(False).astype(bool)
                               & ~audit.pseudogene.fillna(False).astype(bool))
    audit["exclusion_reason"] = np.select(
        [~audit.pair_present, ~audit.unique_MG1655_site | ~audit.unique_TW09308_site,
         ~audit.exact_MG1655_CDS_matches.eq(1), ~audit.included_pair],
        ["missing_or_unparseable_pair", "coordinate_reused_in_mapping", "no_unique_exact_MG1655_CDS",
         "pseudogene_or_no_translation"], default="included")
    pairs = audit.loc[audit.included_pair].copy()
    assert pairs.gene_id.is_unique
    assert pairs.MG1655.is_unique and pairs.TW09308.is_unique

    audit.to_csv(OUT / "ortholog_audit_all.csv", index_label="pan_gene")
    shutil.copyfile(ROOT / "data/checkpoints/selected_sequence_audits.csv", OUT / "selected_sequence_audits.csv")
    print(json.dumps({"samples": len(sample_table), "count_rows": len(counts), "mapping_rows": len(audit)}, indent=2))


if __name__ == "__main__":
    main()
