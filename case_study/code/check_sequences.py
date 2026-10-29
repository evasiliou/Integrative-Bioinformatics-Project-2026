import hashlib
import json
import re
import subprocess
import shutil
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/genomes"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
WORK = OUT / "sequence_work"
WORK.mkdir(exist_ok=True)
SELECTED = ["flhD", "flhC", "fliA", "fliD", "cheZ", "flgK", "flgL", "motB"]
ACCESSIONS = {"TW09308": "AEME01000001.1", "TW11588": "AEMF01000001.1", "IAI1": "NC_011741.1"}
DIAMOND = shutil.which("diamond")


def run():
    if DIAMOND is None:
        raise SystemExit("Install DIAMOND 2.2.8 and put diamond on PATH to regenerate sequence checks.")
    mapping = pd.read_csv(ROOT / "data/expression/GSE77325_panGeneMap.txt.gz", sep="\t", index_col=0).fillna("").map(str.strip)
    reference = pd.read_csv(OUT / "ortholog_audit_all.csv", index_col=0)
    candidates = reference.loc[reference.gene_name.isin(SELECTED)].copy()
    reference_genome = SeqIO.read(DATA / "NC_000913.2.gb", "genbank")
    reference_proteins = {
        feature.qualifiers["locus_tag"][0]: feature.qualifiers["translation"][0]
        for feature in reference_genome.features
        if feature.type == "CDS" and "translation" in feature.qualifiers and "locus_tag" in feature.qualifiers
    }
    reference_fasta = WORK / "MG_proteome.faa"
    reference_fasta.write_text("".join(f">{identifier}\n{protein}\n" for identifier, protein in reference_proteins.items()))
    subprocess.run([DIAMOND, "makedb", "--in", str(reference_fasta), "--db", str(WORK / "MG")], check=True)
    manifest = []
    all_rows = []
    for strain, accession in ACCESSIONS.items():
        destination = DATA / (strain + ".gb")
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id={accession}&rettype=gbwithparts&retmode=text"
        genome = SeqIO.read(destination, "genbank")
        assert genome.id == accession
        manifest.append({"strain": strain, "accession": genome.id, "url": url, "length": len(genome),
                         "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
        proteins = {}
        coordinate_rows = {}
        for identifier, value in mapping[strain].items():
            matched = re.fullmatch(r"([^:]+):(c?)(\d+)-(\d+)", value)
            if matched is None:
                continue
            source, reverse, first, second = matched.groups()
            lower, upper = sorted([int(first), int(second)])
            if upper > len(genome):
                continue
            coding = genome.seq[lower - 1:upper]
            if reverse:
                coding = coding.reverse_complement()
            translated = str(coding[:len(coding) // 3 * 3].translate(table=11))
            protein = translated.rstrip("*")
            valid = len(protein) >= 10 and "*" not in protein and "X" not in protein
            coordinate_rows[identifier] = {"coordinate": value, "lower": lower, "upper": upper,
                                           "strand": -1 if reverse else 1, "cds_length": len(coding),
                                           "translated_length": len(protein), "no_internal_stops_or_X": valid,
                                           "terminal_stop": translated.endswith("*"), "length_multiple_of_three": len(coding) % 3 == 0}
            if valid:
                proteins[identifier] = protein
        fasta = WORK / (strain + ".faa")
        fasta.write_text("".join(f">{identifier}\n{protein}\n" for identifier, protein in proteins.items()))
        database = WORK / strain
        subprocess.run([str(DIAMOND), "makedb", "--in", str(fasta), "--db", str(database)], check=True)
        winners = {}
        for direction, query, target in [("forward", reference_fasta, database),
                                          ("reverse", fasta, WORK / "MG")]:
            output = OUT / f"{strain}_{direction}.tsv"
            subprocess.run([str(DIAMOND), "blastp", "--query", str(query), "--db", str(target),
                            "--out", str(output), "--sensitive", "--threads", "4", "--evalue", "1e-10",
                            "--max-target-seqs", "5", "--outfmt", "6", "qseqid", "sseqid", "pident",
                            "length", "qlen", "slen", "bitscore", "evalue", "qstart", "qend", "sstart", "send"], check=True)
            hits = pd.read_csv(output, sep="\t", names=["query", "subject", "identity", "aligned", "query_length", "subject_length", "bits", "evalue", "query_start", "query_end", "subject_start", "subject_end"])
            best = {}
            for identifier, group in hits.groupby("query"):
                group = group.sort_values("bits", ascending=False).drop_duplicates("subject")
                first = group.iloc[0]
                if len(group) == 1 or first.bits > group.iloc[1].bits:
                    best[identifier] = first
            winners[direction] = best
        for identifier, row in candidates.iterrows():
            detail = coordinate_rows.get(identifier, {})
            forward = winners["forward"].get(row.gene_id)
            backward = winners["reverse"].get(identifier)
            reciprocal = forward is not None and backward is not None and forward.subject == identifier and backward.subject == row.gene_id
            identity = float(forward.identity) if forward is not None and forward.subject == identifier else None
            coverage_reference = float((forward.query_end - forward.query_start + 1) / forward.query_length) if identity is not None else None
            coverage_relative = float((forward.subject_end - forward.subject_start + 1) / forward.subject_length) if identity is not None else None
            length_ratio = detail.get("cds_length", 0) / row.MG1655_cds_length
            supported = reciprocal and identity >= 70 and coverage_reference >= 0.8 and coverage_relative >= 0.8 and 0.8 <= length_ratio <= 1.25 and detail.get("length_multiple_of_three", False)
            all_rows.append({"strain": strain, "pan_gene": identifier, "gene_id": row.gene_id,
                             "gene_name": row.gene_name, **detail, "length_ratio": length_ratio,
                             "reciprocal_best_hit": reciprocal, "identity_percent": identity,
                             "reference_coverage": coverage_reference, "relative_coverage": coverage_relative,
                             "sequence_supported_pair": supported,
                             "best_reference_match_pan_gene": forward.subject if forward is not None else ""})
        pd.DataFrame(all_rows).to_csv(OUT / "selected_sequence_audits.csv", index=False)
        (OUT / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(strain, "audit complete", flush=True)
    print(pd.DataFrame(all_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
