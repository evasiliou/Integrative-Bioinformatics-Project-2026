import io
import json
import re
from pathlib import Path

import pandas as pd


def operon_membership(genes):
    source = Path(__file__).resolve().parents[1] / "data/regulondb"

    def export(name):
        content = json.loads((source / (name + ".json")).read_text())["content"]
        frame = pd.read_csv(io.StringIO(content), sep="\t", dtype=str).fillna("")
        frame.columns = [column.split(")", 1)[-1].strip() for column in frame.columns]
        return frame.map(str.strip)

    products = export("GeneProductSet")
    name_to_gene = dict(zip(products.geneName, products.bnumber))
    parents = {gene: gene for gene in genes}

    def representative(gene):
        while parents[gene] != gene:
            parents[gene] = parents[parents[gene]]
            gene = parents[gene]
        return gene

    for row in export("OperonSet").itertuples():
        included = [name_to_gene[name] for name in re.split(r"[;|]", row.operonGenes)
                    if name in name_to_gene and name_to_gene[name] in parents]
        for gene in included[1:]:
            parents[representative(gene)] = representative(included[0])
    return {gene: representative(gene) for gene in genes}
