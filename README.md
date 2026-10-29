# Integrative-Bioinformatics-Project-2026

The aim of this project is to explore how protein interactions relate to gene expression across environments. Our hypothesis is that the number of interactors (the degree) of each protein is related to the differential gene expression across different conditions, including different carbon and nitrogen sources. Thus, we suppose that proteins with a high number or interactors connect several biological processes that respond differently to environmental change, while proteins with lower number of partners will participate in a single process. We also conducted a case-study to investigate further the gene expression related to cellular motility.

## Pipeline
1. Data preprocessing from STRING and iModulon databases for *E. coli*
2. Data Analysis: (i) Network degree vs. expression coordination, (ii) Grouping partners by biological function, (iii) Grouping conditions into five categories
3. Case Study of flagellar regulatory system

## Dataset
As the data files are not included in the GitHub repository, the datasets required to reproduce the analyses are listed below.

### From iModulonDB precise-MG1655
precise\_mg1655-log\_tpm.csv<br>
precise\_mg1655-samle\_table.csv<br>
precise\_mg1655-gene\_info.csv<br>
precise\_mg1655-trn.csv<br>
precise\_mg1655-A.csv<br>

### Data for case study
`./case_study/data/expression` from GSE77325<br>
GSE77325_raw_count.txt<br>
GSE77325_family.soft.gz<br>
GSE77325_Gene_annotation.txt.gz<br>
GSE77325_panGeneMap.txt.gz<br>
GSE77325_raw_count.txt.gz<br>

`./case_study/data/genomes` from GenBank<br>
IAI1.gb<br>
NC_000913.2.gb<br>
TW09308.gb<br>
TW11588.gb<br>

`./case_study/data/regulondb` from RegulonDB<br>
GeneProductSet.json<br>
NetworkRegulatorGene.json<br>
NetworkSigmaGene.json<br>
OperonSet.json<br>
TFSet.json<br>