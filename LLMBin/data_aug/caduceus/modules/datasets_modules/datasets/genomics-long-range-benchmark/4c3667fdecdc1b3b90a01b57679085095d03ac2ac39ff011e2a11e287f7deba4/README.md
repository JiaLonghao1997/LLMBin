---
license: cc-by-nc-sa-4.0
language:
- en
tags:
- Genomics
- Benchmarks
- Language Models
- DNA
pretty_name: Genomics Long-Range Benchmark
viewer: false
---

## Summary
The motivation of the genomics long-range benchmark (LRB) is to compile a set of 
biologically relevant genomic tasks requiring long-range dependencies which will act as a robust evaluation tool for genomic language models. 
While serving as a strong basis of evaluation, the benchmark must also be efficient and user-friendly. 
To achieve this we strike a balance between task complexity and computational cost through strategic decisions, such as down-sampling or combining datasets.

## Benchmark Tasks
The Genomics LRB is a collection of nine tasks which can be loaded by passing in the 
corresponding `task_name` into the `load_dataset` function. All of the following datasets 
allow the user to specify an arbitrarily long sequence length, giving more context 
to the task, by passing the `sequence_length` kwarg to `load_dataset`. Additional task 
specific kwargs, if applicable, are mentioned in the sections below.<br>
*Note that as you increase the context length to very large numbers you may start to reduce the size of the dataset since a large context size may 
cause indexing outside the boundaries of chromosomes.

| Task  | `task_name` | Sample Output                                                                             | ML Task Type            | # Outputs   | # Train Seqs | # Test Seqs | Data Source | 
|-------|-------------|-------------------------------------------------------------------------------------------|-------------------------|-------------|--------------|----------- |----------- |
| Variant Effect Causal eQTL           | `variant_effect_causal_eqtl`           | {ref sequence, alt sequence, label, tissue, chromosome,position, distance to nearest TSS} | SNP Classification      | 1           | 88717        |     8846    | GTEx (via [Enformer](https://www.nature.com/articles/s41592-021-01252-x)) |
| Variant Effect Pathogenic ClinVar    | `variant_effect_pathogenic_clinvar`    | {ref sequence, alt sequence, label, chromosome, position}                                 | SNP Classification      | 1           | 38634        |     1018    | ClinVar, gnomAD (via [GPN-MSA](https://www.biorxiv.org/content/10.1101/2023.10.10.561776v1)) |
| Variant Effect Pathogenic OMIM       | `variant_effect_pathogenic_omim`       | {ref sequence, alt sequence, label,chromosome, position}                                  | SNP Classification      | 1           | -            |     2321473    |OMIM, gnomAD (via [GPN-MSA](https://www.biorxiv.org/content/10.1101/2023.10.10.561776v1))  |
| CAGE Prediction                      | `cage_prediction`                      | {sequence, labels, chromosome,label_start_position,label_stop_position}                   | Binned Regression       | 50 per bin  | 33891        |     1922    | FANTOM5 (via [Basenji](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008050)) |
| Bulk RNA Expression                  | `bulk_rna_expression`                  | {sequence, labels, chromosome,position}                                                   | Seq-wise Regression     | 218         | 22827        |     990     | GTEx, FANTOM5 (via [ExPecto](https://www.nature.com/articles/s41588-018-0160-6)) |
| Chromatin Features Histone_Marks     | `chromatin_features_histone_marks`     | {sequence, labels,chromosome, position, label_start_position,label_stop_position}         | Seq-wise Classification | 20          | 2203689      |     227456    | ENCODE, Roadmap Epigenomics (via [DeepSea](https://pubmed.ncbi.nlm.nih.gov/30013180/) |
| Chromatin Features DNA_Accessibility | `chromatin_features_dna_accessibility` | {sequence, labels,chromosome, position, label_start_position,label_stop_position}         | Seq-wise Classification | 20          | 2203689      | 227456        | ENCODE, Roadmap Epigenomics (via [DeepSea](https://pubmed.ncbi.nlm.nih.gov/30013180/)) |
| Regulatory Elements Promoter         | `regulatory_element_promoter`          | {sequence, label,chromosome, start, stop, label_start_position,label_stop_position}       | Seq-wise Classification | 1|     953376   |     96240    | SCREEN |
| Regulatory Elements Enhancer         | `regulatory_element_enhancer`          | {sequence, label,chromosome, start, stop, label_start_position,label_stop_position}       | Seq-wise Classification | 1|     1914575  | 192201      | SCREEN |

## Usage Example
```python
from datasets import load_dataset

# Use this parameter to download sequences of arbitrary length (see docs below for edge cases)
sequence_length=2048

# One of:
# ["variant_effect_causal_eqtl","variant_effect_pathogenic_clinvar",
# "variant_effect_pathogenic_omim","cage_prediction", "bulk_rna_expression",
# "chromatin_features_histone_marks","chromatin_features_dna_accessibility",
# "regulatory_element_promoter","regulatory_element_enhancer"] 

task_name = "variant_effect_causal_eqtl"

dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=task_name,
    sequence_length=sequence_length,
    # subset = True, if applicable
)

```

### 1. Variant Effect Causal eQTL
Predicting the effects of genetic variants, particularly expression quantitative trait loci (eQTLs), is essential for understanding the molecular basis of several diseases.
eQTLs are genomic loci that are associated with variations in mRNA expression levels among individuals.
By linking genetic variants to causal changes in mRNA expression, researchers can 
uncover how certain variants contribute to disease development.

#### Source
Original data comes from GTEx. Processed data in the form of vcf files for positive 
and negative variants across 49 different tissue types were obtained from the
[Enformer paper](https://www.nature.com/articles/s41592-021-01252-x) located [here](https://console.cloud.google.com/storage/browser/dm-enformer/data/gtex_fine/vcf?pageState=%28%22StorageObjectListTable%22:%28%22f%22:%22%255B%255D%22%29%29&prefix=&forceOnObjectsSortingFiltering=false). 
Sequence data originates from the GRCh38 genome assembly.

#### Data Processing
Fine-mapped GTEx eQTLs originate from [Wang et al](https://www.nature.com/articles/s41467-021-23134-8), while the negative matched set of 
variants comes from [Avsec et al](https://www.nature.com/articles/s41592-021-01252-x)
. The statistical fine-mapping tool SuSiE was used to label variants. 
Variants from the fine-mapped eQTL set were selected and given positive labels if 
their posterior inclusion probability was > 0.9,
as assigned by SuSiE. Variants from the matched negative set were given negative labels if their
posterior inclusion probability was < 0.01.

#### Task Structure

Type: Binary classification<br>

Task Args:<br>
`sequence_length`: an integer type, the desired final sequence length<br>

Input: a genomic nucleotide sequence centered on the SNP with the reference allele at the SNP location, a genomic nucleotide sequence centered on the SNP with the alternative allele at the SNP location, and tissue type<br>
Output: a binary value referring to whether the variant has a causal effect on gene 
expression

#### Splits
Train: chromosomes 1-8, 11-22, X, Y<br>
Test: chromosomes 9,10

---

### 2. Variant Effect Pathogenic ClinVar
A coding variant refers to a genetic alteration that occurs within the protein-coding regions of the genome, also known as exons.
Such alterations can impact protein structure, function, stability, and interactions 
with other molecules, ultimately influencing cellular processes and potentially contributing to the development of genetic diseases. 
Predicting variant pathogenicity is crucial for guiding research into disease mechanisms and personalized treatment strategies, enhancing our ability to understand and manage genetic disorders effectively.

#### Source
Original data comes from ClinVar and gnomAD. However, we use processed data files 
from the [GPN-MSA paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10592768/) 
located [here](https://huggingface.co/datasets/songlab/human_variants/blob/main/test.parquet).
Sequence data originates from the GRCh38 genome assembly.

#### Data Processing
Positive labels correspond to pathogenic variants originating from ClinVar whose review status was
described as having at least a single submitted record with a classification but without assertion criteria.
The negative set are variants that are defined as common from gnomAD. gnomAD version 3.1.2 was downloaded and filtered to variants with allele number of at least 25,000. Common
variants were defined as those with MAF > 5%.

#### Task Structure

Type: Binary classification<br>

Task Args:<br>
`sequence_length`: an integer type, the desired final sequence length<br>

Input: a genomic nucleotide sequence centered on the SNP with the reference allele at the SNP location, a genomic nucleotide sequence centered on the SNP with the alternative allele at the SNP location<br>
Output: a binary value referring to whether the variant is pathogenic or not

#### Splits
Train: chromosomes 1-7, 9-22, X, Y<br>
Test: chromosomes 8

---

### 3. Variant Effect Pathogenic OMIM
Predicting the effects of regulatory variants on pathogenicity is crucial for understanding disease mechanisms.
Elements that regulate gene expression are often located in non-coding regions, and variants in these areas can disrupt normal cellular function, leading to disease.
Accurate predictions can identify biomarkers and therapeutic targets, enhancing personalized medicine and genetic risk assessment. 

#### Source
Original data comes from the Online Mendelian Inheritance in Man (OMIM) and gnomAD 
databases. 
However, we use processed data files from the 
[GPN-MSA paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10592768/) located [here](
https://huggingface.co/datasets/songlab/omim/blob/main/test.parquet).
Sequence data originates from the GRCh38 genome assembly.

#### Data Processing
Positive labeled data originates from a curated set of pathogenic variants located 
in the Online Mendelian Inheritance in Man (OMIM) catalog. The negative set is 
composed of variants that are defined as common from gnomAD. gnomAD version 3.1.2 was downloaded and filtered to variants with
allele number of at least 25,000. Common variants were defined as those with minor allele frequency
(MAF) > 5%. 

#### Task Structure

Type: Binary classification<br>

Task Args:<br>
`sequence_length`: an integer type, the desired final sequence length<br>
`subset`: a boolean type, whether to use the full dataset or a subset of the dataset (we provide this option as the full dataset has millions of samples)

Input: a genomic nucleotide sequence centered on the SNP with the reference allele at the SNP location, a genomic nucleotide sequence centered on the SNP with the alternative allele at the SNP location<br>
Output: a binary value referring to whether the variant is pathogenic or not

#### Splits
Test: all chromosomes

---

### 4. CAGE Prediction
CAGE provides accurate high-throughput measurements of RNA expression by mapping TSSs at a nucleotide-level resolution.
This is vital for detailed mapping of TSSs, understanding gene regulation mechanisms, and obtaining quantitative expression data to study gene activity comprehensively.

#### Source
Original CAGE data comes from FANTOM5. We used processed labeled data obtained from 
the [Basenji paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5932613/) which 
also used to train Enformer and is located [here](https://console.cloud.google.com/storage/browser/basenji_barnyard/data/human?pageState=%28%22StorageObjectListTable%22:%28%22f%22:%22%255B%255D%22%29%29&prefix=&forceOnObjectsSortingFiltering=false).
Sequence data originates from the GRCh38 genome assembly.

#### Data Processing
The original dataset from the Basenji paper includes labels for 638 CAGE total tracks over 896 bins (each bin corresponding to 128 base pairs) 
totaling over ~70 GB. In the interest of dataset size and user-friendliness, only a 
subset of the labels are selected. 
From the 638 CAGE tracks, 50 of these tracks are selected with the following criteria:

  1. Only select one cell line
  2. Only keep mock treated and remove other treatments
  3. Only select one donor
     
The [896 bins, 50 tracks] labels total in at ~7 GB. A description of the 50 included CAGE tracks can be found here `cage_prediction/label_mapping.csv`.
*Note the data in this repository for this task has not already been log(1+x) normalized. 

#### Task Structure

Type: Multi-variable regression<br>
Because this task involves predicting expression levels for 128bp bins and there are 896 total bins in the dataset, there are in essence labels for 896 * 128 = 114,688 basepair sequences. If
you request a sequence length smaller than 114,688 bps than the labels will be subsetted.

Task Args:<br>
`sequence_length`: an integer type, the desired final sequence length, *must be a multiple of 128 given the binned nature of labels<br>

Input: a genomic nucleotide sequence<br>
Output: a variable length vector depending on the requested sequence length [requested_sequence_length / 128, 50]

#### Splits
Train/Test splits were maintained from Basenji and Enformer where randomly sampling was used to generate the splits. Note that for this dataset a validation set is also returned. In practice we merged the validation 
set with the train set and use cross validation to select a new train and validation set from this combined set.


---

### 5. Bulk RNA Expression
Gene expression involves the process by which information encoded in a gene directs the synthesis of a functional gene product, typically a protein, through transcription and translation.
Transcriptional regulation determines the amount of mRNA produced, which is then translated into proteins. Developing a model that can predict RNA expression levels solely from sequence 
data is crucial for advancing our understanding of gene regulation, elucidating disease mechanisms, and identifying functional sequence variants.

#### Source
Original data comes from GTEx. We use processed data files from the [ExPecto paper](https://www.nature.com/articles/s41588-018-0160-6) found 
[here](https://github.com/FunctionLab/ExPecto/tree/master/resources). Sequence data originates from the GRCh37/hg19 genome assembly.

#### Data Processing
The authors of ExPecto determined representative TSS for Pol II transcribed genes 
based on quantification of CAGE reads from the FANTOM5 project. The specific procedure they used is as
follows, a CAGE peak was associated to a GENCODE gene if it was withing 1000 bps from a
GENCODE v24 annotated TSS. The most abundant CAGE peak for each gene was then selected
as the representative TSS. When no CAGE peak could be assigned to a gene, the annotated gene
start position was used as the representative TSS. We log(1 + x) normalized then standardized the
RNA-seq counts before training models. A list of names of tissues corresponding to 
the labels can be found here: `bulk_rna_expression/label_mapping.csv`. *Note the 
data in this repository for this task has already been log(1+x) normalized and 
standardized to mean 0 and unit variance.

#### Task Structure

Type: Multi-variable regression<br>

Task Args:<br>
`sequence_length`: an integer type, the desired final sequence length<br>

Input: a genomic nucleotide sequence centered around the CAGE representative trancription start site<br>
Output: a 218 length vector of continuous values corresponding to the bulk RNA expression levels in 218 different tissue types

#### Splits
Train: chromosomes 1-7,9-22,X,Y<br>
Test: chromosome 8

---
### 6. Chromatin Features 
Predicting chromatin features, such as histone marks and DNA accessibility, is crucial for understanding gene regulation, as these features indicate chromatin state and are essential for transcription activation.

#### Source
Original data used to generate labels for histone marks and DNase profiles comes from the ENCODE and Roadmap Epigenomics project. We used processed data files from the [Deep Sea paper](https://www.nature.com/articles/nmeth.3547) to build this dataset.
Sequence data originates from the GRCh37/hg19 genome assembly.

#### Data Processing
The authors of DeepSea processed the data by chunking the human genome
into 200 bp bins where for each bin labels were determined for hundreds of different chromatin
features. Only bins with at least one transcription factor binding event were 
considered for the dataset. If the bin overlapped with a peak region of the specific 
chromatin profile by more than half of the
sequence, a positive label was assigned. DNA sequences were obtained from the human reference
genome assembly GRCh37. To make the dataset more accessible, we randomly sub-sampled the
chromatin profiles from 125 to 20 tracks for the histones dataset and from 104 to 20 tracks for the
DNA accessibility dataset.

#### Task Structure

Type: Multi-label binary classification

Task Args:<br>
`sequence_length`: an integer type, the desired final sequence length<br>
`subset`: a boolean type, whether to use the full dataset or a subset of the dataset (we provide this option as the full dataset has millions of samples)

Input: a genomic nucleotide sequence centered on the 200 base pair bin that is associated with the labels<br>
Output: a vector of length 20 with binary entries

#### Splits
Train set: chromosomes 1-7,10-22<br>
Test set: chromosomes 8,9

---
### 7. Regulatory Elements
Cis-regulatory elements, such as promoters and enhancers, control the spatial and temporal expression of genes. 
These elements are essential for understanding gene regulation mechanisms and how genetic variations can lead to differences in gene expression.

#### Source
Original data annotations to build labels came from the Search Candidate cis-Regulatory Elements by ENCODE project. Sequence data originates from the GRCh38 
genome assembly.

#### Data Processing
The data is processed as follows, we break the human
reference genome into 200 bp non-overlapping chunks. If the 200 bp chunk overlaps by at least 50%
or more with a contiguous region from the set of annotated cis-regulatory elements (promoters or
enhancers), we label them as positive, else the chunk is labeled as negative. The resulting dataset
was composed of ∼15M negative samples and ∼50k positive promoter samples and ∼1M positive
enhancer samples. We randomly sub-sampled the negative set to 1M samples, and kept 
all positive
samples, to make this dataset more manageable in size.

#### Task Structure

Type: Binary classification

Task Args:<br>
`sequence_length`: an integer type, the desired final sequence length<br>
`subset`: a boolean type, whether to use the full dataset or a subset of the dataset (we provide this option as the full dataset has millions of samples)

Input: a genomic nucleotide sequence centered on the 200 base pair bin that is associated with the label<br>
Output: a single binary value

#### Splits
Train set: chromosomes 1-7,10-22<br>
Test set: chromosomes 8,9


## Genomic Annotations
The human genome annotations for both hg38 and hg19 reference genomes can be found in the `genome_annotation` folder. These annotations were used in our [visualization tool](https://github.com/kuleshov-group/genomics-lrb-viztool)
to slice test datasets by different genomic region.