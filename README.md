# LLMBin
## <a name="overview"></a>Overview
Metagenomic binning based on microbial language models and contrastive learning. Traditional metagenomic binners could hardly cluster contigs with unusual compositions and abundances correctly based on uncontextualized features of the sequences, which ignores both the semantic relationship between genes and the positional embedding of k-mers. Genomic language models can generate contig representation that consider contextual features such as neighboring genes and regulatory sequences, which is expected to improve the performance of metagenomic binning. However, their capabilities have not been fully studied. <b>Here, we developed LLMBin, a metagenomic binning binner based on language models and contrastive learning.</b> The tool generates contextualized contig embeddings using a microbial genomic language model trained on 110,000 bacterial genomes from the Genome Taxonomy Database (GTDB), and then learns highly discriminative contig representations using multi-view contrastive learning. Benchmarks on three simulated datasets and three real datasets demonstrated that LLMBin outperforms existing state-of-the-art metagenomic binning tools, increasing 10% of nearly complete genomes and enhancing the semantic understanding of contigs.

![流程示意图_2](https://jialh.oss-cn-shanghai.aliyuncs.com/img2/流程示意图_2.jpg)

## <a name="References"></a>References
[1] Wang, Z., You, R., Han, H. et al. Effective binning of metagenomic contigs using contrastive multi-view representation learning. Nat Commun 15, 585 (2024). https://doi.org/10.1038/s41467-023-44290-z

[2] Meyer F, Fritz A, Deng Z L, et al. Critical assessment of metagenome interpretation: the second round of challenges[J]. Nature methods, 2022, 19(4): 429-440.

[3] Parks D H, Imelfort M, Skennerton C T, et al. CheckM: assessing the quality of microbial genomes recovered from isolates, single cells, and metagenomes[J]. Genome research, 2015, 25(7): 1043-1055.

[4] Pan S, Zhu C, Zhao X M, et al. A deep siamese neural network improves metagenome-assembled genomes in microbiome datasets across different environments[J]. Nature communications, 2022, 13(1): 2326.
