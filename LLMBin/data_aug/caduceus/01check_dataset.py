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
    # "/home1/jialh/metaHiC/LLMs/caduceus/genomics-long-range-benchmark",

    task_name=task_name,
    sequence_length=sequence_length,
    # subset = True, if applicable
)

print(f"dataset: {dataset}")