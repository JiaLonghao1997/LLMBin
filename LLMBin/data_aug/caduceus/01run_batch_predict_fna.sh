#!/bin/bash

# Define paths
model_path="/home1/jialh/metaHiC/LLMs/GTDB/outputs/2025-01-04/08-20-52-160198-GTDB+Caduceus+sample10k+split32kb+layer8+step1w"
CONFIG_PATH="${model_path}/config.json"
CHECKPOINT_PATH="${model_path}/checkpoints/last.ckpt"
INPUT_FNA="/home1/jialh/metaHiC/LLMs/GTDB/gtdb_genomes_reps_r220/gtdb_genomes_reps_r220.sample1k.fna"
OUTPUT_PATH="/home1/jialh/metaHiC/LLMs/GTDB/01embedding"
OUTPUT_CSV="$OUTPUT_PATH/gtdb_genomes_reps_r220.sample1k.embed.csv"
SCRIPT_PATH="/home1/jialh/metaHiC/LLMs/caduceus"
mkdir -p $OUTPUT_PATH

# Run prediction
/home1/jialh/tools/anaconda3/envs/mamba/envs/caduceus_env/bin/python \
$SCRIPT_PATH/batch_predict_fna.py \
--config "$CONFIG_PATH" \
--checkpoint "$CHECKPOINT_PATH" \
--input "$INPUT_FNA" \
--outdir "${OUTPUT_PATH}" \
--output "$OUTPUT_CSV" \
--min_contig 1000 \
--contig_max_length 32768 \
--model_max_length 32768 \
--batch_size 16 \
--device cuda:1