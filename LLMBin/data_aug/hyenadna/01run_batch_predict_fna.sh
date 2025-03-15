#!/bin/bash

# Define paths
model_path="/home1/jialh/metaHiC/LLMs/GTDB/outputs/hyenadna-small-32k-seqlen-hf_46wsteps"
CONFIG_PATH="${model_path}/config_tree.txt"
CHECKPOINT_PATH="${model_path}/checkpoints/last.ckpt"
INPUT_FNA="/home1/jialh/metaHiC/LLMs/GTDB/gtdb_genomes_reps_r220/gtdb_genomes_reps_r220.sample1k.fna"
OUTPUT_PATH="/home1/jialh/metaHiC/LLMs/GTDB/01embedding"
OUTPUT_CSV="$OUTPUT_PATH/gtdb_genomes_reps_r220.sample1k.embed.csv"
SCRIPT_PATH="/home1/jialh/metaHiC/LLMs/hyena-dna"
mkdir -p $OUTPUT_PATH

# Run prediction
/home1/jialh/tools/anaconda3/envs/mamba/envs/DNABERT_S/bin/python \
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
--device cuda:3