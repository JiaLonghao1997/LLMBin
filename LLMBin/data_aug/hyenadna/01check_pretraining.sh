export PATH="/home1/jialh/tools/anaconda3/envs/mamba/envs/DNABERT_S/bin":${PATH}

workdir="/home1/jialh/metaHiC/LLMs/hyena-dna"
bed_file=${workdir}/data/hg38/human-sequences.bed
fasta_file=${workdir}/data/hg38/hg38.ml.fa

/home1/jialh/tools/anaconda3/envs/mamba/envs/DNABERT_S/bin/python \
-m train wandb=null experiment=hg38/hg38_hyena \
model.d_model=128 model.n_layer=2 \
dataset.batch_size=256 train.global_batch_size=256 \
dataset.max_length=1024 optimizer.lr=6e-4 trainer.devices=1 \
dataset.bed_file=${bed_file} \
dataset.fasta_file=${fasta_file}