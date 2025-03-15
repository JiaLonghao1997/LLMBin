export HYDRA_FULL_ERROR=1
export PATH="/home1/jialh/tools/anaconda3/envs/mamba/envs/caduceus_env/bin":${PATH}

#workdir="/home1/jialh/metaHiC/LLMs/caduceus"
datadir="/home1/jialh/metaHiC/LLMs/hyena-dna"
bed_file=${datadir}/data/hg38/human-sequences.bed
fasta_file=${datadir}/data/hg38/hg38.ml.fa

CUDA_VISIBLE_DEVICES=0 \
/home1/jialh/tools/anaconda3/envs/mamba/envs/caduceus_env/bin/python \
/home1/jialh/metaHiC/LLMs/caduceus/train.py \
experiment=hg38/hg38 \
callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500 \
dataset.max_length=1024 \
dataset.batch_size=64 \
dataset.mlm=true \
dataset.mlm_probability=0.15 \
dataset.rc_aug=false \
dataset.bed_file=${bed_file} \
dataset.fasta_file=${fasta_file} \
loader.num_workers=16 \
model=caduceus \
model.config.d_model=128 \
model.config.n_layer=4 \
model.config.bidirectional=true \
model.config.bidirectional_strategy=add \
model.config.bidirectional_weight_tie=true \
model.config.rcps=true \
optimizer.lr="8e-3" \
train.global_batch_size=64 \
trainer.max_steps=10000 \
+trainer.val_check_interval=10000 \
wandb=null
