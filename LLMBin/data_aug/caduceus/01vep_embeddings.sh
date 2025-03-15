export PATH=/home1/jialh/tools/anaconda3/envs/mamba/envs/caduceus_env/bin:${PATH}
model_path="/home1/jialh/metaHiC/LLMs/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"

export HF_HOME="/home1/jialh/metaHiC/LLMs/caduceus"

# torchrun 允许用户在多个 GPU 或多个节点上并行训练模型，支持数据并行和模型并行。
/home1/jialh/tools/anaconda3/envs/mamba/envs/caduceus_env/bin/torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    --node-rank=0 \
    /home1/jialh/metaHiC/LLMs/caduceus/vep_embeddings.py \
    --num_workers=2 \
    --seq_len=131072  \
    --bp_per_token=1  \
    --embed_dump_batch_size=1 \
    --name="caduceus-ps_downstream-seqlen=131k"  \
    --model_name_or_path=${model_path} \
    --rcps