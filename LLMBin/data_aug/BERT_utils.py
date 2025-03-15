import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import os
import sys
# from evo import Evo
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import GPUtil

from scipy.optimize import linear_sum_assignment
sys.path.append("/public/home/jialh/metaHiC/tools/BERTBin/BERTBin/data_aug")
# from .hyenadna.batch_hyenadna_predict_fna import gtdb_hyenadna_embedding
# from .caduceus.batch_caduceus_predict_fna import gtdb_caduceus_embedding
# sys.path.append("/public/home/jialh/metaHiC/tools/BERTBin/BERTBin/data_aug")
# from hyenadna.huggingface import HyenaDNAPreTrainedModel
# from hyenadna.standalone_hyenadna import CharacterTokenizer
# sys.path.append("/home1/jialh/metaHiC/LLMs/evo")
# from evo import Evo, score_sequences


def get_embedding(dna_sequences, 
                  model,
                  species,
                  sample,
                  outdir,
                  contig_max_length,
                  model_max_length,
                  logger,
                  device,
                  task_name="clustering",
                  post_fix="",
                  batch_size=16,
                  test_model_dir="./test_model"):

    logger.info(f"Calculate embedding for {model} {species} {sample}")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA Device: {torch.cuda.current_device()}")
    logger.info(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    if model == "tnf":
        embedding = calculate_tnf(dna_sequences, logger=logger, device=device,)
    elif model == "tnf_k":
        embedding = calculate_tnf(dna_sequences, logger=logger, device=device, kernel=True)
    elif model == "dna2vec":
        embedding = calculate_dna2vec_embedding(dna_sequences,
                                                logger=logger,
                                                device=device,
                                                embedding_dir=embedding_dir,)
    elif model == "hyenadna":
        embedding = calculate_llm_embedding(dna_sequences,
                                            logger=logger,
                                            device=device,
                                            model_name_or_path=test_model_dir,
                                            model_max_length=model_max_length,
                                            batch_size=batch_size)
    elif model == "dnabert2":
        embedding = calculate_llm_embedding(dna_sequences,
                                            logger=logger,
                                            device=device,
                                            model_name_or_path=test_model_dir,
                                            model_max_length=model_max_length,
                                            batch_size=batch_size)
    elif model == "nt":
        embedding = calculate_llm_embedding(dna_sequences,
                                            logger=logger,
                                            device=device,
                                            model_name_or_path="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
                                            model_max_length=model_max_length,
                                            batch_size=batch_size)
    elif model == "dnabert-s":
        embedding = calculate_llm_embedding(dna_sequences,
                                            logger=logger,
                                            device=device,
                                            model_name_or_path=test_model_dir,
                                            model_max_length=model_max_length,
                                            batch_size=batch_size)
    elif model in ["evo-1-8k-base", "evo-1-131k-base"]:
        embedding = calculate_evo_embedding(dna_sequences,
                                            logger=logger,
                                            device=device,
                                            model_name_or_path=test_model_dir,
                                            model_max_length=model_max_length,
                                            batch_size=batch_size )
    elif model in ['GTDB_HyenaDNA']:
        from .hyenadna.batch_hyenadna_predict_fna import gtdb_hyenadna_embedding
        embedding = gtdb_hyenadna_embedding(dna_sequences,
                                            logger=logger,
                                            device=device,
                                            model_name_or_path=test_model_dir,
                                            model_max_length=model_max_length,
                                            batch_size=batch_size)
    elif model in ['GTDB_Caduceus']:
        from .caduceus.batch_caduceus_predict_fna import gtdb_caduceus_embedding
        embedding = gtdb_caduceus_embedding(dna_sequences,
                                            logger=logger,
                                            device=device,
                                            model_name_or_path=test_model_dir,
                                            model_max_length=model_max_length,
                                            batch_size=batch_size)
    else:
        raise ValueError(f"Unknown model {model}")

    return embedding


def calculate_tnf(dna_sequences, logger, device, kernel=False):
    # Define all possible tetra-nucleotides
    nucleotides = ['A', 'T', 'C', 'G']
    tetra_nucleotides = [a+b+c+d for a in nucleotides for b in nucleotides for c in nucleotides for d in nucleotides]
    
    # build mapping from tetra-nucleotide to index
    tnf_index = {tn: i for i, tn in enumerate(tetra_nucleotides)}        

    # Iterate over each sequence and update counts
    embedding = np.zeros((len(dna_sequences), len(tetra_nucleotides)))
    for j, seq in enumerate(dna_sequences):
        for i in range(len(seq) - 3):
            tetra_nuc = seq[i:i+4]
            embedding[j, tnf_index[tetra_nuc]] += 1
    
    # Convert counts to frequencies
    total_counts = np.sum(embedding, axis=1)
    embedding = embedding / total_counts[:, None]

    if kernel:
        def validate_input_array(array):
            "Returns array similar to input array but C-contiguous and with own data."
            if not array.flags["C_CONTIGUOUS"]:
                array = np.ascontiguousarray(array)
            if not array.flags["OWNDATA"]:
                array = array.copy()

            assert array.flags["C_CONTIGUOUS"] and array.flags["OWNDATA"]

            return array

        npz = np.load("./helper/kernel.npz")
        kernel = validate_input_array(npz["arr_0"])
        embedding += -(1 / 256)
        embedding = np.dot(embedding, kernel)
        
    return embedding


def calculate_dna2vec_embedding(dna_sequences, logger, device, embedding_dir):
    embedding_file = os.path.join(embedding_dir, "tnf.npy")
    if os.path.exists(embedding_file):
        logger.info(f"Load embedding from file {embedding_file}")
        tnf_embedding = np.load(embedding_file)
    else:
        tnf_embedding = calculate_tnf(dna_sequences, logger, device)
        
    kmer_embedding = np.load("./helper/4mer_embedding.npy")
    # kmer_embedding = np.random.normal(size=(256, 100))
    
    embedding = np.dot(tnf_embedding, kmer_embedding)    
    
    return embedding


# def calculate_evo_embedding(dna_sequences, logger, device, model_name_or_path, model_max_length=400, batch_size=20):
#     if 'evo-1-131k-base' in model_name_or_path:
#         evo_model = Evo('evo-1-131k-base')
#     else:
#         evo_model = Evo('evo-1-8k-base')
#     model, tokenizer = evo_model.model, evo_model.tokenizer
#     model = model.half()
#     model.to(device)
#     model.eval()
#
#     lengths = [len(seq) for seq in dna_sequences]
#     idx = np.argsort(lengths)[::-1]
#     dna_sequences = [dna_sequences[i] for i in idx]
#
#     # Example batched inference.
#     train_loader = torch.utils.data.DataLoader(dna_sequences, batch_size=batch_size, shuffle=False, num_workers=16)
#     for j, batch_seqs in enumerate(tqdm.tqdm(train_loader, desc=f"Get embedding: ", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
#         with torch.inference_mode():
#             from evo.scoring import prepare_batch
#             input_ids, seq_lengths = prepare_batch(
#                 batch_seqs,
#                 tokenizer,
#                 prepend_bos=False,
#                 device=device,
#             )
#             model_output = model.forward(input_ids)[0].detach().cpu()
#             embedding = torch.sum(model_output, dim=1)
#             if j % 1000 == 0:
#                 if hasattr(torch.cuda, 'empty_cache'):
#                     torch.cuda.empty_cache()
#                 batch_lengths = [len(seq) for seq in batch_seqs]
#                 batch_max_length = np.max(batch_lengths)
#                 max_memory_allocated = round(torch.cuda.max_memory_allocated(device=device) / 1024 / 1024 / 1024, 2)
#                 total_memory = round(torch.cuda.get_device_properties(device=device).total_memory / 1024 / 1024 / 1024, 2)
#                 logger.info(f"batch_max_length: {batch_max_length}bp, max_memory_allocated: {max_memory_allocated}GB, total_memory: {total_memory}GB")
#
#             if j == 0:
#                 embeddings = embedding.detach().cpu()
#             else:
#                 embeddings = torch.cat((embeddings, embedding.detach().cpu()), dim=0)
#     embeddings = np.array(embeddings)
#
#     # reorder the embeddings
#     embeddings = embeddings[np.argsort(idx)]
#
#     return embeddings


def calculate_llm_embedding(dna_sequences, logger, device, model_name_or_path, model_max_length=400, batch_size=20):
    # reorder the sequences by length
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)[::-1]
    dna_sequences = [dna_sequences[i] for i in idx]

    logger.info(f"model_name_or_path: {model_name_or_path}")
    logger.info(f"model_max_length={model_max_length}, batch_size={batch_size}")
    # 这是一个通用的标记器类，当使用 AutoTokenizer.from_pretrained() 类方法创建时，它将被实例化为库的标记器类之一。
    # https://huggingface.co/docs/transformers/v4.46.2/model_doc/auto#transformers.AutoTokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    is_hyenadna = "hyenadna" in model_name_or_path
    is_nt = "nucleotide-transformer" in model_name_or_path
    
    if is_nt:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            revision="huggingface",
            trust_remote_code=True,
        )
    else:
        # 如果没有指定，则调用/public/home/jialh/metaHiC/LLMs/DNABERT-S/model下的模型，即DNABERT-S。
        # 核心代码位于/public/home/jialh/metaHiC/LLMs/DNABERT-S/model/bert_layers.py的第430行左右。
        # transformers.AutoModel.from_pretrained: https://huggingface.co/docs/transformers/model_doc/auto
        # 在许多情况下，您可以从提供给 from_pretrained() 方法的预训练模型的名称或路径中猜出要使用的架构。
        # AutoClasses 可以为您完成这项工作，以便您根据预训练权重/配置/词汇的名称/路径自动检索相关模型。
        # 实例化 AutoConfig、AutoModel 和 AutoTokenizer 之一将直接创建相关架构的类。
        # A path to a directory containing a configuration file saved using the save_pretrained() method,
        # or the save_pretrained() method, e.g., ./my_model_directory/.
        # https://github.com/huggingface/transformers/blob/v4.46.2/src/transformers/models/auto/configuration_auto.py#L916
        model = transformers.AutoModel.from_pretrained(
                model_name_or_path,
                revision="huggingface",
                trust_remote_code=True,
            )

    logger.info(f"type(model): {type(model)}")
    # logger.info(f"model: {model}")

    # n_gpu = torch.cuda.device_count()
    # if n_gpu > 1:
    #     model = nn.DataParallel(model)
        
    model.to(device)

    train_loader = torch.utils.data.DataLoader(dna_sequences, batch_size=batch_size, shuffle=False, num_workers=4)
    for j, batch in enumerate(tqdm.tqdm(train_loader,desc=f"Get embedding: ", bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
        # Model Inference Optimization Checklist: https://pytorch.org/serve/performance_checklist.html
        with torch.inference_mode():
            # 参考: https://blog.csdn.net/weixin_48030475/article/details/128688629
            # convert_tokens_to_ids，将token转化成id，在分词之后。
            # convert_ids_to_tokens,将id转化成token，通常用于模型预测出结果，查看时使用。
            # encode,进行分词和token转换，encode=tokenize+convert_tokens_to_ids。
            # encode_plus,在encode的基础之上生成input_ids、token_type_ids、attention_mask。
            # batch_encode_plus,在encode_plus的基础之上，能够批量梳理文本。
            batch_lengths = [len(seq) for seq in batch]
            batch_max_length = np.max(batch_lengths)
            token_feat = tokenizer.batch_encode_plus(
                    batch, 
                    max_length=batch_max_length,
                    return_tensors='pt', 
                    padding='longest', 
                    truncation=True
                )
            input_ids = token_feat['input_ids'].to(device)
            if is_hyenadna:
                model_output = model.forward(input_ids=input_ids)[0].detach().cpu()
                embedding = torch.sum(model_output, dim=1)
            else:
                # 这一步到底做了什么？有什么特点？如何理解？内存消耗如何估算？
                attention_mask = token_feat['attention_mask'].to(device)
                model_output = model.forward(input_ids=input_ids, attention_mask=attention_mask)[0].detach().cpu()
                attention_mask = attention_mask.unsqueeze(-1).detach().cpu()
                embedding = torch.sum(model_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

            if j % 1000 == 0:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                max_memory_allocated = round(torch.cuda.max_memory_allocated(device=device) / 1024 / 1024 / 1024, 2)
                total_memory = round(torch.cuda.get_device_properties(device=device).total_memory / 1024 / 1024 / 1024, 2)
                logger.info(f"batch_max_length: {batch_max_length}, max_memory_allocated: {max_memory_allocated}GB, total_memory: {total_memory}GB")
            
            if j == 0:
                embeddings = embedding.detach().cpu()
            else:
                embeddings = torch.cat((embeddings, embedding.detach().cpu()), dim=0)

    embeddings = np.array(embeddings)
    
    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings

def KMedoid(features,
            min_similarity=0.8,
            min_bin_size=100,
            max_iter=300):
    # rank nodes by the number of neighbors
    features = features.astype(np.float32)
    similarities = np.dot(features, features.T)

    # set the values below min_similarity to 0
    similarities[similarities < min_similarity] = 0
    row_sum = np.sum(similarities, axis=1)

    labels = np.ones(len(features)) * -1
    labels = labels.astype(int)
    count = 0

    while np.any(labels == -1):
        count += 1
        if count > max_iter:
            break
        # i = np.random.choice(np.where(labels == -1)[0])
        i = np.argmax(row_sum)
        # logger.info(f"i: {i} count: {count} row_sum: {row_sum[i]}")
        row_sum[i] = -100

        medoid = features[i]
        idx_within = np.zeros(len(features), dtype=bool)
        idx_available = labels == -1

        for _ in range(3):
            similarity = np.dot(features, medoid)
            idx_within = similarity >= min_similarity
            # idx_within = np.logical_or(idx_within, similarity >= min_similarity)
            idx = np.where(np.logical_and(idx_within, idx_available))[0]
            medoid = np.mean(features[idx], axis=0)

        # assign labels
        labels[idx] = count
        row_sum -= np.sum(similarities[:, idx], axis=1)
        row_sum[idx] = -100
        
    
    # remove bins that are too small
    unique, counts = np.unique(labels, return_counts=True)
    for i, c in zip(unique, counts):
        if c < min_bin_size:
            labels[labels == i] = -1
    
    return labels



def align_labels_via_hungarian_algorithm(true_labels, predicted_labels):
    """
    Aligns the predicted labels with the true labels using the Hungarian algorithm.

    Args:
    true_labels (list or array): The true labels of the data.
    predicted_labels (list or array): The labels predicted by a clustering algorithm.

    Returns:
    dict: A dictionary mapping the predicted labels to the aligned true labels.
    """
    # Create a confusion matrix
    max_label = max(max(true_labels), max(predicted_labels)) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

    # Create a mapping from predicted labels to true labels
    label_mapping = {predicted_label: true_label for true_label, predicted_label in zip(row_ind, col_ind)}

    return label_mapping


def compute_class_center_medium_similarity(embeddings, labels):
    idx = np.argsort(labels)
    embeddings = embeddings[idx]
    labels = labels[idx]
    n_sample_per_class = np.bincount(labels)
        
    all_similarities = np.zeros(len(embeddings))
    count = 0
    
    for i in range(len(n_sample_per_class)):
        start = count
        end = count + n_sample_per_class[i]
        mean = np.mean(embeddings[start:end], axis=0)
        similarities = np.dot(mean, embeddings[start:end].T).reshape(-1)
        
        all_similarities[start:end] = similarities
        
        count += n_sample_per_class[i]
    
    all_similarities.sort()
    percentile_values = []
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        value = all_similarities[int(percentile/100 * len(embeddings))]
        percentile_values.append(value)
    logger.info(percentile_values)
    return percentile_values



