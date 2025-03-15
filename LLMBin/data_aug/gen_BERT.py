import os
import subprocess
import pickle
import math
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from .BERT_utils import get_embedding, KMedoid, align_labels_via_hungarian_algorithm, compute_class_center_medium_similarity


def count_fasta_sequences(file_name):
    """
    Estimate the number of fasta sequences in a file by counting headers. Decompression is automatically attempted
    for files ending in .gz. Counting and decompression is by why of subprocess calls to grep and gzip. Uncompressed
    files are also handled. This is about 8 times faster than parsing a file with BioPython and 6 times faster
    than reading all lines in Python.

    :param file_name: the fasta file to inspect
    :return: the estimated number of records
    """
    if file_name.endswith('.gz'):
        proc_uncomp = subprocess.Popen(['gzip', '-cd', file_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        proc_read = subprocess.Popen(['grep', r'^>'], stdin=proc_uncomp.stdout, stdout=subprocess.PIPE)
    else:
        proc_read = subprocess.Popen(['grep', r'^>', file_name], stdout=subprocess.PIPE)

    n = 0
    for _ in proc_read.stdout:
        n += 1
    return n


def run_gen_BERT(contigfile, sample, model, species, test_model_dir, outdir, logger, device, llm_batch_size=16, model_max_length=2000, contig_max_length=10000):
    ###### load binning data
    fasta_count = count_fasta_sequences(contigfile)
    input_file = contigfile + f'_{contig_max_length}bp.pkl'
    if not os.path.exists(input_file):
        contig_names = []
        dna_sequences = []
        for record in tqdm(SeqIO.parse(contigfile, "fasta"), total=fasta_count, desc='Read sequences from fasta file'):
            step = max(int(0.5 * contig_max_length), 2000)
            if len(record.seq) > contig_max_length:
                for i, start in enumerate(range(0, len(record.seq), step)):
                    if start + contig_max_length < len(record.seq):
                        contig_names.append(record.id + '_' + str(i))
                        dna_sequences.append(str(record.seq)[start:start+contig_max_length])
                    # elif len(record.seq) - start > 0.5 * contig_max_length:
                    elif len(record.seq) - start > (0.5 * contig_max_length):
                        contig_names.append(record.id + '_' + str(i))
                        dna_sequences.append(str(record.seq)[start:len(record.seq)])
            else:
                contig_names.append(record.id + '_1')
                dna_sequences.append(str(record.seq))
        logger.info(f"write contig info to {input_file}")
        with open(input_file, "wb") as output_pkl:
            pickle.dump((contig_names, dna_sequences), output_pkl)
    else:
        logger.info(f"read contig info from {input_file}")
        with open(input_file, "rb") as input_pkl:
            contig_names, dna_sequences = pickle.load(input_pkl)

    logger.info(f"Get {len(dna_sequences)} sequences")

    # generate embedding
    logger.info(f"Begin to get embeddings by {model} with contig_max_length={contig_max_length}")
    embedding_file = os.path.join(outdir, f"{sample}_{model}_{contig_max_length}bp_embedding.pkl")
    if not os.path.exists(embedding_file):
        embedding = get_embedding(dna_sequences, model, species, sample, outdir,
                                  contig_max_length=contig_max_length,
                                  model_max_length=model_max_length,
                                  logger=logger,
                                  device=device,
                                  task_name="binning",
                                  batch_size=llm_batch_size,
                                  test_model_dir=test_model_dir)
        # if len(embedding) > len(filterd_idx):
        #     embedding = embedding[np.array(filterd_idx)]
        with open(embedding_file, "wb") as embedding_out:
            pickle.dump(embedding, embedding_out)
    else:
        with open(embedding_file, 'rb') as embedding_in:
            embedding = pickle.load(embedding_in)

    logger.info(f"embedding.shape: {embedding.shape}")
    embedding_df = pd.DataFrame(data=embedding, index=contig_names)
    print(f"contig_names[0:20]: {contig_names[0:20]}")
    embedding_df.reset_index(inplace=True)
    embedding_df.rename(columns={"index": "contig_name"}, inplace=True)
    embedding_df[['contig', 'num']] = embedding_df['contig_name'].str.rsplit(pat="_", n=1, expand=True)
    embedding_df.drop(columns=['contig_name', 'num'], inplace=True)
    embedding_df_sum = embedding_df.groupby('contig').sum()
    print(f"embedding_df_sum.iloc[0:5, 0:5]: {embedding_df_sum.iloc[0:5, 0:5]}")
    print(f"embedding_df_sum.index.values.tolist()[0:5]: {embedding_df_sum.index.values.tolist()[0:5]}")
    # embedding_df_sum.set_index("contig", inplace=True)
    print(f"embedding_df_sum.shape: {embedding_df_sum.shape}")
    embedding_tsv = os.path.join(outdir, f"{sample}_{model}_{model_max_length}bp_embedding_sum.tsv")
    if not os.path.exists(embedding_tsv):
        embedding_df_sum.to_csv(embedding_tsv, sep="\t", index=True, header=True, index_label="contig")
    os.system("head " + embedding_tsv + " | awk -F '\t' '{print $1,$2,$3,$4,$5}'")