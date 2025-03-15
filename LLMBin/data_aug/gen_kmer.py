#!/usr/bin/env python
from __future__ import print_function
import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from Bio import SeqIO
import subprocess
# optimized sliding window function from
# http://stackoverflow.com/a/7636587
from itertools import tee
from collections import Counter, OrderedDict
import pandas as pd


def window(seq,n):
    els = tee(seq,n)
    for i,el in enumerate(els):
        for _ in range(i):
            next(el, None)
    return zip(*els)

def generate_feature_mapping(kmer_len):
    BASE_COMPLEMENT = {"A":"T","T":"A","G":"C","C":"G"}
    kmer_hash = {}
    counter = 0
    for kmer in product("ATGC",repeat=kmer_len):
        if kmer not in kmer_hash:
            kmer_hash[kmer] = counter
            rev_compl = tuple([BASE_COMPLEMENT[x] for x in reversed(kmer)])
            kmer_hash[rev_compl] = counter
            counter += 1
    return kmer_hash,counter

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

def generate_features_from_fasta(fasta_file: str, length_threshold: int, kmer_len: int, outfile: str):
    """
    Generate composition features from a FASTA file.

    :param fasta_file: The path to the input FASTA file.
    :param length_threshold: The minimum length of sequences to include in the feature generation.
    :param kmer_len: The length of k-mers to consider.
    :param outfile: The path to the output CSV file where features will be saved.
    """
    kmer_dict, nr_features = generate_feature_mapping(kmer_len)
    fasta_count = count_fasta_sequences(fasta_file)

    # Store composition vectors in a dictionary before creating dataframe
    composition_d = OrderedDict()
    contig_lengths = OrderedDict()
    for seq in tqdm(SeqIO.parse(fasta_file,"fasta"), total=fasta_count, desc="generate kmers"):
        seq_len = len(seq)
        if seq_len <= length_threshold:
            continue
        contig_lengths[seq.id] = seq_len
        # Create a list containing all kmers, translated to integers
        kmers = [
            kmer_dict[kmer_tuple]
            for kmer_tuple
            in window(str(seq.seq).upper(), kmer_len)
            if kmer_tuple in kmer_dict
        ]
        kmers.append(nr_features-1)
        composition_v = np.bincount(np.array(kmers,dtype=np.int64))
        composition_v[-1]-=1
        composition_d[seq.id] = composition_v 
    df = pd.DataFrame.from_dict(composition_d, orient='index', dtype=float)
    df.to_csv(outfile)


def run_gen_kmer(fasta_file, length_threshold, kmer_len):
    outfile = os.path.join(os.path.dirname(fasta_file), 'kmer_' + str(kmer_len) + '_f' + str(length_threshold) + '.csv')
    generate_features_from_fasta(fasta_file,length_threshold,kmer_len,outfile)

