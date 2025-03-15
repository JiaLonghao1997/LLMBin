import gzip
import os
import shutil
import urllib
from pathlib import Path
from typing import List
from tqdm import tqdm
from ast import literal_eval

import re
import datasets
import numpy as np
import pandas as pd
from datasets import DatasetInfo
from pyfaidx import Fasta
from abc import ABC, abstractmethod


"""
----------------------------------------------------------------------------------------
Reference Genome URLS:
----------------------------------------------------------------------------------------
"""
H38_REFERENCE_GENOME_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/" "hg38.fa.gz"
)
H19_REFERENCE_GENOME_URL = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/" "hg19.fa.gz"
)

"""
----------------------------------------------------------------------------------------
Task Specific Handlers:
----------------------------------------------------------------------------------------
"""

class GenomicLRATaskHandler(ABC):
    """
    Abstract method for the Genomic LRA task handlers.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def get_info(self, description: str) -> DatasetInfo:
        """
        Returns the DatasetInfo for the task
        """
        pass

    def split_generators(
            self, dl_manager, cache_dir_root
    ) -> List[datasets.SplitGenerator]:
        """
        Downloads required files using dl_manager and separates them by split.
        """
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"handler": self, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"handler": self, "split": "test"}
            ),
        ]

    @abstractmethod
    def generate_examples(self, split):
        """
        A generator that yields examples for the specified split.
        """
        pass

    @staticmethod
    def hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            """
            b  : int, optional
                Number of blocks just transferred [default: 1].
            bsize  : int, optional
                Size of each block (in tqdm units) [default: 1].
            tsize  : int, optional
                Total size (in tqdm units). If [default: None] remains unchanged.
            """
            if tsize is not None:
                t.total = tsize
            t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    def download_and_extract_gz(self, file_url, cache_dir_root):
        """
        Downloads and extracts a gz file into the given cache directory. Returns the
        full file path of the extracted gz file.
        Args:
            file_url: url of the gz file to be downloaded and extracted.
            cache_dir_root: Directory to extract file into.
        """
        file_fname = Path(file_url).stem
        file_complete_path = os.path.join(cache_dir_root, "downloads", file_fname)

        if not os.path.exists(file_complete_path):
            if not os.path.exists(file_complete_path + ".gz"):
                with tqdm(
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        miniters=1,
                        desc=file_url.split("/")[-1],
                ) as t:
                    urllib.request.urlretrieve(
                        file_url, file_complete_path + ".gz", reporthook=self.hook(t)
                    )
            with gzip.open(file_complete_path + ".gz", "rb") as file_in:
                with open(file_complete_path, "wb") as file_out:
                    shutil.copyfileobj(file_in, file_out)
        return file_complete_path


class CagePredictionHandler(GenomicLRATaskHandler):
    """
    Handler for the CAGE prediction task.
    """

    NUM_TRAIN = 33891
    NUM_TEST = 1922
    NUM_VALID = 2195
    DEFAULT_LENGTH = 114688  # 896 x 128bp
    TARGET_SHAPE = (
        896,
        50,
    )  # 50 is a subset of CAGE tracks from the original enformer dataset
    NPZ_SPLIT = 1000  # number of files per npz file.
    NUM_BP_PER_BIN = 128  # number of base pairs per bin in labels

    def __init__(self, sequence_length=DEFAULT_LENGTH, **kwargs):
        """
        Creates a new handler for the CAGE task.
        Args:
            sequence_length: allows for increasing sequence context. Sequence length
            must be an even multiple of 128 to align with binned labels. Note:
            increasing sequence length may decrease the number of usable samples.
        """
        self.reference_genome = None
        self.coordinate_csv_file = None
        self.target_files_by_split = {}


        assert (sequence_length // 128) % 2 == 0, (
            f"Requested sequence length must be an even multuple of 128 to align "
            f"with the binned labels."
        )

        self.sequence_length = sequence_length

        if self.sequence_length < self.DEFAULT_LENGTH:
            self.TARGET_SHAPE = (self.sequence_length // 128, 50)

    def get_info(self, description: str) -> DatasetInfo:
        """
        Returns the DatasetInfo for the CAGE dataset. Each example
        includes a genomic sequence and a 2D array of labels
        """
        features = datasets.Features(
            {
                # DNA sequence
                "sequence": datasets.Value("string"),
                # array of sequence length x num_labels
                "labels": datasets.Array2D(shape=self.TARGET_SHAPE, dtype="float32"),
                # chromosome number
                "chromosome": datasets.Value(dtype="string"),
                # start
                "labels_start": datasets.Value(dtype="int32"),
                # stop
                "labels_stop": datasets.Value(dtype="int32")
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=description,
            # This defines the different columns of the dataset and their types
            features=features,
        )

    def split_generators(self, dl_manager, cache_dir_root):
        """
        Separates files by split and stores filenames in instance variables.
        The CAGE dataset requires reference genome, coordinate
        csv file,and npy files to be saved.
        """

        # Manually download the reference genome since there are difficulties when
        # streaming
        reference_genome_file = self.download_and_extract_gz(
            H38_REFERENCE_GENOME_URL, cache_dir_root
        )
        self.reference_genome = Fasta(reference_genome_file, one_based_attributes=False)

        self.coordinate_csv_file = dl_manager.download_and_extract(
            "cage_prediction/sequences_coordinates.csv"
        )

        train_file_dict = {}
        for train_key, train_file in self.generate_npz_filenames(
                "train", self.NUM_TRAIN, folder="cage_prediction/targets_subset"
        ):
            train_file_dict[train_key] = dl_manager.download(train_file)

        test_file_dict = {}
        for test_key, test_file in self.generate_npz_filenames(
                "test", self.NUM_TEST, folder="cage_prediction/targets_subset"
        ):
            test_file_dict[test_key] = dl_manager.download(test_file)

        valid_file_dict = {}
        for valid_key, valid_file in self.generate_npz_filenames(
                "valid", self.NUM_VALID, folder="cage_prediction/targets_subset"
        ):
            valid_file_dict[valid_key] = dl_manager.download(valid_file)

        # convert file list to a dict keyed by target number
        self.target_files_by_split["train"] = train_file_dict
        self.target_files_by_split["test"] = test_file_dict
        self.target_files_by_split["validation"] = valid_file_dict

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"handler": self, "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"handler": self, "split": "validation"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"handler": self, "split": "test"}
            ),
        ]

    def generate_examples(self, split):
        """
        A generator which produces examples for the given split, each with a sequence
        and the corresponding labels. The sequences are padded to the correct
        sequence length and standardized before returning.
        """

        target_files = self.target_files_by_split[split]

        key = 0
        coordinates_dataframe = pd.read_csv(self.coordinate_csv_file)
        filtered = coordinates_dataframe[coordinates_dataframe["split"] == split]
        for sequential_idx, row in filtered.iterrows():
            start, stop = int(row["start"]) - 1, int(
                row["stop"]) - 1  # -1 since coords are 1-based

            chromosome = row['chrom']

            padded_sequence,new_start,new_stop = pad_sequence(
                chromosome=self.reference_genome[chromosome],
                start=start,
                sequence_length=self.sequence_length,
                end=stop,
                return_new_start_stop=True
            )

            if self.sequence_length >= self.DEFAULT_LENGTH:
                new_start = start
                new_stop = stop

            # floor npy_idx to the nearest 1000
            npz_file = np.load(
                target_files[int((row["npy_idx"] // self.NPZ_SPLIT) * self.NPZ_SPLIT)]
            )

            if (
                    split == "validation"
            ):  # npy files are keyed by ["train", "test", "valid"]
                split = "valid"
            targets = npz_file[f"target-{split}-{row['npy_idx']}.npy"][
                0]  # select 0 since there is extra dimension

            # subset the targets if sequence length is smaller than 114688 (
            # DEFAULT_LENGTH)
            if self.sequence_length < self.DEFAULT_LENGTH:
                idx_diff = (self.DEFAULT_LENGTH - self.sequence_length) // 2 // 128
                targets = targets[idx_diff:-idx_diff]

            if padded_sequence:
                yield key, {
                    "labels": targets,
                    "sequence": standardize_sequence(padded_sequence),
                    "chromosome": re.sub("chr", "", chromosome),
                    "labels_start": new_start,
                    "labels_stop": new_stop
                }
                key += 1

    @staticmethod
    def generate_npz_filenames(split, total, folder, npz_size=NPZ_SPLIT):
        """
        Generates a list of filenames for the npz files stored in the dataset.
        Yields a tuple of floored multiple of 1000, filename
        Args:
            split: split to generate filenames for. Must be in ['train', 'test', 'valid']
                due to the naming of the files.
            total: total number of npy targets for given split
            folder: folder where data is stored.
            npz_size: number of npy files per npz. Defaults to 1000 because
                this is the number currently used in the dataset.
        """

        for i in range(total // npz_size):
            yield i * npz_size, f"{folder}/targets-{split}-{i * npz_size}-{i * npz_size + (npz_size - 1)}.npz"
        if total % npz_size != 0:
            yield (
                npz_size * (total // npz_size),
                f"{folder}/targets-{split}-"
                f"{npz_size * (total // npz_size)}-"
                f"{npz_size * (total // npz_size) + (total % npz_size - 1)}.npz",
            )


class BulkRnaExpressionHandler(GenomicLRATaskHandler):
    """
    Handler for the Bulk RNA Expression task.
    """

    DEFAULT_LENGTH = 100000

    def __init__(self, sequence_length=DEFAULT_LENGTH, **kwargs):
        """
        Creates a new handler for the Bulk RNA Expression Prediction Task.
        Args:
            sequence_length: Length of the sequence around the TSS_CAGE start site

        """
        self.reference_genome = None
        self.coordinate_csv_file = None
        self.labels_csv_file = None
        self.sequence_length = sequence_length

    def get_info(self, description: str) -> DatasetInfo:
        """
        Returns the DatasetInfo for the Bulk RNA Expression dataset. Each example
        includes a genomic sequence and a list of label values.
        """
        features = datasets.Features(
            {
                # DNA sequence
                "sequence": datasets.Value("string"),
                # list of expression values in each tissue
                "labels": datasets.Sequence(datasets.Value("float32")),
                # chromosome number
                "chromosome": datasets.Value(dtype="string"),
                # position
                "position": datasets.Value(dtype="int32"),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=description,
            # This defines the different columns of the dataset and their types
            features=features,

        )

    def split_generators(self, dl_manager, cache_dir_root):
        """
        Separates files by split and stores filenames in instance variables.
        The Bulk RNA Expression dataset requires the reference hg19 genome, coordinate
        csv file,and label csv file to be saved.
        """

        reference_genome_file = self.download_and_extract_gz(
            H19_REFERENCE_GENOME_URL, cache_dir_root
        )
        self.reference_genome = Fasta(reference_genome_file, one_based_attributes=False)

        self.coordinate_csv_file = dl_manager.download_and_extract(
            "bulk_rna_expression/gene_coordinates.csv"
        )

        self.labels_csv_file = dl_manager.download_and_extract(
            "bulk_rna_expression/rna_expression_values.csv"
        )

        return super().split_generators(dl_manager, cache_dir_root)

    def generate_examples(self, split):
        """
        A generator which produces examples for the given split, each with a sequence
        and the corresponding labels. The sequences are padded to the correct sequence
        length and standardized before returning.
        """
        coordinates_df = pd.read_csv(self.coordinate_csv_file)
        labels_df = pd.read_csv(self.labels_csv_file)

        coordinates_split_df = coordinates_df[coordinates_df["split"] == split]

        key = 0
        for idx, coordinates_row in coordinates_split_df.iterrows():
            start = coordinates_row[
                        "CAGE_representative_TSS"] - 1  # -1 since coords are 1-based

            chromosome = coordinates_row["chrom"]
            labels_row = labels_df.loc[idx].values
            padded_sequence = pad_sequence(
                chromosome=self.reference_genome[chromosome],
                start=start,
                sequence_length=self.sequence_length,
                negative_strand=coordinates_row["strand"] == "-",
            )
            if padded_sequence:
                yield key, {
                    "labels": labels_row,
                    "sequence": standardize_sequence(padded_sequence),
                    "chromosome": re.sub("chr", "", chromosome),
                    "position": coordinates_row["CAGE_representative_TSS"]
                }
                key += 1


class VariantEffectCausalEqtl(GenomicLRATaskHandler):
    """
    Handler for the Variant Effect Causal eQTL task.
    """

    DEFAULT_LENGTH = 100000

    def __init__(self, sequence_length=DEFAULT_LENGTH, **kwargs):
        """
        Creates a new handler for the Variant Effect Causal eQTL Task.
        Args:
            sequence_length: Length of the sequence to pad around the SNP position

        """
        self.reference_genome = None
        self.sequence_length = sequence_length

    def get_info(self, description: str) -> DatasetInfo:
        """
        Returns the DatasetInfo for the Variant Effect Causal eQTL dataset. Each example
        includes a  genomic sequence with the reference allele as well as the genomic
        sequence with the alternative allele, and a binary label.
        """
        features = datasets.Features(
            {
                # DNA sequence
                "ref_forward_sequence": datasets.Value("string"),
                "alt_forward_sequence": datasets.Value("string"),
                # binary label
                "label": datasets.Value(dtype="int8"),
                # tissue type
                "tissue": datasets.Value(dtype="string"),
                # chromosome number
                "chromosome": datasets.Value(dtype="string"),
                # variant position
                "position": datasets.Value(dtype="int32"),
                # distance to nearest tss
                "distance_to_nearest_tss": datasets.Value(dtype="int32")
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=description,
            # This defines the different columns of the dataset and their types
            features=features,
        )

    def split_generators(self, dl_manager, cache_dir_root):
        """
        Separates files by split and stores filenames in instance variables.
        The variant effect prediction dataset requires the reference hg38 genome and
        coordinates_labels_csv_file to be saved.
        """

        # Manually download the reference genome since there are difficulties
        # when streaming
        reference_genome_file = self.download_and_extract_gz(
            H38_REFERENCE_GENOME_URL, cache_dir_root
        )

        self.reference_genome = Fasta(reference_genome_file, one_based_attributes=False)
        self.coordinates_labels_csv_file = dl_manager.download_and_extract(
            f"variant_effect_causal_eqtl/All_Tissues.csv"
        )

        return super().split_generators(dl_manager, cache_dir_root)

    def generate_examples(self, split):
        """
        A generator which produces examples each with ref/alt allele
        and corresponding binary label. The sequences are extended to
        the desired sequence length and standardized before returning.
        """

        coordinates_df = pd.read_csv(self.coordinates_labels_csv_file)

        coordinates_split_df = coordinates_df[coordinates_df["split"] == split]

        key = 0
        for idx, row in coordinates_split_df.iterrows():
            start = row["POS"] - 1  # sub 1 to create idx since coords are 1-based
            alt_allele = row["ALT"]
            label = row["label"]
            tissue = row['tissue']
            chromosome = row["CHROM"]
            distance = int(row["distance_to_nearest_TSS"])

            # get reference forward sequence
            ref_forward = pad_sequence(
                chromosome=self.reference_genome[chromosome],
                start=start,
                sequence_length=self.sequence_length,
                negative_strand=False,
            )

            # only if a valid sequence returned
            if ref_forward:
                # Mutate sequence with the alt allele at the SNP position,
                # which is always centered in the string returned from pad_sequence
                alt_forward = list(ref_forward)
                alt_forward[self.sequence_length // 2] = alt_allele
                alt_forward = "".join(alt_forward)

                yield key, {
                    "label": label,
                    "tissue": tissue,
                    "chromosome": re.sub("chr", "", chromosome),
                    "ref_forward_sequence": standardize_sequence(ref_forward),
                    "alt_forward_sequence": standardize_sequence(alt_forward),
                    "distance_to_nearest_tss": distance,
                    "position": row["POS"]
                }
                key += 1


class VariantEffectPathogenicHandler(GenomicLRATaskHandler):
    """
    Handler for the Variant Effect Pathogenic Prediction tasks.
    """

    DEFAULT_LENGTH = 100000

    def __init__(self, sequence_length=DEFAULT_LENGTH, task_name=None, subset=False,
                 **kwargs):
        """
        Creates a new handler for the Variant Effect Pathogenic Tasks.
        Args:
            sequence_length: Length of the sequence to pad around the SNP position
            subset: Whether to return a pre-determined subset of the data.

        """
        self.sequence_length = sequence_length

        if task_name == 'variant_effect_pathogenic_clinvar':
            self.data_file_name = "variant_effect_pathogenic/vep_pathogenic_coding.csv"
        elif task_name == 'variant_effect_pathogenic_omim':
            self.data_file_name = "variant_effect_pathogenic/" \
                                  "vep_pathogenic_non_coding_subset.csv" \
                if subset else "variant_effect_pathogenic/vep_pathogenic_non_coding.csv"

    def get_info(self, description: str) -> DatasetInfo:
        """
        Returns the DatasetInfo for the Variant Effect Pathogenic datasets. Each example
        includes a  genomic sequence with the reference allele as well as the genomic
        sequence with the alternative allele, and a binary label.
        """
        features = datasets.Features(
            {
                # DNA sequence
                "ref_forward_sequence": datasets.Value("string"),
                "alt_forward_sequence": datasets.Value("string"),
                # binary label
                "label": datasets.Value(dtype="int8"),
                # chromosome number
                "chromosome": datasets.Value(dtype="string"),
                # position
                "position": datasets.Value(dtype="int32")
            }
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=description,
            # This defines the different columns of the dataset and their types
            features=features,
        )

    def split_generators(self, dl_manager, cache_dir_root):
        """
        Separates files by split and stores filenames in instance variables.
        The variant effect prediction datasets require the reference hg38 genome and
        coordinates_labels_csv_file to be saved.
        """

        reference_genome_file = self.download_and_extract_gz(
            H38_REFERENCE_GENOME_URL, cache_dir_root
        )

        self.reference_genome = Fasta(reference_genome_file, one_based_attributes=False)
        self.coordinates_labels_csv_file = dl_manager.download_and_extract(
            self.data_file_name)

        if 'non_coding' in self.data_file_name:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"handler": self, "split": "test"}
                ), ]
        else:
            return super().split_generators(dl_manager, cache_dir_root)

    def generate_examples(self, split):
        """
        A generator which produces examples each with ref/alt allele
        and corresponding binary label. The sequences are extended to
        the desired sequence length and standardized before returning.
        """

        coordinates_df = pd.read_csv(self.coordinates_labels_csv_file)
        coordinates_split_df = coordinates_df[coordinates_df["split"] == split]

        key = 0
        for idx, row in coordinates_split_df.iterrows():
            start = row["POS"] - 1  # sub 1 to create idx since coords are 1-based
            alt_allele = row["ALT"]
            label = row["INT_LABEL"]
            chromosome = row["CHROM"]

            # get reference forward sequence
            ref_forward = pad_sequence(
                chromosome=self.reference_genome[chromosome],
                start=start,
                sequence_length=self.sequence_length,
                negative_strand=False,
            )

            # only if a valid sequence returned
            if ref_forward:
                # Mutate sequence with the alt allele at the SNP position,
                # which is always centered in the string returned from pad_sequence
                alt_forward = list(ref_forward)
                alt_forward[self.sequence_length // 2] = alt_allele
                alt_forward = "".join(alt_forward)

                yield key, {
                    "label": label,
                    "chromosome": re.sub("chr", "", chromosome),
                    "ref_forward_sequence": standardize_sequence(ref_forward),
                    "alt_forward_sequence": standardize_sequence(alt_forward),
                    "position": row['POS']
                }
                key += 1


class ChromatinFeaturesHandler(GenomicLRATaskHandler):
    """
    Handler for the histone marks and DNA accessibility tasks also referred to
    collectively as Chromatin features.
    """

    DEFAULT_LENGTH = 100000

    def __init__(self, task_name=None, sequence_length=DEFAULT_LENGTH, subset=False,
                 **kwargs):
        """
        Creates a new handler for the Deep Sea Histone and DNase tasks.
        Args:
            sequence_length: Length of the sequence around and including the
            annotated 200bp bin
            subset: Whether to return a pre-determined subset of the entire dataset.

        """
        self.sequence_length = sequence_length

        if sequence_length < 200:
            raise ValueError(
                'Sequence length for this task must be greater or equal to 200 bp')

        if 'histone' in task_name:
            self.label_name = 'HISTONES'
        elif 'dna' in task_name:
            self.label_name = 'DNASE'

        self.data_file_name = "chromatin_features/histones_and_dnase_subset.csv" if \
            subset else "chromatin_features/histones_and_dnase.csv"

    def get_info(self, description: str) -> DatasetInfo:
        """
        Returns the DatasetInfo for the histone marks and dna accessibility datasets.
        Each example includes a genomic sequence and a list of label values.
        """
        features = datasets.Features(
            {
                # DNA sequence
                "sequence": datasets.Value("string"),
                # list of binary chromatin marks
                "labels": datasets.Sequence(datasets.Value("int8")),
                # chromosome number
                "chromosome": datasets.Value(dtype="string"),
                # starting position in genome which corresponds to label
                "label_start": datasets.Value(dtype="int32"),
                # end position in genome which corresponds to label
                "label_stop": datasets.Value(dtype="int32"),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=description,
            # This defines the different columns of the dataset and their types
            features=features,

        )

    def split_generators(self, dl_manager, cache_dir_root):
        """
        Separates files by split and stores filenames in instance variables.
        The histone marks and dna accessibility datasets require the reference hg19
        genome and coordinate csv file to be saved.
        """
        reference_genome_file = self.download_and_extract_gz(
            H19_REFERENCE_GENOME_URL, cache_dir_root
        )
        self.reference_genome = Fasta(reference_genome_file, one_based_attributes=False)

        self.coordinate_csv_file = dl_manager.download_and_extract(self.data_file_name)

        return super().split_generators(dl_manager, cache_dir_root)

    def generate_examples(self, split):
        """
        A generator which produces examples for the given split, each with a sequence
        and the corresponding labels. The sequences are padded to the correct sequence
        length and standardized before returning.
        """
        coordinates_df = pd.read_csv(self.coordinate_csv_file)
        coordinates_split_df = coordinates_df[coordinates_df["split"] == split]

        key = 0
        for idx, coordinates_row in coordinates_split_df.iterrows():
            start = coordinates_row['POS'] - 1  # -1 since saved coords are 1-based
            chromosome = coordinates_row["CHROM"]

            # literal eval used since lists are saved as strings in csv
            labels_row = literal_eval(coordinates_row[self.label_name])

            padded_sequence = pad_sequence(
                chromosome=self.reference_genome[chromosome],
                start=start,
                sequence_length=self.sequence_length,
            )
            if padded_sequence:
                yield key, {
                    "labels": labels_row,
                    "sequence": standardize_sequence(padded_sequence),
                    "chromosome": re.sub("chr", "", chromosome),
                    "label_start": coordinates_row['POS']-100,
                    "label_stop": coordinates_row['POS'] + 99,
                }
                key += 1


class RegulatoryElementHandler(GenomicLRATaskHandler):
    """
    Handler for the Regulatory Element Prediction tasks.
    """
    DEFAULT_LENGTH = 100000

    def __init__(self, task_name=None, sequence_length=DEFAULT_LENGTH, subset=False,
                 **kwargs):
        """
        Creates a new handler for the Regulatory Element Prediction tasks.
        Args:
            sequence_length: Length of the sequence around the element/non-element
            subset: Whether to return a pre-determined subset of the entire dataset.

        """

        if sequence_length < 200:
            raise ValueError(
                'Sequence length for this task must be greater or equal to 200 bp')

        self.sequence_length = sequence_length

        if 'promoter' in task_name:
            self.data_file_name = 'regulatory_elements/promoter_dataset'

        elif 'enhancer' in task_name:
            self.data_file_name = 'regulatory_elements/enhancer_dataset'

        if subset:
            self.data_file_name += '_subset.csv'
        else:
            self.data_file_name += '.csv'

    def get_info(self, description: str) -> DatasetInfo:
        """
        Returns the DatasetInfo for the Regulatory Element Prediction Tasks.
        Each example includes a genomic sequence and a label.
        """
        features = datasets.Features(
            {
                # DNA sequence
                "sequence": datasets.Value("string"),
                # label corresponding to whether the sequence has
                # the regulatory element of interest or not
                "labels": datasets.Value("int8"),
                # chromosome number
                "chromosome": datasets.Value(dtype="string"),
                # start
                "label_start": datasets.Value(dtype="int32"),
                # stop
                "label_stop": datasets.Value(dtype="int32"),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=description,
            # This defines the different columns of the dataset and their types
            features=features,

        )

    def split_generators(self, dl_manager, cache_dir_root):
        """
        Separates files by split and stores filenames in instance variables.
        """
        reference_genome_file = self.download_and_extract_gz(
            H38_REFERENCE_GENOME_URL, cache_dir_root
        )
        self.reference_genome = Fasta(reference_genome_file, one_based_attributes=False)

        self.coordinate_csv_file = dl_manager.download_and_extract(
            self.data_file_name
        )

        return super().split_generators(dl_manager, cache_dir_root)

    def generate_examples(self, split):
        """
        A generator which produces examples for the given split, each with a sequence
        and the corresponding label. The sequences are padded to the correct sequence
        length and standardized before returning.
        """
        coordinates_df = pd.read_csv(self.coordinate_csv_file)

        coordinates_split_df = coordinates_df[coordinates_df["split"] == split]

        key = 0
        for _, coordinates_row in coordinates_split_df.iterrows():
            start = coordinates_row["START"] - 1  # -1 since vcf coords are 1-based
            end = coordinates_row["STOP"] - 1  # -1 since vcf coords are 1-based
            chromosome = coordinates_row["CHROM"]

            label = coordinates_row['label']

            padded_sequence = pad_sequence(
                chromosome=self.reference_genome[chromosome],
                start=start,
                end=end,
                sequence_length=self.sequence_length,
            )

            if padded_sequence:
                yield key, {
                    "labels": label,
                    "sequence": standardize_sequence(padded_sequence),
                    "chromosome": re.sub("chr", "", chromosome),
                    "label_start": coordinates_row["START"],
                    "label_stop": coordinates_row["STOP"]
                }
                key += 1


"""
----------------------------------------------------------------------------------------
Dataset loader:
----------------------------------------------------------------------------------------
"""

_DESCRIPTION = """
Dataset for benchmark of genomic deep learning models. 
"""

_TASK_HANDLERS = {
    "cage_prediction": CagePredictionHandler,
    "bulk_rna_expression": BulkRnaExpressionHandler,
    "variant_effect_causal_eqtl": VariantEffectCausalEqtl,
    "variant_effect_pathogenic_clinvar": VariantEffectPathogenicHandler,
    "variant_effect_pathogenic_omim": VariantEffectPathogenicHandler,
    "chromatin_features_histone_marks": ChromatinFeaturesHandler,
    "chromatin_features_dna_accessibility": ChromatinFeaturesHandler,
    "regulatory_element_promoter": RegulatoryElementHandler,
    "regulatory_element_enhancer": RegulatoryElementHandler,
}


# define dataset configs
class GenomicsLRAConfig(datasets.BuilderConfig):
    """
    BuilderConfig.
    """

    def __init__(self, *args, task_name: str, **kwargs):  # type: ignore
        """BuilderConfig for the location tasks dataset.
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__()
        self.handler = _TASK_HANDLERS[task_name](task_name=task_name, **kwargs)


# DatasetBuilder
class GenomicsLRATasks(datasets.GeneratorBasedBuilder):
    """
    Tasks to annotate human genome.
    """

    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIG_CLASS = GenomicsLRAConfig

    def _info(self) -> DatasetInfo:
        return self.config.handler.get_info(description=_DESCRIPTION)

    def _split_generators(
            self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """
        Downloads data files and organizes it into train/test/val splits
        """
        return self.config.handler.split_generators(dl_manager, self._cache_dir_root)

    def _generate_examples(self, handler, split):
        """
        Read data files and create examples(yield)
        Args:
            handler: The handler for the current task
            split: A string in ['train', 'test', 'valid']
        """
        yield from handler.generate_examples(split)


"""
----------------------------------------------------------------------------------------
Global Utils:
----------------------------------------------------------------------------------------
"""


def standardize_sequence(sequence: str):
    """
    Standardizes the sequence by replacing all unknown characters with N and
    converting to all uppercase.
    Args:
        sequence: genomic sequence to standardize
    """
    pattern = "[^ATCG]"
    # all characters to upper case
    sequence = sequence.upper()
    # replace all characters that are not A,T,C,G with N
    sequence = re.sub(pattern, "N", sequence)
    return sequence


def pad_sequence(chromosome, start, sequence_length, end=None, negative_strand=False,
                 return_new_start_stop=False):
    """
    Extends a given sequence to length sequence_length. If
    padding to the given length is outside the gene, returns
    None.
    Args:
        chromosome: Chromosome from pyfaidx extracted Fasta.
        start: Start index of original sequence.
        sequence_length: Desired sequence length. If sequence length is odd, the
            remainder is added to the end of the sequence.
        end: End index of original sequence. If no end is specified, it creates a
            centered sequence around the start index.
        negative_strand: If negative_strand, returns the reverse compliment of the
        sequence
    """
    if end:
        pad = (sequence_length - (end - start)) // 2
        start = start - pad
        end = end + pad + (sequence_length % 2)
    else:
        pad = sequence_length // 2
        end = start + pad + (sequence_length % 2)
        start = start - pad

    if start < 0 or end >= len(chromosome):
        return
    if negative_strand:
        if return_new_start_stop:
            return chromosome[start:end].reverse.complement.seq ,start, end

        return chromosome[start:end].reverse.complement.seq

    if return_new_start_stop:
        return chromosome[start:end].seq , start, end

    return chromosome[start:end].seq