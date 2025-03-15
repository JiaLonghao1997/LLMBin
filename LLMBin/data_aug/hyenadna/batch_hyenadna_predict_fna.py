import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl

from transformers import AutoConfig, AutoModelForMaskedLM
# from caduceus.configuration_caduceus import CaduceusConfig
# from caduceus.modeling_caduceus import Caduceus
# from caduceus.modeling_caduceus import CaduceusForSequenceClassification
import json
import os
import random
import time
from functools import wraps
from typing import Callable, List, Sequence
from datetime import datetime
from pathlib import Path
from Bio import SeqIO
import subprocess
import pickle
import tqdm
import sys

import fsspec
import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

sys.path.append("/home1/jialh/metaHiC/tools/BERTBin/BERTBin/data_aug/hyenadna")
import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim_groups import add_optimizer_hooks
# from caduceus.tokenization_caduceus import CaduceusTokenizer
from sequence_lightning import SequenceLightningModule
from standalone_hyenadna import CharacterTokenizer

log = src.utils.train.get_logger(__name__)
import psutil
import subprocess

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends
from src.tasks.decoders import SequenceDecoder

seed = 42
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

OmegaConf.register_new_resolver('eval', eval)
OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver('min', lambda x, y: min([x, y]))

import psutil
import torch
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional
import time
import threading
import queue
from contextlib import contextmanager


@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage"""
    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_total_gb: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None


def load_model(checkpoint_path, config_path):
    # Load config
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['hyper_parameters']
    # print(f"config: {config}")
    model = SequenceLightningModule(config)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model


class GPUMonitor:
    """GPU monitoring with fallback methods"""

    @staticmethod
    def get_gpu_info(device_id: int = 0) -> Dict[str, float]:
        """Get GPU information using available methods"""
        try:
            # Try using nvidia-smi through subprocess
            result = subprocess.check_output(
                [
                    'nvidia-smi',
                    f'--query-gpu=memory.used,memory.total,utilization.gpu',
                    '--format=csv,nounits,noheader'
                ],
                encoding='utf-8'
            )
            used_mem, total_mem, util = map(float, result.strip().split(','))
            return {
                'memory_used_gb': used_mem / 1024,  # Convert MB to GB
                'memory_total_gb': total_mem / 1024,
                'utilization': util
            }
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                # Fallback to torch.cuda
                if torch.cuda.is_available():
                    used_mem = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # Convert bytes to GB
                    total_mem = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
                    return {
                        'memory_used_gb': used_mem,
                        'memory_total_gb': total_mem,
                        'utilization': None  # torch.cuda doesn't provide utilization info
                    }
            except:
                pass

            # Return None if no GPU info available
            return {
                'memory_used_gb': None,
                'memory_total_gb': None,
                'utilization': None
            }


class ResourceMonitor:
    """Monitors system resources in a background thread."""

    def __init__(self, device_id: int = 0, sampling_interval: float = 0.1):
        self.device_id = device_id
        self.sampling_interval = sampling_interval
        self.snapshots = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self.gpu_monitor = GPUMonitor()

    def start(self):
        """Start monitoring in background thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """Stop monitoring and return all snapshots."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        # Collect all snapshots
        snapshots = []
        while not self.snapshots.empty():
            snapshots.append(self.snapshots.get())
        return snapshots

    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Get CPU stats
                cpu_percent = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory()
                ram_used = ram.used / (1024 ** 3)  # Convert to GB
                ram_total = ram.total / (1024 ** 3)

                # Get GPU stats
                gpu_info = self.gpu_monitor.get_gpu_info(self.device_id)

                snapshot = ResourceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    ram_used_gb=ram_used,
                    ram_total_gb=ram_total,
                    gpu_memory_used_gb=gpu_info['memory_used_gb'],
                    gpu_memory_total_gb=gpu_info['memory_total_gb'],
                    gpu_utilization=gpu_info['utilization']
                )
                self.snapshots.put(snapshot)

                time.sleep(self.sampling_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                break


@contextmanager
def timer(name: str, timings: Dict[str, float]):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if name in timings:
            if isinstance(timings[name], list):
                timings[name].append(elapsed)
            else:
                timings[name] = [timings[name], elapsed]
        else:
            timings[name] = elapsed


def get_memory_stats():
    """Get current memory statistics."""
    stats = {
        'cpu': {
            'used_gb': psutil.Process().memory_info().rss / (1024 ** 3),
            'percent': psutil.Process().memory_percent()
        }
    }

    if torch.cuda.is_available():
        stats['gpu'] = {
            'allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3)
        }

    return stats


def log_memory_stats(logger, prefix=""):
    """Log current memory statistics."""
    stats = get_memory_stats()
    logger.info(f"{prefix}CPU Memory: {stats['cpu']['used_gb']:.2f} GB ({stats['cpu']['percent']:.1f}%)")
    if 'gpu' in stats:
        logger.info(f"{prefix}GPU Memory: {stats['gpu']['allocated_gb']:.2f} GB allocated, "
                    f"{stats['gpu']['reserved_gb']:.2f} GB reserved")


# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
            code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        log.error("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        log.warning(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


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


class BatchPredictor:
    def __init__(
            self,
            config_path: str,
            checkpoint_path: str,
            model_max_length: int = 4096,
            batch_size: int = 32,
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading model from checkpoint: {checkpoint_path}")

        # Track timings
        self.batch_times = []
        self.memory_usage = []

        # Load model
        start_time = time.time()
        self.model = load_model(checkpoint_path, config_path)

        self.model.eval()
        self.model.to(device)

        self.model_load_time = time.time() - start_time

        model_hash = hash(str([(k, v.sum().item()) for k, v in self.model.state_dict().items()]))
        logger.info(f"Model weight hash: {model_hash}")

        # Initialize tokenizer
        logger.info("Initializing CharacterTokenizer")
        self.tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=model_max_length,
            add_special_tokens=False
        )

    def get_gpu_memory(self):
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 1024 ** 2
        return 0

    # https://github.com/leannmlindsey/caduceus/blob/main/batch_predict.py#L838
    def process_batch(self, sequences: List[str]) -> torch.Tensor:
        """Process a batch of sequences."""
        batch_start = time.time()

        # Tokenize
        encoded = [
            self.tokenizer(
                seq,
                padding="max_length",
                max_length=self.model.hparams.dataset.max_length,
                truncation=True,
                add_special_tokens=False
            ) for seq in sequences
        ]

        # Convert to tensor
        input_ids = torch.stack([
            torch.tensor(enc["input_ids"]) for enc in encoded
        ]).to(self.device)

        input_ids = torch.where(
            input_ids == self.tokenizer._vocab_str_to_int["N"],
            self.tokenizer.pad_token_id,
            input_ids
        )

        # dummy_labels = torch.zeros(len(sequences), 2, device=self.device)

        # Model inference
        with torch.no_grad():
            # batch = (input_ids, None)  # Model expects (input, target) tuple
            model_output = self.model.model.backbone(input_ids)
            # print(f"type(model_output): {type(model_output)}, model_output.size(): {model_output.size()}")
            batch_emb = torch.mean(model_output, dim=1)
            # print(f"type(batch_emb): {type(batch_emb)}, batch_emb.size(): {batch_emb.size()}")

        # Keep your existing metrics logging code
        batch_time = time.time() - batch_start
        self.batch_times.append(batch_time)
        self.memory_usage.append(self.get_gpu_memory())

        return batch_emb

    def predict_from_fna(self, dna_sequences: list,
                         batch_size: int):
        """Process sequences from CSV and save results with metrics."""
        # reorder the sequences by length
        lengths = [len(seq) for seq in dna_sequences]
        idx = np.argsort(lengths)  # [::-1]
        dna_sequences = [dna_sequences[i] for i in idx]
        
        peak_memory = 0
        # Process in batches
        # n_batches = (len(df) + self.batch_size - 1) // self.batch_size
        # pbar = tqdm(range(0, len(df), self.batch_size), desc="Processing batches", total=n_batches)

        train_loader = torch.utils.data.DataLoader(dna_sequences, batch_size=batch_size, shuffle=False, num_workers=8)
        pbar = tqdm.tqdm(train_loader, desc=f"Get embedding: ", bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for j, batch_sequences in enumerate(pbar):
            # batch_sequences = df['sequence'].iloc[i:i + self.batch_size].tolist()
            batch_emb = self.process_batch(batch_sequences)
            # print(f"batch_emb.size(): {batch_emb.size()}")
            if j == 0:
                embedding = batch_emb.detach().cpu()
            else:
                embedding = torch.cat((embedding, batch_emb.detach().cpu()), dim=0)

            # Update progress bar
            current_memory = self.get_gpu_memory()
            peak_memory = max(peak_memory, current_memory)
            pbar.set_postfix({
                'batch_time': f'{np.mean(self.batch_times[-10:]):.3f}s',
                'gpu_mem': f'{int(current_memory)}MB'
            })

        # Add predictions to dataframe
        embedding = np.array(embedding)

        # reorder the embeddings
        embedding = embedding[np.argsort(idx)]
        return embedding


def gtdb_hyenadna_embedding(dna_sequences, logger, device, model_name_or_path, model_max_length=400, batch_size=20):
    logger.info("**********begin to get embeddings from GTDB+HyenaDNA*************")
    config_path = os.path.join(model_name_or_path, "config_tree.txt")
    checkpoint_path = os.path.join(model_name_or_path, "checkpoints", "last.ckpt")
    predictor = BatchPredictor(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        model_max_length=model_max_length,
        batch_size=batch_size,
        device=device
    )

    embedding = predictor.predict_from_fna(dna_sequences=dna_sequences,
                               batch_size=batch_size)
    logger.info(f"*****************HyenaDNA embedding.shape={embedding.shape}")
    return embedding


# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description='Batch predict phages from CSV')
#     parser.add_argument('--config', required=True, help='Path to model config')
#     parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
#     parser.add_argument('--input', required=True, help='Input CSV file')
#     parser.add_argument('--outdir', required=True, help='Output directory')
#     parser.add_argument('--output', required=True, help='Output TSV file')
#     parser.add_argument('--min_contig', type=int, default=1000, help='minimum length of contigs')
#     parser.add_argument('--contig_max_length', type=int, default=4096, help='contig_max_length')
#     parser.add_argument('--model_max_length', type=int, default=4096, help='model_max_length')
#     parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#     parser.add_argument('--device', default='cuda', help='Device to run on')
#
#     args = parser.parse_args()
#
#     predictor = BatchPredictor(
#         args.config,
#         args.checkpoint,
#         model_max_length=args.model_max_length,
#         batch_size=args.batch_size,
#         device=args.device
#     )
#
#     predictor.predict_from_fna(args.input, args.outdir, args.output,
#                                args.contig_max_length, args.model_max_length,
#                                args.batch_size, args.min_contig)
#
#
# if __name__ == "__main__":
#     main()