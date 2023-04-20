"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import functools
# import os
# import subprocess
# import torch
# import torch.distributed as dist
import torch.multiprocessing as mp
# from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    try:
        rank = int(os.environ['RANK'])
    except:
        os.environ['RANK']='0'
        os.environ['WORLD_SIZE']='1'
        os.environ['MASTER_PORT']='4321'
        os.environ['MASTER_ADDR']='127.0.0.1'
        # rank = int(os.environ['RANK'])
    rank = int(os.environ['RANK'])
    num_gpus = th.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{rank % num_gpus}"

    # comm = MPI.COMM_WORLD
    backend = "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    rank = int(os.environ['RANK'])
    num_gpus = th.cuda.device_count()
    th.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """

    return th.load(path, **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
