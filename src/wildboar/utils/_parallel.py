# Authors: Isak Samsten
# License: BSD 3 clause

from joblib import Parallel, delayed, effective_n_jobs


def partition_n_jobs(n_jobs, n):
    batch_size = n // n_jobs
    overflow = n % n_jobs
    offsets = []
    offset = 0
    batch_sizes = []
    for i in range(n_jobs):
        current_overflow = 0
        if i < overflow:
            current_overflow = 1
        current_batch_size = batch_size + current_overflow
        offsets.append(offset)
        batch_sizes.append(current_batch_size)
        offset += current_batch_size

    return n_jobs, offsets, batch_sizes


def run_in_parallel(work, n_jobs=-1, **parallel_args):
    """Execute work in parallel.

    Parameters
    ----------
    work : Batch
        A work batch
    n_jobs : int, optional
        Number of parallel jobs, by default -1
    """
    n_jobs = min(effective_n_jobs(n_jobs), work.n_work)
    if n_jobs == 1:
        work(0, 0, work.n_work)
    else:
        n_jobs, offsets, batch_sizes = partition_n_jobs(n_jobs, work.n_work)
        Parallel(n_jobs=n_jobs, **parallel_args)(
            delayed(work)(jobid, offsets[jobid], batch_sizes[jobid])
            for jobid in range(n_jobs)
        )
