import multiprocessing

import numpy as np

import utils.io


def determine_n_processes(n_processes, n_tasks):
    if n_processes < 1:
        n_processes = multiprocessing.cpu_count()
    n_processes = min(n_processes, n_tasks)
    return n_processes


def create_switched_mask(switched_sequences, all_sequences):
    is_switched = np.zeros(len(all_sequences), dtype=np.bool)
    if switched_sequences is not None:
        switched_sequences = set(utils.io.read(switched_sequences)
                                 if isinstance(switched_sequences, str) else switched_sequences)
        for idx, seq in enumerate(all_sequences):
            if seq in switched_sequences:
                is_switched[idx] = ~is_switched[idx]
    return is_switched


'''
def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    source: https://www.pythonfixing.com/2022/01/fixed-starmap-combined-with-tqdm.html
    """
    if self._state != multiprocessing.RUN:
        raise ValueError("Pool not running")

    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = multiprocessing.Pool._get_tasks(func, iterable, chunksize)
    result = multiprocessing.IMapIterator(self._cache)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          multiprocessing.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


multiprocessing.Pool.istarmap = istarmap
'''