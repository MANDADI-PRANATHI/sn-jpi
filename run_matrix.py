import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')

from matrix import jobs
from train import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--results', default='results')
    parser.add_argument('--protocol')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    selected = [job for job in jobs() if
                (args.protocol is None or job[0] == args.protocol) and
                (args.seed is None or job[3] == args.seed)]
    Path(args.results).mkdir(parents=True, exist_ok=True)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run, *job, args.results): job for job in selected}
        for future in as_completed(pending):
            try:
                print('Completed', future.result(), flush=True)
            except Exception as error:
                failures.append(pending[future])
                print('Failed', pending[future], repr(error), flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} of {len(selected)} runs failed')


if __name__ == '__main__':
    main()
