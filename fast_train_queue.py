#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class JobResult:
    exp_name: str
    gpu_id: str
    train_returncode: int
    infer_returncode: Optional[int]


def parse_gpus(gpus: Optional[str]) -> List[str]:
    if gpus is None or gpus.strip() == "":
        return ["0", "1", "2", "3"]
    return [g.strip() for g in gpus.split(",") if g.strip() != ""]


def run_one(exp_name: str, gpu_id: str, deepspeed: Optional[str], do_infer: bool, extra_args: List[str]) -> JobResult:
    repo_root = Path(__file__).resolve().parent
    config = repo_root / "configs" / "finetune" / f"{exp_name}.py"
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")

    work_dir = repo_root / "work_dirs" / exp_name
    work_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["GPUS_PER_NODE"] = "1"
    env["NNODES"] = "1"

    train_cmd = ["bash", str(repo_root / "scripts" / "train_ddp.sh"), str(config)]
    if deepspeed is not None:
        train_cmd += ["--deepspeed", deepspeed]
    train_cmd += extra_args

    train_log = work_dir / f"fast_train_gpu{gpu_id}.log"
    with train_log.open("w", encoding="utf-8") as f:
        train_proc = subprocess.run(train_cmd, cwd=str(repo_root), env=env, stdout=f, stderr=subprocess.STDOUT)

    infer_returncode = None
    if do_infer and train_proc.returncode == 0:
        infer_cmd = ["bash", str(repo_root / "inference.sh"), exp_name]
        infer_log = work_dir / f"fast_infer_gpu{gpu_id}.log"
        with infer_log.open("w", encoding="utf-8") as f:
            infer_proc = subprocess.run(infer_cmd, cwd=str(repo_root), env=env, stdout=f, stderr=subprocess.STDOUT)
        infer_returncode = infer_proc.returncode

    return JobResult(
        exp_name=exp_name,
        gpu_id=str(gpu_id),
        train_returncode=train_proc.returncode,
        infer_returncode=infer_returncode,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_names", nargs="+", help="experiment names under configs/finetune")
    parser.add_argument("--gpus", default=None, help='GPU ids, e.g. "0,1,2,3"')
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--deepspeed", type=str, default='deepspeed_zero2')
    parser.add_argument("--no_infer", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    gpus = parse_gpus(args.gpus)
    if len(gpus) == 0:
        raise ValueError("No GPUs specified")

    max_parallel = max(1, min(int(args.max_parallel), len(gpus)))
    exp_queue = list(args.exp_names)
    extra_args = args.extra_args
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    running: List[tuple[subprocess.Popen, str, str, Path]] = []
    results: List[JobResult] = []

    repo_root = Path(__file__).resolve().parent

    def start(exp_name: str, gpu_id: str) -> tuple[subprocess.Popen, str, str, Path]:
        config = repo_root / "configs" / "finetune" / f"{exp_name}.py"
        if not config.is_file():
            raise FileNotFoundError(f"Config not found: {config}")
        work_dir = repo_root / "work_dirs" / exp_name
        work_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["GPUS_PER_NODE"] = "1"
        env["NNODES"] = "1"
        env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
        try:
            port = 26000 + int(gpu_id)
        except ValueError:
            port = 26000
        env["MASTER_PORT"] = str(port)
        env["MASTER_ADDR"] = env.get("MASTER_ADDR", "127.0.0.1")
        try:
            port = 26000 + int(gpu_id)
        except ValueError:
            port = 26000
        env["MASTER_PORT"] = str(port)
        train_cmd = ["bash", str(repo_root / "scripts" / "train_ddp.sh"), str(config)]
        if args.deepspeed is not None:
            train_cmd += ["--deepspeed", args.deepspeed]
        train_cmd += extra_args
        train_log = work_dir / f"fast_train_gpu{gpu_id}.log"
        f = train_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(train_cmd, cwd=str(repo_root), env=env, stdout=f, stderr=subprocess.STDOUT)
        return proc, exp_name, gpu_id, train_log

    success_exps: List[str] = []

    gpu_pool = gpus[:]

    while exp_queue or running:
        while exp_queue and len(running) < max_parallel and gpu_pool:
            exp_name = exp_queue.pop(0)
            gpu_id = gpu_pool.pop(0)
            proc, exp_name, gpu_id, train_log = start(exp_name, gpu_id)
            running.append((proc, exp_name, gpu_id, train_log))

        if not running:
            continue

        done_index = None
        for i, (proc, exp_name, gpu_id, train_log) in enumerate(running):
            rc = proc.poll()
            if rc is not None:
                done_index = i
                break

        if done_index is None:
            import time
            time.sleep(1)
            continue

        proc, exp_name, gpu_id, _ = running.pop(done_index)
        train_rc = proc.wait()
        if train_rc == 0:
            success_exps.append(exp_name)
        results.append(JobResult(exp_name=exp_name, gpu_id=gpu_id, train_returncode=train_rc, infer_returncode=None))
        gpu_pool.append(gpu_id)

    if success_exps and not args.no_infer:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        infer_cmd = ["bash", str(repo_root / "inference.sh"), *success_exps]
        infer_log = repo_root / "work_dirs" / "fast_infer_all.log"
        infer_log.parent.mkdir(parents=True, exist_ok=True)
        with infer_log.open("w", encoding="utf-8") as f:
            subprocess.run(infer_cmd, cwd=str(repo_root), env=env, stdout=f, stderr=subprocess.STDOUT)

    failed = [r for r in results if r.train_returncode != 0 or (r.infer_returncode not in (None, 0))]
    for r in results:
        status = "ok" if (r.train_returncode == 0 and r.infer_returncode in (None, 0)) else "failed"
        print(f"{r.exp_name}\tgpu={r.gpu_id}\ttrain={r.train_returncode}\tinfer={r.infer_returncode}\t{status}")
    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
