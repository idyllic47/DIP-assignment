import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from data_utils import ColmapDataset
from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer
from train import GaussianTrainer, TrainConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir", default="data/chair")
    parser.add_argument("--checkpoint", default="data/chair/checkpoints/checkpoint_000180.pt")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", default="data/chair/simplified_step_benchmark.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ColmapDataset(args.colmap_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = GaussianModel(
        points3D_xyz=dataset.points3D_xyz,
        points3D_rgb=dataset.points3D_rgb,
    )
    sample = dataset[0]
    renderer = GaussianRenderer(sample["image"].shape[0], sample["image"].shape[1])
    config = TrainConfig(
        num_epochs=1,
        batch_size=1,
        checkpoint_dir="data/chair/benchmark_tmp",
        log_dir="data/chair/benchmark_tmp",
    )
    trainer = GaussianTrainer(model, renderer, config, device)
    trainer.load_checkpoint(args.checkpoint)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    times = []
    iterator = iter(loader)
    for _ in range(args.steps):
        batch = next(iterator)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        trainer.train_step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    result = {
        "steps": args.steps,
        "avg_step_seconds": sum(times) / len(times),
        "min_step_seconds": min(times),
        "max_step_seconds": max(times),
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device.type == "cuda"
            else None
        ),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
