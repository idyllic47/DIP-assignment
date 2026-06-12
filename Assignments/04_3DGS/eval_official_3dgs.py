import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", default="data/chair")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath("gaussian-splatting-main"))

    from gaussian_renderer import render
    from scene import GaussianModel, Scene

    dataset = SimpleNamespace(
        sh_degree=3,
        source_path=os.path.abspath(args.source_path),
        model_path=args.model_path,
        images="images",
        depths="",
        resolution=args.resolution,
        white_background=args.white_background,
        data_device="cpu",
        eval=True,
        train_test_exp=False,
    )
    pipe = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    def evaluate_views(views):
        mae_sum = 0.0
        mse_sum = 0.0
        psnr_sum = 0.0
        count = 0
        with torch.no_grad():
            for view in views:
                image = render(view, gaussians, pipe, bg)["render"]
                gt = view.original_image.cuda()
                if view.alpha_mask is not None:
                    mask = view.alpha_mask.cuda()
                    image = image * mask
                    gt = gt * mask
                image = image.clamp(0.0, 1.0)
                gt = gt.clamp(0.0, 1.0)
                diff = image - gt
                mae = diff.abs().mean()
                mse = (diff * diff).mean()
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
                mae_sum += mae.item()
                mse_sum += mse.item()
                psnr_sum += psnr.item()
                count += 1
        return {
            "views": count,
            "mae": mae_sum / count,
            "mse": mse_sum / count,
            "psnr": psnr_sum / count,
        }

    metrics = {
        "model_path": args.model_path,
        "iteration": scene.loaded_iter,
        "resolution": args.resolution,
        "white_background": args.white_background,
        "train": evaluate_views(scene.getTrainCameras()),
        "test": evaluate_views(scene.getTestCameras()),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
