"""
Bundle Adjustment Implementation using PyTorch
Recover 3D points, camera extrinsics, and focal length from 2D observations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import math

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "data"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def euler_angles_to_matrix(euler_angles, convention="XYZ"):
    """
    Convert Euler angles to rotation matrices.
    
    Args:
        euler_angles: (..., 3) tensor of Euler angles in radians
        convention: string specifying the rotation convention (e.g., "XYZ", "ZYX")
    
    Returns:
        R: (..., 3, 3) tensor of rotation matrices
    """
    # Get the shape
    shape = euler_angles.shape
    batch_shape = shape[:-1]
    
    # Extract angles
    angles = euler_angles.reshape(-1, 3)
    alpha = angles[:, 0]  # First rotation
    beta = angles[:, 1]   # Second rotation
    gamma = angles[:, 2]  # Third rotation
    
    # Compute trigonometric functions
    cos_a = torch.cos(alpha)
    sin_a = torch.sin(alpha)
    cos_b = torch.cos(beta)
    sin_b = torch.sin(beta)
    cos_g = torch.cos(gamma)
    sin_g = torch.sin(gamma)
    
    # Create rotation matrices for each axis
    zeros = torch.zeros_like(alpha)
    ones = torch.ones_like(alpha)
    
    # Rotation around X axis
    Rx = torch.stack([
        torch.stack([ones, zeros, zeros], dim=-1),
        torch.stack([zeros, cos_a, -sin_a], dim=-1),
        torch.stack([zeros, sin_a, cos_a], dim=-1),
    ], dim=-1)
    
    # Rotation around Y axis
    Ry = torch.stack([
        torch.stack([cos_b, zeros, sin_b], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_b, zeros, cos_b], dim=-1),
    ], dim=-1)
    
    # Rotation around Z axis
    Rz = torch.stack([
        torch.stack([cos_g, -sin_g, zeros], dim=-1),
        torch.stack([sin_g, cos_g, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=-1)
    
    # Combine rotations based on convention
    if convention == "XYZ":
        R = Rz @ Ry @ Rx
    elif convention == "ZYX":
        R = Rx @ Ry @ Rz
    elif convention == "XZY":
        R = Ry @ Rz @ Rx
    elif convention == "YZX":
        R = Rx @ Rz @ Ry
    elif convention == "YXZ":
        R = Rz @ Rx @ Ry
    elif convention == "ZXY":
        R = Ry @ Rx @ Rz
    else:
        raise ValueError(f"Unknown convention: {convention}")
    
    # Reshape to original batch shape
    R = R.reshape(batch_shape + (3, 3))
    
    return R

# Known parameters
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
NUM_VIEWS = 50
NUM_POINTS = 20000
CX = IMAGE_WIDTH / 2.0
CY = IMAGE_HEIGHT / 2.0


def load_data():
    """Load 2D observations and point colors."""
    print("Loading data...")
    points2d = np.load(f"{DATA_DIR}/points2d.npz")
    points3d_colors = np.load(f"{DATA_DIR}/points3d_colors.npy")
    
    # Convert to tensors
    observations = {}
    for i in range(NUM_VIEWS):
        key = f"view_{i:03d}"
        obs = points2d[key]  # (N, 3): [x, y, visibility]
        observations[key] = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
    
    colors = torch.tensor(points3d_colors, dtype=torch.float32, device=DEVICE)
    
    print(f"Loaded {NUM_VIEWS} views, {NUM_POINTS} points per view")
    return observations, colors


def project_points(points3d, R, T, f, cx, cy):
    """
    Project 3D points to 2D using camera parameters.
    
    Args:
        points3d: (N, 3) 3D point coordinates
        R: (3, 3) rotation matrix
        T: (3,) translation vector
        f: focal length (scalar)
        cx, cy: principal point coordinates
    
    Returns:
        points2d: (N, 2) projected 2D coordinates [u, v]
    """
    # Transform to camera coordinates: [Xc, Yc, Zc] = R @ P + T
    points_cam = points3d @ R.T + T.unsqueeze(0)  # (N, 3)
    
    Xc = points_cam[:, 0]
    Yc = points_cam[:, 1]
    Zc = points_cam[:, 2]
    
    # Avoid division by zero
    epsilon = 1e-6
    Zc = torch.where(torch.abs(Zc) < epsilon, torch.sign(Zc) * epsilon, Zc)
    
    # Project to 2D: u = -f * Xc/Zc + cx, v = f * Yc/Zc + cy
    u = -f * Xc / Zc + cx
    v = f * Yc / Zc + cy
    
    return torch.stack([u, v], dim=1)  # (N, 2)


def compute_reprojection_error(points3d, R, T, f, observations, cx, cy):
    """
    Compute reprojection error for all views.
    
    Args:
        points3d: (N, 3) 3D point coordinates
        R: (V, 3, 3) rotation matrices for all views
        T: (V, 3) translation vectors for all views
        f: focal length (scalar)
        observations: dict of {view_key: (N, 3)} containing [x, y, visibility]
        cx, cy: principal point coordinates
    
    Returns:
        loss: scalar reprojection error
    """
    total_loss = 0.0
    total_visible = 0
    
    for i in range(NUM_VIEWS):
        key = f"view_{i:03d}"
        obs = observations[key]  # (N, 3): [x, y, visibility]
        
        # Project 3D points to 2D
        pred_2d = project_points(points3d, R[i], T[i], f, cx, cy)  # (N, 2)
        
        # Get visibility mask
        visibility = obs[:, 2]  # (N,)
        visible_mask = visibility > 0.5
        
        if visible_mask.sum() > 0:
            # Compute error for visible points only
            error = pred_2d[visible_mask] - obs[visible_mask, :2]  # (N_visible, 2)
            loss = torch.sum(error ** 2)
            total_loss += loss
            total_visible += visible_mask.sum()
    
    # Normalize by total number of visible observations
    if total_visible > 0:
        loss = total_loss / total_visible
    else:
        loss = torch.tensor(0.0, device=DEVICE)
    
    return loss


def initialize_parameters():
    """
    Initialize all parameters for optimization.
    
    Returns:
        points3d: (N, 3) 3D point coordinates
        euler_angles: (V, 3) Euler angles for rotation
        translations: (V, 3) translation vectors
        f: focal length
    """
    # Initialize 3D points near origin with small random noise
    points3d = torch.randn(NUM_POINTS, 3, device=DEVICE) * 0.1
    
    # Initialize rotations to identity (Euler angles = 0)
    euler_angles = torch.zeros(NUM_VIEWS, 3, device=DEVICE)
    
    # Initialize translations: cameras at [0, 0, -d] where d ~ 2-3
    # This places cameras in front of the object (which is at origin)
    d = 2.5
    translations = torch.zeros(NUM_VIEWS, 3, device=DEVICE)
    translations[:, 2] = -d
    
    # Initialize focal length based on reasonable FoV (e.g., 60 degrees)
    fov = 60.0  # degrees
    fov_rad = np.deg2rad(fov)
    f_init = IMAGE_HEIGHT / (2.0 * np.tan(fov_rad / 2.0))
    f = torch.tensor(f_init, device=DEVICE)
    
    print(f"Initialized parameters:")
    print(f"  - 3D points: {points3d.shape}, mean={points3d.mean().item():.4f}, std={points3d.std().item():.4f}")
    print(f"  - Euler angles: {euler_angles.shape}")
    print(f"  - Translations: {translations.shape}, mean={translations.mean().item():.4f}")
    print(f"  - Focal length: {f.item():.2f}")
    
    return points3d, euler_angles, translations, f


def bundle_adjustment(observations, num_iterations=1000, lr=0.01):
    """
    Perform Bundle Adjustment optimization.
    
    Args:
        observations: dict of 2D observations
        num_iterations: number of optimization iterations
        lr: learning rate
    
    Returns:
        points3d: optimized 3D points
        euler_angles: optimized Euler angles
        translations: optimized translations
        f: optimized focal length
        losses: list of loss values during optimization
    """
    # Initialize parameters
    points3d, euler_angles, translations, f = initialize_parameters()
    
    # Set requires_grad for optimization
    points3d.requires_grad = True
    euler_angles.requires_grad = True
    translations.requires_grad = True
    f.requires_grad = True
    
    # Create optimizer
    optimizer = torch.optim.Adam([
        {'params': [points3d], 'lr': lr * 0.1},  # Points need smaller learning rate
        {'params': [euler_angles], 'lr': lr},
        {'params': [translations], 'lr': lr},
        {'params': [f], 'lr': lr * 0.01}  # Focal length needs even smaller learning rate
    ])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    losses = []
    
    print("\nStarting optimization...")
    print(f"{'Iter':>6} | {'Loss':>12} | {'Focal':>8} | {'LR':>10}")
    print("-" * 45)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Convert Euler angles to rotation matrices
        R = euler_angles_to_matrix(euler_angles, convention="XYZ")  # (V, 3, 3)
        
        # Compute loss
        loss = compute_reprojection_error(points3d, R, translations, f, observations, CX, CY)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(points3d, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(euler_angles, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(translations, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(f, max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step(loss)
        
        # Record loss
        losses.append(loss.item())
        
        # Print progress
        if (iteration + 1) % 50 == 0 or iteration == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"{iteration+1:6d} | {loss.item():12.4f} | {f.item():8.2f} | {current_lr:10.6f}")
    
    print("-" * 45)
    print(f"Optimization complete! Final loss: {loss.item():.4f}")
    print(f"Final focal length: {f.item():.2f}")
    
    # Convert to numpy for saving
    points3d_np = points3d.detach().cpu().numpy()
    euler_angles_np = euler_angles.detach().cpu().numpy()
    translations_np = translations.detach().cpu().numpy()
    f_np = f.detach().cpu().numpy()
    
    return points3d_np, euler_angles_np, translations_np, f_np, losses


def plot_losses(losses, save_path=None):
    """Plot and optionally save loss curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Reprojection Error (MSE)', fontsize=12)
    plt.title('Bundle Adjustment Optimization Progress', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss plot to {save_path}")
    else:
        plt.show()
    plt.close()


def save_obj(points3d, colors, save_path):
    """
    Save 3D point cloud as colored OBJ file.
    
    Args:
        points3d: (N, 3) numpy array of 3D coordinates
        colors: (N, 3) numpy array of RGB colors in [0, 1] range
        save_path: path to save the OBJ file
    """
    with open(save_path, 'w') as f:
        for i in range(len(points3d)):
            x, y, z = points3d[i]
            r, g, b = colors[i]
            # OBJ format: v x y z r g b
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
    
    print(f"Saved OBJ file to {save_path}")


def save_camera_params(euler_angles, translations, f, save_path):
    """Save camera parameters to a text file."""
    with open(save_path, 'w') as f_out:
        f_out.write(f"# Camera Parameters from Bundle Adjustment\n")
        f_out.write(f"# Focal length: {f:.6f}\n")
        f_out.write(f"# Image size: {IMAGE_WIDTH} x {IMAGE_HEIGHT}\n")
        f_out.write(f"# Principal point: ({CX:.2f}, {CY:.2f})\n")
        f_out.write(f"\n")
        f_out.write(f"# Format: view_id, euler_x, euler_y, euler_z, tx, ty, tz\n")
        for i in range(NUM_VIEWS):
            ea = euler_angles[i]
            t = translations[i]
            f_out.write(f"view_{i:03d}, {ea[0]:.6f}, {ea[1]:.6f}, {ea[2]:.6f}, "
                       f"{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}\n")
    
    print(f"Saved camera parameters to {save_path}")


def main():
    """Main function to run Bundle Adjustment."""
    print("=" * 60)
    print("Bundle Adjustment with PyTorch")
    print("=" * 60)
    
    # Load data
    observations, colors = load_data()
    colors_np = colors.cpu().numpy()
    
    # Run Bundle Adjustment
    points3d_np, euler_angles_np, translations_np, f_np, losses = bundle_adjustment(
        observations, num_iterations=2000, lr=0.01
    )
    
    # Plot and save loss curve
    plot_losses(losses, save_path=f"{OUTPUT_DIR}/loss_curve.png")
    
    # Save results
    save_obj(points3d_np, colors_np, save_path=f"{OUTPUT_DIR}/reconstructed_pointcloud.obj")
    save_camera_params(euler_angles_np, translations_np, f_np, 
                       save_path=f"{OUTPUT_DIR}/camera_params.txt")
    
    # Save numpy arrays for further analysis
    np.save(f"{OUTPUT_DIR}/points3d.npy", points3d_np)
    np.save(f"{OUTPUT_DIR}/euler_angles.npy", euler_angles_np)
    np.save(f"{OUTPUT_DIR}/translations.npy", translations_np)
    np.save(f"{OUTPUT_DIR}/focal_length.npy", f_np)
    
    print("\n" + "=" * 60)
    print("All results saved to 'results/' directory!")
    print("=" * 60)


if __name__ == "__main__":
    main()
