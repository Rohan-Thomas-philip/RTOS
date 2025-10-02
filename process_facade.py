#!/usr/bin/env python3
"""
Process a facade point cloud (.ply): filter, segment main plane, colorize, export.

Usage:
  python process_facade.py --input path/to/input.ply --output colored_output.ply \
      [--voxel 0.01] [--nb_neighbors 30] [--std_ratio 1.5] [--ransac_dist 0.01] \
      [--min_plane_ratio 0.2] [--screenshot screenshot.png] [--report metrics.json]

Notes:
- Pure NumPy/SciPy implementation compatible with Python 3.13 (no Open3D).
- Colors: points on plane -> green (0,255,0), others -> red (255,0,0).
- Outputs colorized .ply and optional screenshot and metrics JSON.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


GREEN = np.array([0, 255, 0], dtype=np.uint8)
RED = np.array([255, 0, 0], dtype=np.uint8)


@dataclass
class PlaneModel:
    normal: np.ndarray  # (a, b, c)
    d: float
    inlier_indices: np.ndarray

    def distance(self, points: np.ndarray) -> np.ndarray:
        n = self.normal / (np.linalg.norm(self.normal) + 1e-12)
        d = self.d / (np.linalg.norm(self.normal) + 1e-12)
        return np.abs(points @ n + d)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment main plane in a facade point cloud")
    p.add_argument("--input", required=True, help="Path to input .ply point cloud")
    p.add_argument("--output", required=True, help="Path to output colorized .ply")
    p.add_argument("--voxel", type=float, default=0.01, help="Voxel size for downsampling in meters")
    p.add_argument("--nb_neighbors", type=int, default=30, help="KNN for statistical outlier removal")
    p.add_argument("--std_ratio", type=float, default=1.5, help="Std ratio for outlier removal")
    p.add_argument("--ransac_dist", type=float, default=0.01, help="RANSAC distance threshold in meters")
    p.add_argument("--min_plane_ratio", type=float, default=0.2, help="Minimum inlier ratio to accept plane")
    p.add_argument("--screenshot", default=None, help="Optional PNG path to save a screenshot")
    p.add_argument("--report", default=None, help="Optional JSON path to save metrics")
    return p.parse_args()


def load_point_cloud(path: str) -> Tuple[np.ndarray, np.ndarray | None]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    ply = PlyData.read(path)
    if 'vertex' not in ply.elements[0].name and (not any(e.name == 'vertex' for e in ply.elements)):
        # Fallback: assume first element is vertices
        vertex = ply.elements[0]
    else:
        vertex = next(e for e in ply.elements if e.name == 'vertex')

    data = vertex.data
    # Support common field names
    x = data['x'].astype(np.float64)
    y = data['y'].astype(np.float64)
    z = data['z'].astype(np.float64)
    points = np.vstack([x, y, z]).T.copy()

    colors = None
    color_fields = [f for f in ('red', 'green', 'blue') if f in data.dtype.names]
    if len(color_fields) == 3:
        colors = np.vstack([data['red'], data['green'], data['blue']]).T.astype(np.uint8)
    return points, colors


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return points
    grid = np.floor(points / voxel).astype(np.int64)
    # Unique cells
    # Pack rows for np.unique
    grid_view = grid.view([('x', np.int64), ('y', np.int64), ('z', np.int64)])
    unique_cells, inverse, counts = np.unique(grid_view, return_inverse=True, return_counts=True)
    sums = np.zeros((unique_cells.shape[0], 3), dtype=np.float64)
    np.add.at(sums, inverse, points)
    centroids = sums / counts[:, None]
    return centroids


def statistical_outlier_removal(points: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    if nb_neighbors < 2 or points.shape[0] <= nb_neighbors:
        return points
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors + 1)  # include self
    mean_dists = dists[:, 1:].mean(axis=1)
    mu = mean_dists.mean()
    sigma = mean_dists.std(ddof=0)
    threshold = mu + std_ratio * sigma
    mask = mean_dists <= threshold
    return points[mask]


def filter_point_cloud(points: np.ndarray, voxel: float, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    pts = voxel_downsample(points, voxel) if voxel and voxel > 0 else points
    pts = statistical_outlier_removal(pts, nb_neighbors, std_ratio)
    return pts


def segment_main_plane(points: np.ndarray, ransac_dist: float, min_plane_ratio: float, max_iterations: int = 3000) -> PlaneModel:
    if points.shape[0] < 100:
        raise ValueError("Not enough points for plane segmentation")

    best_inliers = None
    best_normal = None
    best_d = None

    rng = np.random.default_rng()
    for _ in range(max_iterations):
        idx = rng.choice(points.shape[0], size=3, replace=False)
        p0, p1, p2 = points[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-9:
            continue
        n = n / norm_n
        d = -float(n @ p0)
        distances = np.abs(points @ n + d)
        inliers = np.nonzero(distances <= ransac_dist)[0]
        if best_inliers is None or inliers.size > best_inliers.size:
            best_inliers = inliers
            best_normal = n
            best_d = d

    if best_inliers is None:
        raise RuntimeError("RANSAC failed to find a plane")

    if best_inliers.size / points.shape[0] < min_plane_ratio:
        raise ValueError(f"Plane too small: {best_inliers.size} / {points.shape[0]} < {min_plane_ratio}")

    # Optional refinement: least squares fit on inliers
    inlier_points = points[best_inliers]
    # Fit plane ax+by+cz+d=0 with ||[a,b,c]||=1 via SVD of centered points
    centroid = inlier_points.mean(axis=0)
    U, S, Vt = np.linalg.svd(inlier_points - centroid)
    normal = Vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d_refined = -float(normal @ centroid)

    return PlaneModel(normal=normal.astype(float), d=d_refined, inlier_indices=best_inliers)


def colorize_points(num_points: int, inlier_indices: np.ndarray) -> np.ndarray:
    colors = np.tile(RED, (num_points, 1))
    colors[inlier_indices] = GREEN
    return colors


def compute_plane_extent(points: np.ndarray, plane: PlaneModel) -> Tuple[float, float, float]:
    # Build orthonormal basis on the plane
    n = plane.normal / (np.linalg.norm(plane.normal) + 1e-12)
    # Choose a vector not parallel to n
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(n, u)
    v /= np.linalg.norm(v) + 1e-12

    points = points[plane.inlier_indices]
    # Project onto plane basis
    u_coords = points @ u
    v_coords = points @ v

    width = float(u_coords.max() - u_coords.min())
    height = float(v_coords.max() - v_coords.min())
    area = width * height
    return width, height, area


def save_screenshot(points: np.ndarray, colors: np.ndarray, screenshot_path: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    c = colors.astype(np.float32) / 255.0
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c, s=1)
    # Set equal aspect
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    ranges = xyz_max - xyz_min
    max_range = ranges.max()
    centers = (xyz_max + xyz_min) / 2.0
    ax.set_xlim(centers[0] - max_range / 2, centers[0] + max_range / 2)
    ax.set_ylim(centers[1] - max_range / 2, centers[1] + max_range / 2)
    ax.set_zlim(centers[2] - max_range / 2, centers[2] + max_range / 2)
    ax.view_init(elev=15, azim=60)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(screenshot_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def write_report(report_path: str, plane: PlaneModel, width: float, height: float, area: float, inlier_ratio: float) -> None:
    report = {
        "plane_normal": plane.normal.tolist(),
        "plane_d": plane.d,
        "width_m": width,
        "height_m": height,
        "area_m2": area,
        "inlier_ratio": inlier_ratio,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_ply(points: np.ndarray, colors: np.ndarray, output_path: str) -> None:
    assert points.shape[0] == colors.shape[0]
    vertex = np.empty(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'] = points[:, 0].astype(np.float32)
    vertex['y'] = points[:, 1].astype(np.float32)
    vertex['z'] = points[:, 2].astype(np.float32)
    vertex['red'] = colors[:, 0]
    vertex['green'] = colors[:, 1]
    vertex['blue'] = colors[:, 2]
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(output_path)


def main():
    args = parse_args()

    points_raw, _ = load_point_cloud(args.input)
    points = filter_point_cloud(points_raw, args.voxel, args.nb_neighbors, args.std_ratio)

    plane = segment_main_plane(points, args.ransac_dist, args.min_plane_ratio)

    colors = colorize_points(points.shape[0], plane.inlier_indices)

    # Export colorized cloud
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_ply(points, colors, args.output)

    # Optional metrics and screenshot
    inlier_ratio = float(plane.inlier_indices.size) / float(points.shape[0])
    width = height = area = float('nan')
    try:
        width, height, area = compute_plane_extent(points, plane)
    except Exception:
        pass

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
        write_report(args.report, plane, width, height, area, inlier_ratio)

    if args.screenshot:
        os.makedirs(os.path.dirname(os.path.abspath(args.screenshot)) or ".", exist_ok=True)
        save_screenshot(points, colors, args.screenshot)

    print(f"Saved colorized cloud to: {args.output}")
    if args.report:
        print(f"Saved metrics to: {args.report}")
    if args.screenshot:
        print(f"Saved screenshot to: {args.screenshot}")


if __name__ == "__main__":
    main()
