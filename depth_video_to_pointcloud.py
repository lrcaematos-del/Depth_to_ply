#!/usr/bin/env python3
import argparse
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def compute_intrinsics_from_fov(image_width: int, image_height: int, horizontal_fov_degrees: float) -> Tuple[float, float, float, float]:
	"""Compute simple pinhole intrinsics from an assumed horizontal FOV.

	Returns (fx, fy, cx, cy). Assumes square pixels (fx == fy). Principal point is at the image center.
	The resulting point cloud will be correct up to an unknown global scale if the true intrinsics differ.
	"""
	fov_rad = math.radians(horizontal_fov_degrees)
	# fx derived from horizontal FOV, assume square pixels => fy = fx
	fx = image_width / (2.0 * math.tan(fov_rad / 2.0))
	fy = fx
	cx = (image_width - 1) / 2.0
	cy = (image_height - 1) / 2.0
	return fx, fy, cx, cy


def map_frame_to_depth(
	frame: np.ndarray,
	min_depth: Optional[float],
	max_depth: Optional[float],
	depth_scale: float,
	clip_min: Optional[float],
	clip_max: Optional[float],
) -> np.ndarray:
	"""Convert a video frame to a depth map in float32.

	- If frame is 3-channel, convert to grayscale and interpret intensity as depth.
	- If min_depth and max_depth are provided, linearly map 0..255 -> [min_depth, max_depth].
	- Else, multiply intensity by depth_scale.
	- Optionally clip the resulting depth to [clip_min, clip_max] if provided.
	"""
	if frame.ndim == 3 and frame.shape[2] == 3:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		gray = frame

	gray_f32 = gray.astype(np.float32)

	if min_depth is not None and max_depth is not None:
		# Map 0..255 -> [min_depth, max_depth]
		depth = min_depth + (gray_f32 / 255.0) * (max_depth - min_depth)
	else:
		# Interpret intensity directly scaled to depth units
		depth = gray_f32 * depth_scale

	if clip_min is not None:
		depth = np.maximum(depth, np.float32(clip_min))
	if clip_max is not None:
		depth = np.minimum(depth, np.float32(clip_max))

	return depth


def generate_point_cloud_from_depth(
	depth: np.ndarray,
	fx: float,
	fy: float,
	cx: float,
	cy: float,
	downsample: int = 1,
	valid_min: Optional[float] = None,
	valid_max: Optional[float] = None,
) -> np.ndarray:
	"""Project a depth map to a 3D point cloud using a simple pinhole model.

	Returns an array of shape (N, 3) with XYZ.
	"""
	if downsample > 1:
		depth = depth[::downsample, ::downsample]
		cx = cx / downsample
		cy = cy / downsample
		fx = fx / downsample
		fy = fy / downsample

	h, w = depth.shape[:2]
	u_coords = np.arange(w, dtype=np.float32)
	v_coords = np.arange(h, dtype=np.float32)
	u_grid, v_grid = np.meshgrid(u_coords, v_coords)

	z = depth.astype(np.float32)
	if valid_min is not None:
		z = np.where(z >= valid_min, z, np.nan)
	if valid_max is not None:
		z = np.where(z <= valid_max, z, np.nan)

	# Remove zeros by default as commonly used invalid depth value
	z = np.where(z > 0.0, z, np.nan)

	x = (u_grid - cx) * z / fx
	y = (v_grid - cy) * z / fy

	points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
	# Drop NaNs
	points = points[~np.isnan(points).any(axis=1)]
	return points


def generate_colors_from_depth(
	depth: np.ndarray,
	downsample: int = 1,
	colormap: str = "turbo",
	valid_min: Optional[float] = None,
	valid_max: Optional[float] = None,
) -> np.ndarray:
	"""Create pseudo-color RGB colors from depth using an OpenCV colormap."""
	if downsample > 1:
		depth = depth[::downsample, ::downsample]

	depth_norm = depth.copy().astype(np.float32)
	if valid_min is None:
		valid_min = np.nanmin(depth_norm[depth_norm > 0]) if np.any(depth_norm > 0) else 0.0
	if valid_max is None:
		valid_max = np.nanmax(depth_norm)

	# Normalize to 0..255 for colormap
	den = (valid_max - valid_min) if (valid_max - valid_min) > 1e-8 else 1.0
	depth_norm = (np.clip(depth_norm, valid_min, valid_max) - valid_min) / den
	depth_u8 = np.uint8(np.nan_to_num(depth_norm) * 255.0)

	cv2_map = {
		"jet": cv2.COLORMAP_JET,
		"turbo": cv2.COLORMAP_TURBO,
		"viridis": cv2.COLORMAP_VIRIDIS,
		"magma": cv2.COLORMAP_MAGMA,
		"plasma": cv2.COLORMAP_PLASMA,
	}
	cv2_cmap = cv2_map.get(colormap.lower(), cv2.COLORMAP_TURBO)
	colored_bgr = cv2.applyColorMap(depth_u8, cv2_cmap)
	rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
	rgb = rgb.reshape(-1, 3)
	return rgb


def write_ply(vertices_xyz: np.ndarray, colors_rgb: Optional[np.ndarray], filepath: Path) -> None:
	"""Write an ASCII PLY with optional per-vertex colors (uint8)."""
	num_vertices = vertices_xyz.shape[0]
	with open(filepath, "w", encoding="utf-8") as f:
		f.write("ply\n")
		f.write("format ascii 1.0\n")
		f.write(f"element vertex {num_vertices}\n")
		f.write("property float x\n")
		f.write("property float y\n")
		f.write("property float z\n")
		if colors_rgb is not None:
			f.write("property uchar red\n")
			f.write("property uchar green\n")
			f.write("property uchar blue\n")
		f.write("end_header\n")

		if colors_rgb is not None and colors_rgb.shape[0] == num_vertices:
			for (x, y, z), (r, g, b) in zip(vertices_xyz, colors_rgb.astype(np.uint8)):
				f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
		else:
			for x, y, z in vertices_xyz:
				f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def process_video(
	video_path: Path,
	output_dir: Path,
	frame_step: int,
	downsample: int,
	assumed_fov_deg: float,
	min_depth: Optional[float],
	max_depth: Optional[float],
	depth_scale: float,
	clip_min: Optional[float],
	clip_max: Optional[float],
	merge: bool,
	colorize: bool,
	colormap: str,
	valid_min: Optional[float],
	valid_max: Optional[float],
) -> None:
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Unable to open video: {video_path}")

	image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fx, fy, cx, cy = compute_intrinsics_from_fov(image_width, image_height, assumed_fov_deg)

	output_dir.mkdir(parents=True, exist_ok=True)

	frame_index = -1
	merged_points: Optional[np.ndarray] = None
	merged_colors: Optional[np.ndarray] = None

	while True:
		ret, frame = cap.read()
		if not ret:
			break
		frame_index += 1
		if frame_step > 1 and (frame_index % frame_step) != 0:
			continue

		depth = map_frame_to_depth(frame, min_depth, max_depth, depth_scale, clip_min, clip_max)

		points = generate_point_cloud_from_depth(
			depth=depth,
			fx=fx,
			fy=fy,
			cx=cx,
			cy=cy,
			downsample=downsample,
			valid_min=valid_min,
			valid_max=valid_max,
		)

		colors = None
		if colorize:
			colors = generate_colors_from_depth(
				depth=depth,
				downsample=downsample,
				colormap=colormap,
				valid_min=valid_min,
				valid_max=valid_max,
			)
			# Align color array with valid points only, using the same criteria used for points
			depth_ds = depth[::downsample, ::downsample]
			mask_valid = (depth_ds > 0)
			if valid_min is not None:
				mask_valid &= (depth_ds >= valid_min)
			if valid_max is not None:
				mask_valid &= (depth_ds <= valid_max)
			colors = colors[mask_valid.reshape(-1)]

		frame_out = output_dir / f"frame_{frame_index:05d}.ply"
		write_ply(points, colors, frame_out)

		if merge:
			if merged_points is None:
				merged_points = points
				merged_colors = colors
			else:
				merged_points = np.concatenate([merged_points, points], axis=0)
				if merged_colors is not None and colors is not None:
					merged_colors = np.concatenate([merged_colors, colors], axis=0)
				else:
					merged_colors = None

	cap.release()

	if merge and merged_points is not None and merged_points.shape[0] > 0:
		write_ply(merged_points, merged_colors, output_dir / "merged.ply")


def build_argparser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(description="Convert a depth video to PLY point clouds without known intrinsics.")
	p.add_argument("--input", required=True, help="Path to depth video file (e.g., MP4/AVI)")
	p.add_argument("--out", required=True, help="Output directory for PLY files")
	p.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame")
	p.add_argument("--downsample", type=int, default=1, help="Spatial downsampling factor for speed (>=1)")
	p.add_argument("--fov", type=float, default=60.0, help="Assumed horizontal FOV in degrees (used to derive fx, fy)")
	p.add_argument("--min-depth", type=float, default=None, help="If set with --max-depth, map 0..255 -> [min,max] depth units")
	p.add_argument("--max-depth", type=float, default=None, help="If set with --min-depth, map 0..255 -> [min,max] depth units")
	p.add_argument("--depth-scale", type=float, default=1.0, help="If min/max not set, multiply 8-bit intensity by this scale to get depth units")
	p.add_argument("--clip-min", type=float, default=None, help="Clip depths below this value")
	p.add_argument("--clip-max", type=float, default=None, help="Clip depths above this value")
	p.add_argument("--valid-min", type=float, default=None, help="Discard points with depth < valid-min")
	p.add_argument("--valid-max", type=float, default=None, help="Discard points with depth > valid-max")
	p.add_argument("--merge", action="store_true", help="Also write a merged PLY across processed frames (assumes static camera)")
	p.add_argument("--colorize", action="store_true", help="Add pseudo-color based on depth to the PLY")
	p.add_argument("--colormap", type=str, default="turbo", help="OpenCV colormap: turbo, jet, viridis, magma, plasma")
	return p


def main() -> None:
	args = build_argparser().parse_args()

	video_path = Path(args.input)
	output_dir = Path(args.out)

	process_video(
		video_path=video_path,
		output_dir=output_dir,
		frame_step=max(1, int(args.frame_step)),
		downsample=max(1, int(args.downsample)),
		assumed_fov_deg=float(args.fov),
		min_depth=args.min_depth,
		max_depth=args.max_depth,
		depth_scale=float(args.depth_scale),
		clip_min=args.clip_min,
		clip_max=args.clip_max,
		merge=bool(args.merge),
		colorize=bool(args.colorize),
		colormap=str(args.colormap),
		valid_min=args.valid_min,
		valid_max=args.valid_max,
	)


if __name__ == "__main__":
	main()