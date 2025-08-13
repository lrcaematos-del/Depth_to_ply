# Depth Video to Point Cloud (No Intrinsics)

This script converts a depth-map video into PLY point clouds without requiring known camera intrinsics. It assumes a simple pinhole camera model derived from an assumed horizontal field-of-view (FOV) and uses the video frame intensities as depth values.

Note: Without true intrinsics (fx, fy, cx, cy) and a calibrated depth scale, the point cloud is only correct up to an unknown global scale and may be distorted. Provide reasonable `--fov` and depth mapping parameters for best results.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Usage

```bash
python3 depth_video_to_pointcloud.py \
  --input /path/to/depth_video.mp4 \
  --out /path/to/output_dir \
  --fov 60 \
  --frame-step 1 \
  --downsample 2 \
  --colorize \
  --min-depth 0.3 --max-depth 5.0
```

- **--fov**: Assumed horizontal field of view in degrees (default 60). Used to derive intrinsics (fx, fy) with the principal point at image center.
- **--min-depth/--max-depth**: If provided, map 8-bit intensity 0..255 to this depth range (e.g., meters). If not provided, `--depth-scale` is used to multiply intensity directly.
- **--downsample**: Spatial downsampling to speed up processing.
- **--frame-step**: Process every Nth frame.
- **--colorize/--colormap**: Optional pseudo-color per vertex in the PLY.
- **--merge**: Additionally write `merged.ply` as a concatenation of all processed frames (assumes static camera; otherwise geometrically incorrect).

Outputs per-frame PLY files: `frame_00000.ply`, `frame_00010.ply`, ... and optionally `merged.ply`.