#!/usr/bin/env python3
"""
Extract all frames from a PrimeSense/OpenNI .oni recording.

Depth frames  -> out_dir/depth/depth_000000.npy   (uint16, millimeters)
Color frames  -> out_dir/color/color_000000.png   (if the recording has RGB)
Metadata      -> out_dir/meta.txt

Safe to re-run: already-extracted frames are skipped (resume support).
Usage: python3 extract_oni.py recording.oni out_dir
"""
import glob
import os
import sys

import numpy as np
from openni import openni2

LIB_CANDIDATES = glob.glob("/usr/lib/*-linux-gnu*/libOpenNI2.so*")
LIB_DIR = os.path.dirname(LIB_CANDIDATES[0]) if LIB_CANDIDATES else None

END_TIMEOUT_S = 3  # no new frame for 3 seconds => end of recording


def pump_stream(stream, save_fn, label):
    """Read frames until the recording ends (timeout-based, never blocks forever)."""
    n = 0
    while True:
        ready = openni2.wait_for_any_stream([stream], END_TIMEOUT_S)
        if not ready:
            break  # no frame arrived within timeout -> end of recording
        frame = stream.read_frame()
        save_fn(frame, n)
        n += 1
        if n % 100 == 0:
            print(f"  {label}: {n} frames...", flush=True)
    print(f"{label} done: {n} frames", flush=True)
    return n


def extract(oni_path: str, out_dir: str) -> None:
    openni2.initialize(LIB_DIR)
    dev = openni2.Device.open_file(oni_path.encode())

    try:
        pbc = openni2.PlaybackSupport(dev)
        pbc.set_speed(-1.0)          # read as fast as possible
        pbc.set_repeat_enabled(False)
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    meta_lines = [f"source: {oni_path}"]
    n_depth = n_color = 0

    if dev.has_sensor(openni2.SENSOR_DEPTH):
        depth_dir = os.path.join(out_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        ds = dev.create_depth_stream()
        ds.start()
        vm = ds.get_video_mode()
        meta_lines.append(
            f"depth: {vm.resolutionX}x{vm.resolutionY} @ {vm.fps}fps, "
            f"pixelFormat={vm.pixelFormat}"
        )

        def save_depth(frame, i):
            path = os.path.join(depth_dir, f"depth_{i:06d}.npy")
            if os.path.exists(path):
                return  # resume: already extracted
            arr = np.frombuffer(
                frame.get_buffer_as_uint16(), dtype=np.uint16
            ).reshape(frame.height, frame.width)
            np.save(path, arr)

        n_depth = pump_stream(ds, save_depth, "depth")
        ds.stop()

    if dev.has_sensor(openni2.SENSOR_COLOR):
        import cv2
        color_dir = os.path.join(out_dir, "color")
        os.makedirs(color_dir, exist_ok=True)
        cs = dev.create_color_stream()
        cs.start()
        vm = cs.get_video_mode()
        meta_lines.append(f"color: {vm.resolutionX}x{vm.resolutionY} @ {vm.fps}fps")

        def save_color(frame, i):
            path = os.path.join(color_dir, f"color_{i:06d}.png")
            if os.path.exists(path):
                return
            arr = np.frombuffer(
                frame.get_buffer_as_uint8(), dtype=np.uint8
            ).reshape(frame.height, frame.width, 3)
            cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

        n_color = pump_stream(cs, save_color, "color")
        cs.stop()

    meta_lines.append(f"extracted: depth={n_depth} color={n_color}")
    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write("\n".join(meta_lines) + "\n")

    dev.close()
    openni2.unload()
    print("All done. Metadata written to meta.txt", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2])
