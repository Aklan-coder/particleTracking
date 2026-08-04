#!/usr/bin/env python3
"""Probe .oni: true duration & drops. Reads through frames, no seek."""
import glob, os, sys
from openni import openni2

LIBS = glob.glob("/usr/lib/*-linux-gnu*/libOpenNI2.so*")
openni2.initialize(os.path.dirname(LIBS[0]) if LIBS else None)
dev = openni2.Device.open_file(sys.argv[1].encode())
try:
    pbc = openni2.PlaybackSupport(dev)
    pbc.set_speed(-1.0)
    pbc.set_repeat_enabled(False)
except Exception as e:
    print("playback ctl:", e, flush=True)

print(f"=== {os.path.basename(sys.argv[1])} ===", flush=True)
for sensor, mk, label in [
    (openni2.SENSOR_DEPTH, dev.create_depth_stream, "depth"),
    (openni2.SENSOR_COLOR, dev.create_color_stream, "color"),
]:
    if not dev.has_sensor(sensor):
        print(f"{label}: not present", flush=True)
        continue
    s = mk(); s.start()
    vm = s.get_video_mode()
    n, ts0, ts1 = 0, None, None
    while True:
        ready = openni2.wait_for_any_stream([s], 3)
        if ready is None:
            break
        f = s.read_frame()
        if ts0 is None:
            ts0 = f.timestamp
        ts1 = f.timestamp
        n += 1
        if n % 500 == 0:
            print(f"  ...{n}", flush=True)
    s.stop()
    dur = (ts1 - ts0) / 1e6 if (ts0 is not None and n > 1) else 0.0
    eff = (n - 1) / dur if dur > 0 else 0.0
    exp = int(round(dur * vm.fps)) + 1 if dur > 0 else n
    print(f"{label}: {vm.resolutionX}x{vm.resolutionY} nominal {vm.fps}fps", flush=True)
    print(f"  stored frames: {n}", flush=True)
    print(f"  true duration: {dur:.1f} s ({dur/60:.2f} min)", flush=True)
    print(f"  effective fps: {eff:.2f}", flush=True)
    print(f"  est. dropped:  {max(0, exp-n)} of expected {exp}", flush=True)
dev.close(); openni2.unload()
