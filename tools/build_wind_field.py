#!/usr/bin/env python3
"""Build the app's wind-field texture from NOAA GFS.

Downloads just the two 10 m wind records out of the latest GFS analysis —
using the GRIB index to ask for a byte range rather than the whole 500 MB
file — and writes them out as one small PNG the app can fetch as a static
file, plus a JSON manifest saying which model run it came from.

Run by `.github/workflows/wind-field.yml` four times a day; there is no
server involved anywhere in this, and nothing here needs an API key.

Encoding, which `WindFieldService.swift` on the app side decodes:

    R = eastward  wind, 0…255 mapped over ±`SPEED_RANGE` m/s
    G = northward wind, same mapping
    B = 0
    A = 255

Eight bits per component is about 0.4 m/s per step over the range — far
finer than anything the streak animation can show.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import tempfile
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
# NOAA publishes each cycle a few hours after its nominal time, so the newest
# one is often not there yet. Walk back until a cycle answers.
MAX_CYCLES_BACK = 8
# Native GFS here is 0.25°; halving it drops the file to a quarter of the
# size for detail the streaks cannot resolve anyway.
OUTPUT_STEP_DEGREES = 0.5
SPEED_RANGE = 50.0  # m/s, per component
WANTED = ("UGRD:10 m above ground", "VGRD:10 m above ground")


def latest_cycles(now: dt.datetime):
    """Candidate (date, cycle-hour) pairs, newest first."""
    cycle = now.replace(minute=0, second=0, microsecond=0)
    cycle -= dt.timedelta(hours=cycle.hour % 6)
    for _ in range(MAX_CYCLES_BACK):
        yield cycle
        cycle -= dt.timedelta(hours=6)


def fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    request = urllib.request.Request(url)
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def index_ranges(index_text: str) -> dict[str, tuple[int, int]]:
    """Byte range of each wanted record, from the `.idx` sidecar.

    Lines look like `585:423868602:d=2026091306:UGRD:10 m above ground:anl:`
    — a record's data runs from its own offset up to the next record's.
    """
    offsets: list[int] = []
    labels: list[str] = []
    for line in index_text.splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        offsets.append(int(parts[1]))
        labels.append(f"{parts[3]}:{parts[4]}")

    ranges: dict[str, tuple[int, int]] = {}
    for wanted in WANTED:
        if wanted not in labels:
            raise KeyError(f"{wanted} missing from index")
        position = labels.index(wanted)
        start = offsets[position]
        end = offsets[position + 1] - 1 if position + 1 < len(offsets) else start + 8_000_000
        ranges[wanted] = (start, end)
    return ranges


def decode_grib(message: bytes) -> tuple[np.ndarray, int, int]:
    """One GRIB2 record → (values, width, height), north-first row order."""
    import eccodes

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as handle:
        handle.write(message)
        path = handle.name
    try:
        with open(path, "rb") as stream:
            handle_id = eccodes.codes_grib_new_from_file(stream)
            if handle_id is None:
                raise ValueError("not a GRIB message")
            try:
                width = eccodes.codes_get(handle_id, "Ni")
                height = eccodes.codes_get(handle_id, "Nj")
                values = eccodes.codes_get_values(handle_id).reshape(height, width)
                # GFS ships rows from the north pole down and columns from 0°E
                # eastwards; the app's texture is −180°…180°, same row order.
                values = np.roll(values, width // 2, axis=1)
                return values, width, height
            finally:
                eccodes.codes_release(handle_id)
    finally:
        os.unlink(path)


def downsample(values: np.ndarray, step: float) -> np.ndarray:
    """0.25° grid → whatever `step` asks for, by plain block averaging."""
    factor = int(round(step / 0.25))
    if factor <= 1:
        return values
    height, width = values.shape
    # The last row (the south pole) is dropped if it doesn't divide evenly;
    # nothing in the app draws latitude 90 anyway.
    height -= height % factor
    width -= width % factor
    trimmed = values[:height, :width]
    return trimmed.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def encode(u: np.ndarray, v: np.ndarray) -> Image.Image:
    def channel(values: np.ndarray) -> np.ndarray:
        scaled = (values + SPEED_RANGE) / (2 * SPEED_RANGE) * 255
        return np.clip(scaled, 0, 255).astype(np.uint8)

    height, width = u.shape
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[..., 0] = channel(u)
    pixels[..., 1] = channel(v)
    pixels[..., 3] = 255
    return Image.fromarray(pixels)


def main() -> int:
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "site/wind"
    os.makedirs(output_dir, exist_ok=True)

    for cycle in latest_cycles(dt.datetime.now(dt.timezone.utc)):
        stamp = cycle.strftime("%Y%m%d")
        hour = cycle.strftime("%H")
        base = f"{BUCKET}/gfs.{stamp}/{hour}/atmos/gfs.t{hour}z.pgrb2.0p25.f000"
        try:
            index = fetch(f"{base}.idx").decode("utf-8")
            ranges = index_ranges(index)
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as error:
            print(f"cycle {stamp} {hour}Z unavailable ({error}); trying the one before")
            continue

        print(f"using GFS {stamp} {hour}Z")
        fields = {}
        for name, byte_range in ranges.items():
            size = byte_range[1] - byte_range[0] + 1
            print(f"  {name}: {size / 1_048_576:.2f} MiB")
            values, width, height = decode_grib(fetch(base, byte_range))
            fields[name[:4]] = downsample(values, OUTPUT_STEP_DEGREES)

        image = encode(fields["UGRD"], fields["VGRD"])
        image_path = os.path.join(output_dir, "wind.png")
        image.save(image_path, optimize=True)

        manifest = {
            "model": "NOAA GFS 0.25° analysis",
            "cycle": cycle.strftime("%Y-%m-%dT%H:00:00Z"),
            "builtAt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "width": image.width,
            "height": image.height,
            "stepDegrees": OUTPUT_STEP_DEGREES,
            "speedRange": SPEED_RANGE,
            "encoding": "R=eastward, G=northward, linear over ±speedRange m/s",
        }
        with open(os.path.join(output_dir, "wind.json"), "w") as handle:
            json.dump(manifest, handle, indent=2)

        size = os.path.getsize(image_path)
        print(f"wrote {image_path} — {image.width}×{image.height}, {size / 1024:.0f} KiB")
        return 0

    print("no GFS cycle available", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
