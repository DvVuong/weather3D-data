#!/usr/bin/env python3
"""Bakes one forecast file for every country the globe draws, from NOAA GFS.

Why this exists: the app used to ask WeatherKit for each of the 171 countries
separately, every time the globe went stale — about 171 billed calls per
refresh, *per device*. That scales with the number of people using the app
rather than with the amount of weather in the world. The readings are
identical for everyone, so they are baked once here, on a schedule, and
published as a single file the app fetches in one request.

Source is NOAA's Global Forecast System, straight from the public AWS mirror.
Public domain, no key, no rate plan, no commercial restriction — the same
model the wind layer already draws from. NOAA asks only for attribution,
which `SettingsScreen` already carries.

Only the handful of variables the map actually draws are downloaded, using the
`.idx` sidecar each GRIB file publishes: it lists the byte offset of every
message, so a range request fetches one variable (a few hundred KB) instead of
the whole 535 MB file.

Output matches Swift's `[String: StateTimeline]` exactly, so the app decodes it
with no conversion layer.

The country list is `tools/country-points.json`: id, name, lat, lon for each
of the 171, stripped out of the app's `globe.json` (which is 2.1 MB of
polygon geometry this has no use for). `globe.json` itself also works, since
the two share a shape.

Usage:  python3 tools/build_country_field.py tools/country-points.json out/countries.json
        python3 tools/build_country_field.py tools/country-points.json out.json --hours 6 --days 2
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import eccodes
import numpy as np

BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
PRODUCT = "pgrb2.0p25"

# GRIB labels exactly as they appear in the `.idx`, mapped to the short names
# the rest of this script uses.
VARIABLES = {
    "temperature": "TMP:2 m above ground",
    "precipitation": "APCP:surface",
    "windU": "UGRD:10 m above ground",
    "windV": "VGRD:10 m above ground",
    "cloud": "TCDC:entire atmosphere",
    # Convective energy, the one thing that separates "raining" from
    # "thunderstorm" — without it the app's thunderstorm icon could never
    # appear.
    "cape": "CAPE:surface",
}


def fetch(url, byte_range=None, timeout=120):
    request = urllib.request.Request(url)
    if byte_range:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def step_key(day, run, step):
    return f"gfs.{day}/{run:02d}/atmos/gfs.t{run:02d}z.{PRODUCT}.f{step:03d}"


def latest_run(max_age_hours=30):
    """The newest run whose first forecast hour has finished uploading."""
    now = datetime.now(timezone.utc)
    for back in range(0, max_age_hours, 6):
        moment = now - timedelta(hours=back)
        run = (moment.hour // 6) * 6
        day = moment.strftime("%Y%m%d")
        try:
            request = urllib.request.Request(f"{BUCKET}/{step_key(day, run, 1)}.idx",
                                             method="HEAD")
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status == 200:
                    return day, run, moment.replace(hour=run, minute=0, second=0, microsecond=0)
        except urllib.error.URLError:
            continue
    raise SystemExit("no GFS run available")


def byte_ranges(day, run, step):
    """Where each wanted variable sits inside this step's GRIB file."""
    lines = fetch(f"{BUCKET}/{step_key(day, run, step)}.idx", timeout=60).decode().strip().split("\n")
    starts = [int(line.split(":")[1]) for line in lines]
    found = {}
    for index, line in enumerate(lines):
        for name, label in VARIABLES.items():
            # First match wins: several messages can share a label (an
            # instantaneous field and an averaged one), and the first is the
            # instantaneous one the map wants.
            if name not in found and f":{label}:" in line:
                end = starts[index + 1] - 1 if index + 1 < len(starts) else ""
                found[name] = (starts[index], end)
    return found


def grid(message):
    handle = eccodes.codes_new_from_message(message)
    try:
        ni = eccodes.codes_get(handle, "Ni")
        nj = eccodes.codes_get(handle, "Nj")
        return {
            "values": eccodes.codes_get_values(handle).reshape(nj, ni),
            "lat1": eccodes.codes_get(handle, "latitudeOfFirstGridPointInDegrees"),
            "lon1": eccodes.codes_get(handle, "longitudeOfFirstGridPointInDegrees"),
            "di": eccodes.codes_get(handle, "iDirectionIncrementInDegrees"),
            "dj": eccodes.codes_get(handle, "jDirectionIncrementInDegrees"),
            "ni": ni, "nj": nj,
        }
    finally:
        eccodes.codes_release(handle)


def sample(field, lat, lon):
    """Nearest grid point. Latitudes run north to south in these files."""
    j = int(round((field["lat1"] - lat) / field["dj"]))
    i = int(round((lon % 360 - field["lon1"]) / field["di"])) % field["ni"]
    return float(field["values"][min(max(j, 0), field["nj"] - 1), i])


def read_step(day, run, step, names):
    ranges = byte_ranges(day, run, step)
    url = f"{BUCKET}/{step_key(day, run, step)}"
    wanted = [name for name in names if name in ranges]
    # Downloads run together because this job is almost entirely spent
    # waiting on the network; decoding each message is milliseconds.
    with ThreadPoolExecutor(max_workers=len(wanted) or 1) as pool:
        messages = pool.map(lambda name: (name, fetch(url, ranges[name])), wanted)
        return {name: grid(message) for name, message in messages}


def condition(celsius, rain_mm_h, cloud_fraction, cape):
    """GFS reports physics, not a weather code, so the icon is derived here.

    Thresholds follow the same bands the app's own rain palette uses, so a
    country drawn as "heavy rain" on the rain layer carries the heavy-rain
    icon on the temperature one.
    """
    freezing = celsius is not None and celsius <= 0.5
    if rain_mm_h is not None and rain_mm_h >= 0.2:
        if cape is not None and cape >= 1000 and not freezing:
            return "thunderstorm"
        if freezing:
            return "snow"
        if rain_mm_h >= 7.5:
            return "heavyRain"
        if rain_mm_h >= 2.5:
            return "rain"
        return "drizzle"
    if cloud_fraction is None:
        return "partlyCloudy"
    if cloud_fraction >= 0.85:
        return "cloudy"
    if cloud_fraction >= 0.35:
        return "partlyCloudy"
    return "clear"


def reading(when, celsius, rain_mm_h, cloud_fraction, cape, wind=None):
    out = {
        "date": when.isoformat(),
        "temperature": round(celsius, 1),
        "condition": condition(celsius, rain_mm_h, cloud_fraction, cape),
    }
    if rain_mm_h is not None:
        out["precipitation"] = round(rain_mm_h, 2)
    if wind is not None:
        speed, direction = wind
        out["windSpeed"] = round(speed, 1)
        out["windDirection"] = round(direction, 1)
    if cloud_fraction is not None:
        out["cloudCover"] = round(cloud_fraction, 3)
    return out


def wind_from(u, v):
    """Speed in km/h and the meteorological direction the wind blows *from*."""
    speed = (u * u + v * v) ** 0.5 * 3.6
    direction = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
    return speed, float(direction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("points", help="country list: tools/country-points.json")
    parser.add_argument("out")
    parser.add_argument("--hours", type=int, default=24, help="length of the hourly series")
    parser.add_argument("--days", type=int, default=10, help="length of the daily series")
    args = parser.parse_args()

    countries = json.load(open(args.points))["countries"]
    points = [(c["id"], c["lat"], c["lon"]) for c in countries]
    day, run, run_time = latest_run()
    print(f"run {day} {run:02d}z", file=sys.stderr)

    series = {cid: {"hourly": [], "daily": []} for cid, _, _ in points}
    now_reading = {}

    # The hourly series: every forecast hour, all variables.
    for step in range(1, args.hours + 1):
        fields = read_step(day, run, step, VARIABLES)
        when = run_time + timedelta(hours=step)
        for cid, lat, lon in points:
            celsius = sample(fields["temperature"], lat, lon) - 273.15
            rain = sample(fields["precipitation"], lat, lon) if "precipitation" in fields else None
            cloud = sample(fields["cloud"], lat, lon) / 100 if "cloud" in fields else None
            cape = sample(fields["cape"], lat, lon) if "cape" in fields else None
            wind = wind_from(sample(fields["windU"], lat, lon), sample(fields["windV"], lat, lon))
            entry = reading(when, celsius, rain, cloud, cape, wind)
            series[cid]["hourly"].append(entry)
            # The app's "now" is the first hour still ahead, the same reading
            # its own WeatherKit path used to show.
            if step == 1:
                now_reading[cid] = entry
        print(f"  hour {step}/{args.hours}", file=sys.stderr)

    # The daily series. Four steps a day, six hours apart: the high is the
    # warmest of them, and the sky is read from whichever of the four lands
    # nearest that country's own local noon.
    #
    # Sampling a single fixed UTC hour instead was tried first and was wrong
    # by half a planet — hour 12 UTC is the middle of the night in Vietnam,
    # so its "daily high" came out ten degrees below the temperature the same
    # file reported for right now.
    for ahead in range(1, args.days + 1):
        base = ahead * 24
        offsets = (0, 6, 12, 18)
        steps = {offset: read_step(day, run, base + offset, VARIABLES) for offset in offsets}
        when = run_time + timedelta(hours=base + 12)
        for cid, lat, lon in points:
            highs = [sample(steps[o]["temperature"], lat, lon) - 273.15 for o in offsets]
            # Local noon expressed as an hour of the UTC day, then the sampled
            # step sitting closest to it.
            local_noon = (12 - lon / 15) % 24
            nearest = min(offsets, key=lambda o: min(abs((run + base + o) % 24 - local_noon),
                                                     24 - abs((run + base + o) % 24 - local_noon)))
            sky = steps[nearest]
            rain = sample(sky["precipitation"], lat, lon) if "precipitation" in sky else None
            cloud = sample(sky["cloud"], lat, lon) / 100 if "cloud" in sky else None
            cape = sample(sky["cape"], lat, lon) if "cape" in sky else None
            series[cid]["daily"].append(reading(when, max(highs), rain, cloud, cape))
        print(f"  day {ahead}/{args.days}", file=sys.stderr)

    out = {
        cid: {
            "stateCode": cid,
            "now": now_reading[cid],
            "hourly": series[cid]["hourly"],
            "daily": series[cid]["daily"],
        }
        for cid, _, _ in points if cid in now_reading
    }
    payload = {"updatedAt": datetime.now(timezone.utc).isoformat(), "countries": out}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    print(f"wrote {args.out}: {len(out)} countries")


if __name__ == "__main__":
    main()
