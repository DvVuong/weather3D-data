# weather3D-data

Static data for the **weather3D** iOS app. No server, no API key, no cost:
a GitHub Actions job rebuilds a small texture four times a day and GitHub
Pages serves it as a plain file over a CDN.

This repository is public only because GitHub Pages requires it on a free
plan — the app's own source stays private. Nothing here is secret: every
byte is derived from public-domain NOAA data.

## What it publishes

| File | What it is |
| --- | --- |
| [`wind/wind.png`](https://dvvuong.github.io/weather3D-data/wind/wind.png) | 10 m wind field. `R` = eastward, `G` = northward, each 0–255 mapped over ±50 m/s. Equirectangular, 0.5°. |
| [`wind/wind.json`](https://dvvuong.github.io/weather3D-data/wind/wind.json) | Which GFS run the PNG came from, and when it was built. |

The app decodes these in `Weather/Services/WindFieldService.swift`.

## How it works

`tools/build_wind_field.py` reads NOAA's GFS `.idx` index to find the byte
offsets of just the two 10 m wind records, fetches those ranges with HTTP
`Range` requests instead of the ~500 MB GRIB2 file, decodes them with
ecCodes, downsamples to 0.5° and writes the PNG.

GFS publishes four cycles a day (00/06/12/18 UTC), each landing three to
five hours after its nominal time, so the workflow runs at 05:10, 11:10,
17:10 and 23:10 UTC. If the newest cycle is not up yet the script walks
back through earlier ones, so a late publication costs freshness, not the
run.

Run it by hand from the **Actions** tab → **Wind field** → **Run workflow**.

## Data source

[NOAA GFS](https://registry.opendata.aws/noaa-gfs-bdp-pds/) via the AWS Open
Data registry. U.S. government work, public domain — free to use and
redistribute, no licence key and no attribution requirement (the app credits
NOAA anyway, in Settings).
