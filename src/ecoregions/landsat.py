"""Stage ``landsat``: a 1984-1993 Landsat 5 surface-reflectance composite and the land-surface layers drawn from it.

Imagery from before most modern canal colonies and city growth shows natural surfaces that later disappeared, which
suits a map of natural communities. For each Landsat tile (WRS-2 path/row) over the study area, up to
``landsat.scenes`` clear scenes from the same season (``landsat.months``, June-September by default; different years
first) (Collection 2 Level-2, Microsoft Planetary Computer) are read
at the ``landsat.overview`` level of their cloud-optimised files (8: about 240 m), masked with the QA band (fill,
cloud, cloud shadow, cirrus), and reduced to a per-pixel median of six bands; one int16 GeoTIFF per tile is kept in
``raw/regions/<area>/landsat``. The tiles are then mosaicked onto the 270 m grid as:

- ``ls_bare``: bare-soil index ((SWIR1 + red) - (NIR + blue)) / ((SWIR1 + red) + (NIR + blue));
- ``ls_green``: NDVI;
- ``ls_bright``: mean visible and near-infrared reflectance (salt crusts, bare sand and rock are bright);
- ``ls_salt``: SWIR2 / SWIR1 (gypsum and salt crusts absorb in SWIR2 less than vegetation and moist soil do).

Checked on the Thal (17 Nov 2026): bare-soil index 0.120 inside the sand desert against 0.023 in a 20 km ring.
"""
from __future__ import annotations

import logging

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from .config import Paths
from .grid import Grid, write

log = logging.getLogger(__name__)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")


def scenes(bbox, years, max_cloud, per_tile, months=(6, 9)) -> dict:
    """Clearest Landsat 5 scenes per tile, all from the same season (``months``, inclusive) and from different years
    where possible: tiles mixing seasons gave each tile its own greenness and brightness, seen as rectangles."""
    body = {"collections": ["landsat-c2-l2"], "bbox": list(bbox), "datetime": f"{years[0]}-01-01/{years[1]}-12-31",
            "query": {"eo:cloud_cover": {"lt": max_cloud}, "platform": {"in": ["landsat-5"]}}, "limit": 1000}
    feats, url = [], STAC
    while url:
        r = requests.post(url, json=body, timeout=120)
        r.raise_for_status()
        d = r.json()
        feats += d["features"]
        nxt = [link for link in d.get("links", []) if link.get("rel") == "next"]
        url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    return pick(feats, per_tile, months)


def pick(feats: list, per_tile: int, months=(6, 9)) -> dict:
    """Per tile, the clearest in-season scenes: one per year first, then the next clearest. A tile with fewer than two
    in-season scenes takes the season widened by a month at each end."""
    by, wide = {}, {}
    for f in sorted(feats, key=lambda f: f["properties"]["eo:cloud_cover"]):
        p = f["properties"]
        k, m = f"{p['landsat:wrs_path']}{p['landsat:wrs_row']}", int(p["datetime"][5:7])
        if months[0] <= m <= months[1]:
            by.setdefault(k, []).append(f)
        if months[0] - 1 <= m <= months[1] + 1:
            wide.setdefault(k, []).append(f)
    by.update({k: fs for k, fs in wide.items() if len(by.get(k, [])) < 2})
    out = {}
    for k, fs in by.items():
        years, first = set(), []
        for f in fs:
            if f["properties"]["datetime"][:4] not in years:
                years.add(f["properties"]["datetime"][:4])
                first.append(f)
        out[k] = (first + [f for f in fs if f not in first])[:per_tile]
    return out


def tile_median(fs: list, token: str, overview: int, out) -> None:
    """Per-pixel median of the clear observations of one tile's scenes, six bands, reflectance x 10000, int16."""
    stack, prof = [], None
    for f in fs:
        with rasterio.open(f["assets"]["qa_pixel"]["href"] + "?" + token) as q:
            shp = (q.height // overview, q.width // overview)
            qa = q.read(1, out_shape=shp, resampling=Resampling.nearest)
            if prof is None:
                tr = q.transform * q.transform.scale(q.width / shp[1], q.height / shp[0])
                prof = dict(driver="GTiff", height=shp[0], width=shp[1], count=len(BANDS), dtype="int16", crs=q.crs,
                            transform=tr, nodata=-32768, compress="deflate", tiled=True)
        if qa.shape != (prof["height"], prof["width"]):
            continue  # a scene framed differently from the tile's first; skip rather than misalign
        clear = (qa & 0b11111) == 0  # fill, dilated cloud, cirrus, cloud, shadow
        b = np.full((len(BANDS), *qa.shape), np.nan, "float32")
        for i, name in enumerate(BANDS):
            with rasterio.open(f["assets"][name]["href"] + "?" + token) as s:
                a = s.read(1, out_shape=qa.shape, resampling=Resampling.average).astype("float32") * 0.0000275 - 0.2
            b[i] = np.where(clear, a, np.nan)
        stack.append(b)
    if not stack:
        return
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.stack(stack), axis=0)
    med = np.where(np.isfinite(med), np.clip(med * 10000, -32000, 32000), -32768).astype("int16")
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(med)


INDICES = {
    "ls_bare": lambda b: ((b["swir16"] + b["red"]) - (b["nir08"] + b["blue"])) / ((b["swir16"] + b["red"]) + (b["nir08"] + b["blue"])),
    "ls_green": lambda b: (b["nir08"] - b["red"]) / (b["nir08"] + b["red"]),
    "ls_bright": lambda b: (b["blue"] + b["green"] + b["red"] + b["nir08"]) / 4,
    "ls_salt": lambda b: b["swir22"] / b["swir16"],
}


def mosaic_index(tiles: list, name: str, grid: Grid) -> np.ndarray:
    """One index computed in each tile, then mosaicked on ``grid`` (mean where tiles overlap); one layer in memory."""
    acc = np.zeros(grid.shape, "float32")
    n = np.zeros(grid.shape, "uint8")
    out = np.empty(grid.shape, "float32")
    for t in tiles:
        with rasterio.open(t) as s:
            a = s.read().astype("float32")
            a[a == -32768] = np.nan
            with np.errstate(all="ignore"):
                idx = INDICES[name](dict(zip(BANDS, a / 10000))).astype("float32")
            out.fill(np.nan)
            reproject(idx, out, src_transform=s.transform, src_crs=s.crs, dst_transform=grid.transform,
                      dst_crs=grid.rio_crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
        ok = np.isfinite(out)
        acc[ok] += out[ok]
        n[ok] += 1
    with np.errstate(all="ignore"):
        acc /= n
    acc[n == 0] = np.nan
    return acc


def run(cfg) -> None:
    paths = Paths(cfg)
    lc = {"years": [1984, 1993], "max_cloud": 10, "scenes": 5, "months": [6, 9], "overview": 8, **cfg.get("landsat", {})}
    grid = Grid.load(paths.grid_json).coarsen(cfg["grid"]["coarse_factor"])
    bbox = transform_bounds(grid.rio_crs, "EPSG:4326", *grid.bounds)
    raw = paths.raw_region / "landsat"
    raw.mkdir(parents=True, exist_ok=True)
    by = scenes(bbox, lc["years"], lc["max_cloud"], lc["scenes"], lc["months"])
    from shapely.geometry import shape

    from .boundaries import load_boundaries, mapped_area

    area = mapped_area(cfg, paths, *load_boundaries(paths)).to_crs(grid.rio_crs)
    area = area.buffer(1000 * (cfg["grid"].get("overlap") or {}).get("km", 0) + 20000).to_crs(4326).union_all()  # strip + margin
    by = {k: fs for k, fs in by.items() if shape(fs[0]["geometry"]).intersects(area)}
    log.info("landsat: %d tiles, %d scenes (%d-%d, cloud < %d %%)", len(by), sum(map(len, by.values())), *lc["years"],
             lc["max_cloud"])
    token = requests.get(TOKEN, timeout=60).json()["token"]
    for i, (k, fs) in enumerate(sorted(by.items())):
        out = raw / f"{k}.tif"
        if out.exists():
            continue
        if i % 20 == 0:  # the signed URLs expire after about an hour
            token = requests.get(TOKEN, timeout=60).json()["token"]
        try:
            tile_median(fs, token, lc["overview"], out)
        except (rasterio.errors.RasterioIOError, requests.RequestException) as e:
            log.warning("landsat: tile %s failed (%s); it is retried on the next run", k, e)
            out.unlink(missing_ok=True)
            continue
        log.info("landsat: tile %s (%d of %d)", k, i + 1, len(by))
    tiles = sorted(raw.glob("*.tif"))
    for name in INDICES:
        a = mosaic_index(tiles, name, grid)
        write(paths.coarse(name), a, grid, nodata=np.nan)
        log.info("landsat: %s %.3f..%.3f, %.1f %% of cells without a clear observation", name,
                 np.nanpercentile(a, 1), np.nanpercentile(a, 99), 100 * np.isnan(a).mean())
