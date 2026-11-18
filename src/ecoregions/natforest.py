"""Stage ``natforest``: Natural Forests of the World 2020 (Neumann et al. 2025, Sci Data 12:1715; CC BY 4.0) on the
coarse grid.

The map gives, at 10 m, the probability (0-250) that a pixel is natural forest rather than planted forest, tree crops
or non-forest. It is published as one zip per UTM zone (Figshare+ 10.25452/figshare.plus.28881731), each holding
cloud-optimised GeoTIFF tiles with overviews stored ahead of the full-resolution data. For every tile over the mapped
area (plus the overlap and a 20 km margin) only the leading bytes up to the end of the ``natforest.overview`` level
(8: 80 m) are streamed out of the remote zip; those 80 m rasters are cached in ``raw/natural_forests``. Written:

- ``nf_prob``: mean natural-forest probability, 0-1;
- ``nf_share``: share of 80 m pixels at or above ``natforest.threshold`` (0.52, the authors' recommended cut).

An independent check on forest types (not a rule input): weak on sparse woodland, savanna and orchards.
"""
from __future__ import annotations

import io
import logging
import zipfile

import numpy as np
import rasterio
import requests
import tifffile
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds

from .config import Paths
from .grid import Grid, write

log = logging.getLogger(__name__)
ARTICLE = "https://api.figshare.com/v2/articles/28881731"
NODATA = 255


class HttpFile(io.RawIOBase):
    """A remote file read by HTTP range requests; the signed link is renewed when it expires."""

    def __init__(self, url: str):
        self.src, self.pos = url, 0
        self._sign()

    def _sign(self):
        r = requests.get(self.src, headers={"Range": "bytes=0-0"}, timeout=60)
        r.raise_for_status()
        self.url, self.size = r.url, int(r.headers["Content-Range"].split("/")[1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, o, whence=0):
        self.pos = o if whence == 0 else self.pos + o if whence == 1 else self.size + o
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        rng = {"Range": f"bytes={self.pos}-{min(self.pos + len(b), self.size) - 1}"}
        for attempt in range(4):
            r = requests.get(self.url, headers=rng, timeout=180)
            if r.status_code == 206:
                break
            if attempt == 3:
                r.raise_for_status()
            self._sign()
        d = r.content
        b[: len(d)] = d
        self.pos += len(d)
        return len(d)


def zone_files() -> dict:
    return {f["name"]: f["download_url"] for f in requests.get(ARTICLE, timeout=60).json()["files"]}


def overview(head: bytes, level: int):
    """From the first bytes of a COG: its 4326 bounds, the index of the ``level`` overview, and the bytes needed to
    read that overview (overviews are stored ahead of the full-resolution data)."""
    with tifffile.TiffFile(io.BytesIO(head)) as t:
        full = t.pages[0]
        ovs = [p for p in t.pages[1:] if p.imagewidth < full.imagewidth]
        i = next(i for i, p in enumerate(ovs) if round(full.imagewidth / p.imagewidth) == level)
        need = max(o + c for o, c in zip(ovs[i].dataoffsets, ovs[i].databytecounts) if c)
    with rasterio.MemoryFile(head) as mf, mf.open() as s:
        bounds = transform_bounds(s.crs, "EPSG:4326", *s.bounds)
    return bounds, i, need


def fetch(url: str, members_wanted, out_dir, level: int, area4326) -> int:
    """Stream each wanted member's leading bytes up to its ``level`` overview; keep those over ``area4326``."""
    from shapely.geometry import box

    z = zipfile.ZipFile(io.BufferedReader(HttpFile(url), 1 << 20))
    kept = 0
    for info in z.infolist():
        if not info.filename.endswith(".tif") or not members_wanted(info.filename):
            continue
        out = out_dir / info.filename.rsplit("/", 1)[-1]
        if out.exists() or out.with_suffix(".none").exists():
            kept += out.exists()
            continue
        with z.open(info) as m:
            buf = bytearray()

            def prefix(n):
                while len(buf) < n:
                    d = m.read(max(n - len(buf), 1 << 20))
                    if not d:
                        break
                    buf.extend(d)
                return bytes(buf[:n])

            bounds, i, need = overview(prefix(1 << 16), level)
            if not box(*bounds).intersects(area4326):
                out.with_suffix(".none").touch()  # outside the area: remembered so the header is not read again
                continue
            data = prefix(need)
        with rasterio.MemoryFile(data) as mf, mf.open(overview_level=i) as s:
            a, prof = s.read(1), s.profile
        prof.update(driver="GTiff", nodata=NODATA, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        prof.pop("photometric", None)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(a, 1)
        kept += 1
    return kept


def run(cfg) -> None:
    paths = Paths(cfg)
    nc = {"overview": 8, "threshold": 0.52, **cfg.get("natforest", {})}
    grid = Grid.load(paths.grid_json).coarsen(cfg["grid"]["coarse_factor"])
    from .boundaries import load_boundaries, mapped_area

    area = mapped_area(cfg, paths, *load_boundaries(paths)).to_crs(grid.rio_crs)
    area = area.buffer(1000 * (cfg["grid"].get("overlap") or {}).get("km", 0) + 20000).to_crs(4326).union_all()
    w, s, e, n = area.bounds
    zones = range(int((w + 180) // 6) + 1, int((e + 180) // 6) + 2)
    bands = "CDEFGHJKLMNPQRSTUVWX"  # MGRS latitude bands, 8 degrees each from 80 S; tiles are named raster_<zone><band>_...
    near = {b for i, b in enumerate(bands) if -80 + 8 * i < n and -80 + 8 * (i + 1) > s}
    files = zone_files()
    raw = paths.raw / "natural_forests"
    for zn in zones:
        d = raw / f"zone_{zn}"
        d.mkdir(parents=True, exist_ok=True)
        k = fetch(files[f"zone_{zn}.zip"], lambda name: name.rsplit("/", 1)[-1].split("_")[1][-1] in near, d,
                  nc["overview"], area)
        log.info("natforest: zone %d, %d tiles over the area", zn, k)
    tiles = [t for zn in zones for t in sorted((raw / f"zone_{zn}").glob("*.tif"))]
    prob, share, n = (np.zeros(grid.shape, "float32") for _ in range(3))
    out = np.empty(grid.shape, "float32")
    for t in tiles:
        with rasterio.open(t) as s:
            a = s.read(1).astype("float32")
            a[a == NODATA] = np.nan
            for acc, src in ((prob, a / 250), (share, np.where(np.isnan(a), np.nan, (a / 250 >= nc["threshold"]) * 1.0))):
                out.fill(np.nan)
                reproject(src.astype("float32"), out, src_transform=s.transform, src_crs=s.crs,
                          dst_transform=grid.transform, dst_crs=grid.rio_crs, src_nodata=np.nan, dst_nodata=np.nan,
                          resampling=Resampling.average)
                ok = np.isfinite(out)
                acc[ok] += out[ok]
                if acc is prob:
                    n[ok] += 1
    with np.errstate(all="ignore"):
        prob /= n
        share /= n
    prob[n == 0] = share[n == 0] = np.nan
    for name, a in (("nf_prob", prob), ("nf_share", share)):
        write(paths.coarse(name), a, grid, nodata=np.nan)
    log.info("natforest: %d tiles; natural forest on %.1f %% of observed cells (share >= 0.5), %.1f %% unobserved",
             len(tiles), 100 * np.nanmean(share >= 0.5), 100 * np.isnan(share).mean())
