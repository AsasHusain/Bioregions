"""Do political borders show through a map? Share of its internal region boundaries lying along international borders.

A region boundary that follows a border for long stretches is suspect: a parcel seam, or range maps drawn to national
borders. The observed share (boundary length within ``tol_km`` of an international border) is compared with the share
when the borders are shifted 30-80 km in random directions (same shapes, unrelated positions); every stretch of at
least ``min_km`` along a border is listed with its location. The outline of the mapped area is not counted.

    PYTHONPATH=src python -m ecoregions.bordercheck outputs/pak-afg-irn/ecoregions.gpkg:faunal_regions \
        outputs/central-asia/ecoregions.gpkg:faunal_regions
"""
from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely import affinity

from .config import ROOT


def international_borders(countries: gpd.GeoDataFrame) -> shapely.Geometry:
    """Land borders between countries (coastlines removed), in the frame's CRS."""
    lines = shapely.union_all(countries.boundary.values)
    coast = shapely.union_all(countries.geometry.values).boundary
    return lines.difference(coast.buffer(500))


def internal_boundaries(regions: gpd.GeoDataFrame) -> shapely.Geometry:
    """Edges between regions, without the outline of the whole mapped area."""
    lines = shapely.union_all(regions.boundary.values)
    return lines.difference(shapely.union_all(regions.geometry.values).boundary.buffer(500))


def coincidence(bound, borders, tol_m: float) -> float:
    return bound.intersection(borders.buffer(tol_m)).length / bound.length if bound.length else 0.0


def stretches(bound, borders, tol_m: float, min_m: float) -> list:
    near = shapely.line_merge(bound.intersection(borders.buffer(tol_m)))
    parts = getattr(near, "geoms", [near])
    return [p for p in parts if p.length >= min_m]


def check(regions: gpd.GeoDataFrame, countries: gpd.GeoDataFrame, tol_km=3.0, min_km=50.0, runs=20, seed=1) -> dict:
    """``regions`` and ``countries`` in one metric CRS. Observed share, shifted-border shares, long stretches."""
    bound = internal_boundaries(regions)
    borders = international_borders(countries).intersection(shapely.union_all(regions.geometry.values).buffer(100000))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(runs):
        d, a = rng.uniform(30000, 80000), rng.uniform(0, 2 * np.pi)
        null.append(coincidence(bound, affinity.translate(borders, d * np.cos(a), d * np.sin(a)), tol_km * 1000))
    obs = coincidence(bound, borders, tol_km * 1000)
    return dict(observed=obs, shifted_mean=float(np.mean(null)), shifted_max=float(np.max(null)),
                boundary_km=bound.length / 1000, stretches=stretches(bound, borders, tol_km * 1000, min_km * 1000))


def contrast(lines, rasters: list, transform, step_m=2000.0, off_m=3000.0) -> float:
    """Median over points every ``step_m`` along ``lines`` of the summed relative difference of ``rasters`` (each
    scaled by its own spread) between the points ``off_m`` either side of the line."""
    from shapely.geometry import LineString

    inv = ~transform
    diffs = []
    for ln in getattr(lines, "geoms", [lines]):
        if not isinstance(ln, LineString) or ln.length < 2 * step_m:
            continue
        for d in np.arange(step_m / 2, ln.length - step_m / 2, step_m):
            p, q = ln.interpolate(d - 50), ln.interpolate(d + 50)
            nx, ny = -(q.y - p.y), q.x - p.x
            n = np.hypot(nx, ny) or 1.0
            m = ln.interpolate(d)
            sides = []
            for sgn in (1, -1):
                c, r = inv * (m.x + sgn * off_m * nx / n, m.y + sgn * off_m * ny / n)
                sides.append((int(r), int(c)))
            v = 0.0
            for a, spread in rasters:
                (r1, c1), (r2, c2) = sides
                if not (0 <= r1 < a.shape[0] and 0 <= c1 < a.shape[1] and 0 <= r2 < a.shape[0] and 0 <= c2 < a.shape[1]):
                    v = np.nan
                    break
                v += abs(float(a[r1, c1]) - float(a[r2, c2])) / spread
            diffs.append(v)
    return float(np.nanmedian(diffs)) if diffs else np.nan


def run(cfg) -> None:
    """Gate before a freeze: every stretch of a Floral or Faunal Ecoregion boundary that runs along an international
    border for ``bordercheck.min_km`` or more must show at least the terrain and rainfall contrast of the map's other
    region boundaries (a crest, an escarpment); otherwise it is a political artefact and the stage fails."""
    import logging

    from .config import Paths
    from .grid import Grid, read

    log = logging.getLogger(__name__)
    paths, bc = Paths(cfg), cfg.get("bordercheck", {})
    tol, min_km = bc.get("tol_km", 3.0), bc.get("min_km", 50.0)
    grid = Grid.load(paths.grid_json)
    f = 3  # 270 m is plenty for a 3 km offset
    rasters = []
    for name in ("dem", "clim_bio12"):
        a = read(paths.raster(name))[::f, ::f].astype("float32")
        rasters.append((a, float(np.nanpercentile(a, 90) - np.nanpercentile(a, 10)) or 1.0))
    tr = grid.transform * grid.transform.scale(f, f)
    countries = gpd.read_file(paths.raw / "naturalearth" / "ne_10m_admin_0_countries.shp").to_crs(grid.crs)
    gpkg = paths.outputs / "ecoregions.gpkg"
    bad, rows = 0, []
    for layer in ("ecoregions", "faunal_regions"):
        regions = gpd.read_file(gpkg, layer=layer).to_crs(grid.crs)
        cty = countries[countries.intersects(regions.union_all().buffer(200000))]
        r = check(regions, cty, tol, min_km)
        bound = internal_boundaries(regions)
        borders = international_borders(cty)
        typical = contrast(bound.difference(borders.buffer(tol * 1000)), rasters, tr)
        log.info("bordercheck %s: %.1f %% of %.0f km of boundaries along borders (shifted borders %.1f %%, max %.1f %%); "
                 "typical contrast %.2f", layer, 100 * r["observed"], r["boundary_km"], 100 * r["shifted_mean"],
                 100 * r["shifted_max"], typical)
        for s in sorted(r["stretches"], key=lambda g: -g.length):
            c = contrast(s, rasters, tr)
            m = gpd.GeoSeries([s.interpolate(0.5, normalized=True)], crs=grid.crs).to_crs(4326).iloc[0]
            natural = np.isfinite(c) and c >= typical
            bad += not natural
            rows.append(dict(layer=layer, km=round(s.length / 1000), lat=round(m.y, 2), lon=round(m.x, 2),
                             contrast=round(c, 2), typical=round(typical, 2), verdict="natural break" if natural else "ARTEFACT"))
            log.log(logging.INFO if natural else logging.ERROR, "  %s: %.0f km along a border near %.2f N, %.2f E; contrast %.2f (typical %.2f): %s",
                    layer, s.length / 1000, m.y, m.x, c, typical, rows[-1]["verdict"])
    pd.DataFrame(rows, columns=["layer", "km", "lat", "lon", "contrast", "typical", "verdict"]).to_csv(
        paths.outputs / "bordercheck.csv", index=False)
    if bad:
        raise SystemExit(f"bordercheck: {bad} boundary stretch(es) follow a political border without a natural break")


def main(specs: list[str]) -> None:
    parts = [gpd.read_file(ROOT / f, layer=layer) for f, layer in (s.rsplit(":", 1) for s in specs)]
    crs = parts[0].crs
    c = shapely.union_all([p.to_crs(4326).union_all() for p in parts]).centroid
    aeqd = f"+proj=aeqd +lat_0={c.y:.2f} +lon_0={c.x:.2f} +datum=WGS84 +units=m"
    regions = pd.concat([p.to_crs(crs).assign(src=s) for p, s in zip(parts, specs)]).to_crs(aeqd)
    countries = gpd.read_file(ROOT / "data/raw/naturalearth/ne_10m_admin_0_countries.shp").to_crs(aeqd)
    countries = countries[countries.intersects(regions.union_all().buffer(200000))]
    r = check(regions, countries)
    print(f"internal boundaries {r['boundary_km']:,.0f} km; along international borders {100 * r['observed']:.1f} % "
          f"(borders shifted at random: mean {100 * r['shifted_mean']:.1f} %, max {100 * r['shifted_max']:.1f} %)")
    for s in sorted(r["stretches"], key=lambda g: -g.length):
        m = gpd.GeoSeries([s.interpolate(0.5, normalized=True)], crs=aeqd).to_crs(4326).iloc[0]
        print(f"  {s.length / 1000:5.0f} km along a border near {m.y:.2f} N, {m.x:.2f} E")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:  # self-check: a region edge laid on a border is found; one 50 km away is not
        from shapely.geometry import box
        cty = gpd.GeoDataFrame(geometry=[box(0, 0, 100000, 100000), box(100000, 0, 200000, 100000)])
        on = gpd.GeoDataFrame(geometry=[box(0, 0, 100000, 100000), box(100000, 0, 200000, 100000)])
        off = gpd.GeoDataFrame(geometry=[box(0, 0, 150000, 100000), box(150000, 0, 200000, 100000)])
        assert check(on, cty, runs=2)["observed"] > 0.99 and len(check(on, cty, runs=2)["stretches"]) == 1
        assert check(off, cty, runs=2)["observed"] < 0.01
        from affine import Affine
        from shapely.geometry import LineString
        step = np.zeros((100, 100), "float32")
        step[:, 50:] = 1000.0  # a 1,000 m cliff at x = 50 km (1 km cells)
        tr = Affine(1000, 0, 0, 0, -1000, 100000)
        assert contrast(LineString([(50000, 10000), (50000, 90000)]), [(step, 1000.0)], tr) > 0.9  # on the cliff
        assert contrast(LineString([(20000, 10000), (20000, 90000)]), [(step, 1000.0)], tr) == 0.0  # flat ground
        print("ok")
