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
import shapely.ops
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


def side_points(lines, off_m: float, step_m: float, min_len=0.0):
    """Points ``off_m`` either side of ``lines``, every ``step_m`` along each part at least ``min_len`` long
    (two (n, 2) arrays), computed with shapely's array functions."""
    a, b = [], []
    for ln in shapely.get_parts(lines):
        if ln.geom_type != "LineString" or ln.length < max(min_len, step_m):
            continue
        d = np.arange(step_m / 2, ln.length - step_m / 2, step_m)
        if not len(d):
            continue
        p, q, m = (shapely.get_coordinates(shapely.line_interpolate_point(ln, d + k)) for k in (-50.0, 50.0, 0.0))
        n = np.stack([-(q[:, 1] - p[:, 1]), q[:, 0] - p[:, 0]], axis=1)
        n /= np.maximum(np.hypot(n[:, 0], n[:, 1]), 1e-9)[:, None]
        a.append(m + off_m * n)
        b.append(m - off_m * n)
    return (np.vstack(a), np.vstack(b)) if a else (np.empty((0, 2)), np.empty((0, 2)))


def pixels(a, b, transform, shape):
    """Row/column of the point pairs ``a``, ``b`` where both fall inside a raster of ``shape``."""
    inv = ~transform
    c1, r1 = inv * (a[:, 0], a[:, 1])
    c2, r2 = inv * (b[:, 0], b[:, 1])
    r1, c1, r2, c2 = (np.floor(np.asarray(v)).astype(int) for v in (r1, c1, r2, c2))
    ok = (r1 >= 0) & (r1 < shape[0]) & (c1 >= 0) & (c1 < shape[1]) & (r2 >= 0) & (r2 < shape[0]) & (c2 >= 0) & (c2 < shape[1])
    return r1[ok], c1[ok], r2[ok], c2[ok]


def contrast(lines, rasters: list, transform, step_m=2000.0, off_m=3000.0) -> float:
    """Median over points every ``step_m`` along ``lines`` of the summed relative difference of ``rasters`` (each
    scaled by its own spread) between the points ``off_m`` either side of the line."""
    a, b = side_points(lines, off_m, step_m, min_len=2 * step_m)
    if not len(a):
        return np.nan
    r1, c1, r2, c2 = pixels(a, b, transform, rasters[0][0].shape)
    if not len(r1):
        return np.nan
    v = sum(np.abs(arr[r1, c1].astype("float64") - arr[r2, c2]) / spread for arr, spread in rasters)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan


def _rasters(paths, grid, f=3):
    """What makes a natural break, every ``f``-th cell (270 m is plenty for a 3 km offset), each with its p90-p10
    spread: elevation, annual rainfall, soil sand (a dune field's edge on a flat plain) and height above the nearest
    channel (log; a floodplain's edge). Elevation and rainfall alone took the edges of the Thar and of the Indus
    floodplain for arbitrary lines."""
    from .grid import read

    out = []
    for name in ("dem", "clim_bio12", "soil_sand", "hand"):
        if not paths.raster(name).exists():
            continue
        a = read(paths.raster(name))[::f, ::f].astype("float32")
        if name == "hand":
            a = np.log1p(np.clip(a, 0, None))
        out.append((a, float(np.nanpercentile(a, 90) - np.nanpercentile(a, 10)) or 1.0))
    return out, grid.transform * grid.transform.scale(f, f)


def pieces(lines, length_m: float, step_m=2000.0, max_n=60, seed=1) -> list:
    """Up to ``max_n`` pieces of ``length_m`` cut end to end from ``lines`` (picked at random)."""
    out = []
    for ln in getattr(lines, "geoms", [lines]):
        if not hasattr(ln, "interpolate"):
            continue
        for d in np.arange(0, ln.length - length_m, length_m):
            out.append(shapely.ops.substring(ln, d, d + length_m))
    rng = np.random.default_rng(seed)
    return [out[i] for i in rng.permutation(len(out))[:max_n]]


def judge(layer, regions, countries, rasters, tr, tol, min_km, prov=None) -> list[dict]:
    """Every stretch of ``regions``' internal boundaries running ``min_km`` or more within ``tol`` km of an international
    border, with its verdict. With ``prov`` (province edges): a stretch mostly off them lies inside one province, where
    rules on terrain, climate and water draw the line. Otherwise it must be at least as sharp (terrain and rainfall
    contrast) as the least sharp quarter of the layer's boundary pieces of the same length away from borders, or it is
    an ARTEFACT. (Until 2026-11-14 the bar was the median contrast, which half of all natural boundaries fail.)"""
    import logging

    log = logging.getLogger(__name__)
    crs = regions.crs
    cty = countries[countries.intersects(regions.union_all().buffer(200000))]
    r = check(regions, cty, tol, min_km)
    bound, borders = internal_boundaries(regions), international_borders(cty)
    typical = contrast(bound.difference(borders.buffer(tol * 1000)), rasters, tr)
    log.info("bordercheck %s: %.1f %% of %.0f km of boundaries within %.0f km of borders (shifted borders %.1f %%, "
             "max %.1f %%); typical contrast %.2f", layer, 100 * r["observed"], r["boundary_km"], tol,
             100 * r["shifted_mean"], 100 * r["shifted_max"], typical)
    rows = []
    away = shapely.line_merge(bound.difference(borders.buffer(tol * 1000)))
    for st in sorted(r["stretches"], key=lambda g: -g.length):
        c = contrast(st, rasters, tr)
        ref = [contrast(pc, rasters, tr) for pc in pieces(away, st.length)]
        ref = [v for v in ref if np.isfinite(v)]
        bar = float(np.percentile(ref, 25)) if len(ref) >= 8 else typical
        m = gpd.GeoSeries([st.interpolate(0.5, normalized=True)], crs=crs).to_crs(4326).iloc[0]
        on_prov = np.nan
        if prov is not None:
            near = prov.intersection(shapely.box(*st.buffer(tol * 2000).bounds))  # buffering all edges took hours
            on_prov = st.intersection(near.buffer(tol * 1000)).length / st.length
        if prov is not None and on_prov < 0.5:
            verdict = "inside a province (terrain and climate)"
        else:
            verdict = "natural break" if np.isfinite(c) and c >= bar else "ARTEFACT"
        rows.append(dict(layer=layer, km=round(st.length / 1000), lat=round(m.y, 2), lon=round(m.x, 2),
                         on_province_edge=round(on_prov, 2), contrast=round(c, 2), typical=round(typical, 2),
                         bar=round(bar, 2), verdict=verdict))
        log.log(logging.ERROR if verdict == "ARTEFACT" else logging.INFO,
                "  %s: %.0f km along a border near %.2f N, %.2f E; on a province edge %s; contrast %.2f (bar %.2f, "
                "typical %.2f): %s", layer, st.length / 1000, m.y, m.x, "-" if prov is None else f"{100 * on_prov:.0f} %",
                c, bar, typical, verdict)
    return rows


COLS = ["layer", "km", "lat", "lon", "on_province_edge", "contrast", "bar", "typical", "verdict"]


def _countries(paths, crs):
    return gpd.read_file(paths.raw / "naturalearth" / "ne_10m_admin_0_countries.shp").to_crs(crs)


def run(cfg) -> None:
    """Gate before a freeze. A Floral Ecoregion boundary is drawn by rules on terrain, climate and water inside a
    province, so only a stretch along a province edge can carry a border; a Faunal Ecoregion boundary comes from a
    clustering. Every stretch that runs within ``bordercheck.tol_km`` of an international border for
    ``bordercheck.min_km`` or more, and (Floral) lies mostly on a province edge, must show at least the terrain and
    rainfall contrast of the map's other region boundaries (a crest, an escarpment); otherwise it is a political
    artefact and the stage fails. The tolerance is 10 km because source maps carry their own, older border lines:
    the Iran geology sheet's edge ran 5-10 km from Natural Earth's Iran-Afghanistan border."""
    from .config import Paths
    from .grid import Grid

    paths, bc = Paths(cfg), cfg.get("bordercheck", {})
    tol, min_km = bc.get("tol_km", 10.0), bc.get("min_km", 50.0)
    grid = Grid.load(paths.grid_json)
    rasters, tr = _rasters(paths, grid)
    countries = _countries(paths, grid.crs)
    gpkg = paths.outputs / "ecoregions.gpkg"
    prov = internal_boundaries(gpd.read_file(paths.outputs / "physiographic.gpkg", layer="l2_provinces").to_crs(grid.crs))
    rows = []
    for layer in ("ecoregions", "faunal_regions"):
        regions = gpd.read_file(gpkg, layer=layer).to_crs(grid.crs)
        rows += judge(layer, regions, countries, rasters, tr, tol, min_km, prov if layer == "ecoregions" else None)
    pd.DataFrame(rows, columns=COLS).to_csv(paths.outputs / "bordercheck.csv", index=False)
    eco = gpd.read_file(gpkg, layer="ecoregions").to_crs(grid.crs)
    cuts(eco, prov, rasters, tr, min_km).to_csv(paths.qa / "province_cuts.csv", index=False)
    bad = sum(r["verdict"] == "ARTEFACT" for r in rows)
    if bad:
        raise SystemExit(f"bordercheck: {bad} boundary stretch(es) follow a political border without a natural break")


def split_near(lines, other, tol: float, step=1000.0):
    """``lines`` cut into ``step`` pieces and sorted into those within ``tol`` of ``other`` and the rest, each merged
    back into stretches. A spatial index on ``other``'s parts: buffering a whole network of edges runs out of memory."""
    segs = []
    for ln in shapely.get_parts(shapely.line_merge(lines)):
        if ln.length == 0:
            continue
        cut = np.append(np.arange(0, ln.length, step), ln.length)
        segs += [shapely.ops.substring(ln, a, b) for a, b in zip(cut[:-1], cut[1:]) if b > a]
    segs = np.array(segs, dtype=object)
    near = np.zeros(len(segs), bool)
    if len(segs):
        tree = shapely.STRtree(shapely.get_parts(other))
        near[np.unique(tree.query(segs, predicate="dwithin", distance=tol)[0])] = True
    merge = lambda s: shapely.line_merge(shapely.union_all(s)) if len(s) else shapely.LineString()  # noqa: E731
    return merge(segs[near]), merge(segs[~near])


def cuts(eco: gpd.GeoDataFrame, prov, rasters, tr, min_km=50.0) -> pd.DataFrame:
    """Review list: stretches of ``min_km`` or more where the Floral Ecoregion type changes along a province edge that
    has less terrain and rainfall contrast than the least sharp quarter of the map's type boundaries of that length.
    Province edges are drawn between hand-placed seeds; where the land does not change, a rule keyed to a province
    ends there for no reason (the Kirthar scrub ended on a straight line across the Kirthar and Brahui ranges). Each is
    reviewed: the type's rule should follow elevation and climate across the edge, or the seeds should move."""
    import logging

    log = logging.getLogger(__name__)
    tb = internal_boundaries(eco.dissolve("type").reset_index())
    on, off = split_near(tb, prov, 1000.0)
    rows = []
    for st in sorted((g for g in getattr(on, "geoms", [on]) if g.length >= min_km * 1000), key=lambda g: -g.length):
        c = contrast(st, rasters, tr)
        ref = [v for v in (contrast(pc, rasters, tr) for pc in pieces(off, st.length)) if np.isfinite(v)]
        bar = float(np.percentile(ref, 25)) if len(ref) >= 8 else np.nan
        if np.isfinite(c) and np.isfinite(bar) and c >= bar:
            continue
        m = gpd.GeoSeries([st.interpolate(0.5, normalized=True)], crs=eco.crs).to_crs(4326).iloc[0]
        near = eco[eco.intersects(st.buffer(3000))]
        rows.append(dict(km=round(st.length / 1000), lat=round(m.y, 2), lon=round(m.x, 2), contrast=round(c, 2),
                         bar=round(bar, 2), types="; ".join(sorted(near.type_name.unique()))))
        log.warning("province cut: %.0f km near %.2f N, %.2f E (contrast %.2f, bar %.2f): %s", st.length / 1000, m.y,
                    m.x, c, bar, rows[-1]["types"])
    return pd.DataFrame(rows, columns=["km", "lat", "lon", "contrast", "bar", "types"])


def provinces(cfg, lab, pg, units) -> None:
    """Province gate, run by the provinces stage: a province edge along a border without a natural break would carry
    the border into every province-keyed rule. Seeds are placed by hand, so this is where a border slips in."""
    from rasterio.features import shapes

    from .config import Paths
    from .grid import Grid

    paths, bc = Paths(cfg), cfg.get("bordercheck", {})
    tol, min_km = bc.get("tol_km", 10.0), bc.get("min_km", 50.0)
    grid = Grid.load(paths.grid_json)
    rasters, tr = _rasters(paths, grid)
    polys = [(shapely.geometry.shape(g), int(v)) for g, v in shapes(lab, mask=lab > 0, transform=pg.transform)]
    regions = gpd.GeoDataFrame({"code": [units[v - 1]["code"] for _, v in polys]}, geometry=[g for g, _ in polys],
                               crs=grid.crs).dissolve("code").reset_index()
    rows = judge("provinces", regions, _countries(paths, grid.crs), rasters, tr, tol, min_km)
    pd.DataFrame(rows, columns=COLS).to_csv(paths.qa / "bordercheck_provinces.csv", index=False)
    bad = [r for r in rows if r["verdict"] == "ARTEFACT"]
    if bad:
        raise SystemExit("bordercheck: province edges follow a political border without a natural break near "
                         + "; ".join(f"{r['lat']} N {r['lon']} E ({r['km']} km)" for r in bad)
                         + ". Move the seeds so the edge follows the landscape.")


# Layers compiled country by country (geology, soils) or derived from them (the dune mask reads mapped aeolian sand);
# terrain and climate come from uniform global products and are not tested.
COMPILED = ("litho_", "age", "clay", "sand", "ph", "soc", "dune_frac")


def border_pairs(line, off=10000.0, step=5000.0):
    """Point pairs ``off`` m either side of ``line``, every ``step`` m."""
    return side_points(line, off, step)


def seam(arr, transform, line, shifts) -> tuple[float, list[float]]:
    """Mean absolute difference of ``arr`` across ``line`` and across the line moved by each of ``shifts``."""
    px = [pixels(*border_pairs(ln), transform, arr.shape) for ln in [line] + [affinity.translate(line, dx, dy) for dx, dy in shifts]]
    d = [mean_diff(arr, p) for p in px]
    return d[0], d[1:]


def mean_diff(arr, px) -> float:
    r1, c1, r2, c2 = px
    if not len(r1):
        return np.nan
    v = np.abs(arr[r1, c1].astype("float64") - arr[r2, c2])
    return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan


def input_seams(cfg, names, runs=20, seed=1, level=20) -> tuple[list[dict], dict]:
    """A compiled layer (geology, soils) that changes across one of the area's international borders more than across
    the same border shifted 30-80 km (above every shift), and by a larger factor than elevation, rainfall and
    temperature change there (a border on a mountain front or a river plain changes all of them), has a seam between
    national source maps there. Rows for every layer and border, and each border's line (coarse grid CRS)."""
    import logging

    from .config import Paths
    from .grid import Grid, read

    log = logging.getLogger(__name__)
    paths = Paths(cfg)
    cg = Grid.load(paths.grid_json).coarsen(cfg["grid"]["coarse_factor"])
    area = shapely.box(*cg.transform * (0, cg.height), *cg.transform * (cg.width, 0))
    cty = _countries(paths, cg.crs)
    cty = cty[cty.intersects(area)]
    rng = np.random.default_rng(seed)
    rows, lines = [], {}
    cache: dict[str, np.ndarray] = {}

    def layer(name):
        if name not in cache:
            cache[name] = read(paths.coarse(name))
        return cache[name]

    for i, x in enumerate(cty.ADM0_A3):
        for y in list(cty.ADM0_A3)[i + 1:]:
            A, B = cty[cty.ADM0_A3 == x].union_all(), cty[cty.ADM0_A3 == y].union_all()
            line = shapely.line_merge(A.boundary.intersection(B.buffer(500)).intersection(area))
            if line.length < 200000:
                continue
            lines[f"{x}-{y}"] = line
            shifts = [(d * np.cos(t), d * np.sin(t)) for d, t in zip(rng.uniform(30000, 80000, runs), rng.uniform(0, 2 * np.pi, runs))]
            px = [pixels(*border_pairs(ln), cg.transform, cg.shape)
                  for ln in [line] + [affinity.translate(line, dx, dy) for dx, dy in shifts]]
            ratio = {}
            for name in ("elev", "bio12", "bio1", *names):
                if name in ratio or not paths.coarse(name).exists() or (name not in ("elev", "bio12", "bio1")
                                                                         and not name.startswith(COMPILED)):
                    continue
                obs, *null = [mean_diff(layer(name), p) for p in px]
                mean, mx = float(np.nanmean(null)), float(np.nanmax(null))
                ratio[name] = obs / mean if mean > 0 else (np.inf if obs > 0 else 1.0)
                if name in ("elev", "bio12", "bio1"):
                    continue
                land = max(1.25, *(ratio.get(k, 1.0) for k in ("elev", "bio12", "bio1")))
                bad = bool(obs > mx and ratio[name] > land)
                rows.append(dict(border=f"{x}-{y}", km=round(line.length / 1000), layer=name, across=round(obs, 4),
                                 shifted_mean=round(mean, 4), shifted_max=round(mx, 4), ratio=round(ratio[name], 2),
                                 landscape_ratio=round(land, 2), seam=bad))
                log.log(level if bad else logging.DEBUG,
                        "inputs %s across %s-%s: %.3f (shifted %.3f, max %.3f; x%.2f against x%.2f for the landscape)%s",
                        name, x, y, obs, mean, mx, ratio[name], land, ": SEAM" if bad else "")
    return rows, lines


def inputs(cfg, names) -> None:
    """Input gate, run by the provinces stage after the features stage has blended the seams it found
    (``features.blend_seams``): no compiled layer may still change at a border more than the landscape does."""
    import logging

    from .config import Paths

    rows, _ = input_seams(cfg, names, level=logging.ERROR)
    pd.DataFrame(rows).to_csv(Paths(cfg).qa / "bordercheck_inputs.csv", index=False)
    bad = sorted({f"{r['layer']} ({r['border']})" for r in rows if r["seam"]})
    if bad:
        raise SystemExit("bordercheck: input layers still change at international borders after blending: "
                         + ", ".join(bad) + ". Widen seams.band_km or look at the source maps there.")


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
        a, b = split_near(LineString([(0, 0), (100000, 0)]), LineString([(0, 500), (40000, 500)]), 1000.0)
        assert abs(a.length - 40000) <= 1000 and abs(b.length - 60000) <= 1000, (a.length, b.length)
        line = LineString([(50000, 0), (50000, 100000)])  # a seam exactly on a "border" at x = 50 km
        obs, null = seam(step, tr, line, [(20000, 0), (-20000, 0), (0, 30000)])
        assert obs == 1000.0 and max(null[:2]) == 0.0
        print("ok")
