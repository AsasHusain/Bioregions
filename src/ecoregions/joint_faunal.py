"""Faunal Ecoregions clustered once over several areas, so no region ends at the seam between two areas.

Each area's ``fauna`` stage gives species presence on its own 10 km cells. Here the cells whose centres lie inside
each area's mapped countries (no overlap strip, so no cell is counted twice) are pooled into ``bin_km`` equal-area
bins (EPSG:6933; presence = any cell of the bin), and the bins of all areas are clustered once, as in ``faunal``.
Species whose range map stops at a border on level ground for 100 km or more are left out (``clipped_species``):
such maps were drawn country by country and would draw the border into the regions. The rest are clustered with
Simpson's beta diversity and average linkage (UPGMA). Bins keep the matrix within memory (a 10 km clustering of
three areas would need some 12 GB per matrix). The number of regions is read from the turnover curve as before
(the start of a plateau: a step of at least ``step`` into k, then at most ``step`` per region for ``flat`` more),
the key's groups from the largest step above 5 regions.

Every 10 km cell of every area, overlap strip included, then takes the region, among those of its own and the eight
neighbouring bins, whose species frequencies its species list matches best, so regions keep 10 km detail and
neighbouring areas see the same regions across the seam. Written into each area's ``interim/faunal`` as the
``faunal`` stage would (regions_k<k>.tif, region_species_k<k>.csv, region_members_k<k>.csv, k_scan.csv,
n_species.txt); the areas' ``assemble`` stage then draws them as usual, with region numbers shared by all areas.

    PYTHONPATH=src python -m ecoregions.joint_faunal pak-afg-irn central-asia caucasus-anatolia
"""
from __future__ import annotations

import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from pyproj import Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .config import Paths, load_config
from .faunal import beta_sim, characteristic_species, clean_fragments, explained_turnover, region_members
from .grid import Grid, write

log = logging.getLogger(__name__)
CRS = "EPSG:6933"


def plateau(scan: pd.DataFrame, step=0.01, flat=5, k_min=8) -> int:
    """Start of the first plateau from ``k_min``: a step of at least ``step`` into k, then less per region for ``flat``."""
    e = scan.set_index("k").explained
    d = e.diff()
    return int(next((k for k in e.index if k >= k_min and d.get(k, 0) >= step
                     and all(d.get(j, 0) < step for j in range(k + 1, k + 1 + flat) if j in d)), int(e.index.max())))


def choose_k(scan: pd.DataFrame, step=0.01, flat=5, k_min=8) -> tuple[int, int]:
    """Start of the first plateau after ``k_min``, and the key's groups (largest step above 5 regions, below k)."""
    k = plateau(scan, step, flat, k_min)
    d = scan.set_index("k").explained.diff()
    g = int(d[(d.index > 5) & (d.index < k)].idxmax())
    return k, g


def nested(D: np.ndarray, grp: np.ndarray, step=0.01, flat=5, k_max=30) -> tuple[np.ndarray, dict]:
    """Regions within each group: the group's own UPGMA tree (the global tree's subtree) cut where its own turnover
    curve first levels off (``plateau`` from 2 regions), so each part of a large joint area gets the detail its own
    turnover supports instead of one cut for all (which left Caucasus & Anatolia one region). Region labels 1..K,
    numbered group by group, and each group's number of regions."""
    lab = np.zeros(len(grp), int)
    ks, nxt = {}, 0
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        sub = np.ones(len(idx), int)
        if len(idx) > 2:
            dv = squareform(D[np.ix_(idx, idx)], checks=False)
            Z = linkage(dv, method="average")
            i, j = np.triu_indices(len(idx), 1)
            sc = pd.DataFrame([dict(k=k, explained=explained_turnover(dv, i, j, fcluster(Z, k, criterion="maxclust")))
                               for k in range(1, min(k_max, len(idx)) + 1)])
            sub = fcluster(Z, plateau(sc, step, flat, k_min=2), criterion="maxclust")
        lab[idx] = nxt + sub
        ks[int(g)] = int(sub.max())
        nxt += int(sub.max())
    return lab, ks


def assign(cells_sp: sp.csr_matrix, cand: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """For each cell, the candidate region (columns of ``cand``, 0 = none) with the highest mean frequency of the
    cell's species; a cell without species takes its first candidate."""
    score = np.full(cand.shape, -1.0)
    n = np.maximum(np.asarray(cells_sp.sum(axis=1)).ravel(), 1)
    for c in range(cand.shape[1]):
        r = cand[:, c]
        ok = r > 0
        if ok.any():
            # mean of freq[r, s] over the cell's species s
            score[ok, c] = np.asarray(cells_sp[ok].multiply(freq[r[ok] - 1]).sum(axis=1)).ravel() / n[ok]
    best = cand[np.arange(len(cand)), score.argmax(axis=1)]
    return np.where(best > 0, best, cand[:, 0])


def longest_runs(d: np.ndarray) -> np.ndarray:
    """Per column, the longest run of consecutive rows equal to +1 or to -1 (one-sided presence along a border)."""
    best = np.zeros(d.shape[1], int)
    for sgn in (1, -1):
        run = np.zeros(d.shape[1], int)
        for row in d == sgn:
            run = np.where(row, run + 1, 0)
            best = np.maximum(best, run)
    return best


def clipped_species(grid, cells, X, names, countries, dem_path, off_m=12000.0, step_m=10000.0, min_km=100.0,
                    max_dz=300.0) -> set:
    """Species whose range map stops at an international border: present on one side and absent on the other at
    pairs of cells ``off_m`` either side of the border, every ``step_m``, for at least ``min_km`` in a row, where
    the two cells differ in height by less than ``max_dz`` (so neither height nor a crest explains it). IUCN and
    BirdLife ranges drawn country by country end at borders (the Cape and woolly hares at the Iran-Pakistan line),
    and such a map cannot say where the species really stops."""
    import rasterio
    from .bordercheck import international_borders, side_points

    idx = np.full(grid.shape, -1)
    idx[cells.row, cells.col] = np.arange(len(cells))
    x0, x1, y0, y1 = grid.x0, grid.x0 + grid.width * grid.res, grid.y0 - grid.height * grid.res, grid.y0
    cg = countries.to_crs(grid.crs).cx[x0:x1, y0:y1]
    lines = shapely.line_merge(international_borders(cg).intersection(shapely.box(x0, y0, x1, y1)))
    tl = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    out = set()
    with rasterio.open(dem_path) as dem:
        for ln in shapely.get_parts(lines):
            if ln.length < min_km * 1000:
                continue
            a, b = side_points(ln, off_m, step_m)
            cell, z = [], []
            for p in (a, b):
                r, c = ((grid.y0 - p[:, 1]) / grid.res).astype(int), ((p[:, 0] - grid.x0) / grid.res).astype(int)
                ok = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
                cell.append(np.where(ok, idx[np.clip(r, 0, grid.height - 1), np.clip(c, 0, grid.width - 1)], -1))
                z.append(np.array([v[0] for v in dem.sample(list(zip(*tl.transform(p[:, 0], p[:, 1]))))], float))
            good = (cell[0] >= 0) & (cell[1] >= 0) & (np.abs(z[0] - z[1]) < max_dz)
            if good.sum() * step_m < min_km * 1000:
                continue
            d = (X[np.where(good, cell[0], 0)].toarray().astype(np.int8) - X[np.where(good, cell[1], 0)].toarray().astype(np.int8))
            d[~good] = 0
            out |= {names[j] for j in np.flatnonzero(longest_runs(d) * step_m >= min_km * 1000)}
    return out


def run(areas: list[str], bin_km=25.0) -> dict:
    from .boundaries import load_boundaries, mapped_area

    cfgs = {r: load_config(region=r) for r in areas}
    fc = cfgs[areas[0]]["faunal"]
    countries, disputed = load_boundaries(Paths(cfgs[areas[0]]))
    data, names, clipped = {}, {}, {}
    dem = f"/vsizip/{(Paths(cfgs[areas[0]]).raw / 'gmted2010' / 'mn30_grd.zip').as_posix()}/mn30_grd"
    for r, cfg in cfgs.items():
        src = Paths(cfg).interim / "fauna"
        grid = Grid.load(src / "grid.json")
        cells = pd.read_csv(src / "cells.csv")
        spc = pd.read_csv(src / "species.csv")
        X = sp.load_npz(src / f"presence_{fc['presence']}.npz").tocsr()
        X.data[:] = 1
        for n in clipped_species(grid, cells, X, spc.name.tolist(), countries, dem):
            clipped.setdefault(n, []).append(r)
        for n in spc.name:
            names.setdefault(n, len(names))
        cols = spc.name.map(names).to_numpy()
        Xg = sp.csr_matrix((X.data, cols[X.indices], X.indptr), shape=(X.shape[0], 100000))  # species ids shared by all areas; trimmed below
        area = mapped_area(cfg, Paths(cfg), countries, disputed).to_crs(grid.crs).union_all()
        inside = shapely.contains_xy(area, cells.x.to_numpy(), cells.y.to_numpy())
        x, y = Transformer.from_crs(grid.crs, CRS, always_xy=True).transform(cells.x.to_numpy(), cells.y.to_numpy())
        data[r] = dict(cfg=cfg, grid=grid, cells=cells, X=Xg, inside=inside,
                       bx=np.floor(np.asarray(x) / (bin_km * 1000)).astype(int),
                       by=np.floor(np.asarray(y) / (bin_km * 1000)).astype(int))
    keep = np.array([n not in clipped for n in sorted(names, key=names.get)])
    names_list = [n for n, k in zip(sorted(names, key=names.get), keep) if k]
    S = len(names_list)
    for d in data.values():
        d["X"] = d["X"][:, :len(keep)][:, np.flatnonzero(keep)].tocsr()
    log.info("joint faunal: %d of %d species left out, their range maps stopping at a border on level ground "
             "(qa/clipped_species.csv in each area)", len(clipped), len(keep))
    cl = pd.DataFrame(sorted((n, ", ".join(a)) for n, a in clipped.items()), columns=["species", "areas"])
    for cfg in cfgs.values():
        (Paths(cfg).qa).mkdir(parents=True, exist_ok=True)
        cl.to_csv(Paths(cfg).qa / "clipped_species.csv", index=False)

    # bins from the cells inside the areas
    key = np.concatenate([d["bx"][d["inside"]] * 100000 + d["by"][d["inside"]] for d in data.values()])
    Xin = sp.vstack([d["X"][d["inside"]] for d in data.values()]).tocsr()
    ub, inv = np.unique(key, return_inverse=True)
    B = (sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(ub), len(inv))) @ Xin)
    B.data[:] = 1
    keep = np.asarray(B.sum(axis=1)).ravel() > 0
    ub, B = ub[keep], B[keep]
    log.info("joint faunal: %d bins of %.0f km x %d species from %s", B.shape[0], bin_km, S, ", ".join(areas))

    d = beta_sim(B)
    dv = squareform(d, checks=False)
    i, j = np.triu_indices(B.shape[0], 1)
    Z = linkage(dv, method="average")
    scan = pd.DataFrame([dict(k=k, explained=round(explained_turnover(dv, i, j, fcluster(Z, k, criterion="maxclust")), 4))
                         for k in range(fc["k_range"][0], max(fc["k_range"][1], 40) + 1)])
    del dv, i, j
    _, G = choose_k(scan)
    grp_b = fcluster(Z, G, criterion="maxclust")
    lab_b, per = nested(d, grp_b)
    del d
    K = int(lab_b.max())
    log.info("joint faunal: turnover explained\n%s\n-> %d groups; regions within each group (own plateau): %s -> %d regions",
             scan.to_string(index=False), G, per, K)
    parent = pd.crosstab(lab_b, grp_b).idxmax(axis=1)  # nested by construction
    freq = np.vstack([np.asarray(B[lab_b == k].mean(axis=0)).ravel() for k in range(1, K + 1)])
    binlab = dict(zip(ub, lab_b))

    pooled_X, pooled_lab = [], []
    for r, dd in data.items():
        cand = np.zeros((len(dd["cells"]), 9), int)
        c = 0
        for ox in (0, -1, 1):
            for oy in (0, -1, 1):
                cand[:, c] = [binlab.get(k, 0) for k in (dd["bx"] + ox) * 100000 + dd["by"] + oy]
                c += 1
        has = cand.max(axis=1) > 0
        cand[:, 0] = np.where(cand[:, 0] > 0, cand[:, 0], cand.max(axis=1))  # a cell's fallback: its own bin, else a neighbour
        lab = np.where(has, assign(dd["X"], cand, freq), 0)
        g = dd["grid"]
        study = dd["cells"].study_share.to_numpy() >= fc["min_study_share"]
        lab2d = np.zeros(len(dd["cells"]), "uint8")
        lab2d[study] = lab[study]
        lab2d = clean_fragments(lab2d.reshape(g.shape), fc["min_fragment_cells"])
        grp2d = np.where(lab2d > 0, parent.reindex(np.arange(K + 1)).fillna(0).astype(int).to_numpy()[lab2d], 0).astype("uint8")
        out = Paths(dd["cfg"]).interim / "faunal"
        out.mkdir(parents=True, exist_ok=True)
        write(out / f"regions_k{K}.tif", lab2d, g, nodata=0)
        write(out / f"regions_k{G}.tif", grp2d, g, nodata=0)
        scan.to_csv(out / "k_scan.csv", index=False)
        dd["lab"] = lab2d.ravel()
        pooled_X.append(dd["X"][dd["inside"]])
        pooled_lab.append(dd["lab"][dd["inside"]])
    PX = sp.vstack(pooled_X).tocsc()
    PL = np.concatenate(pooled_lab)
    sel = PL > 0
    PX, PL = PX[sel], PL[sel]
    used = np.asarray(PX.sum(axis=0)).ravel() > 0
    ch = characteristic_species(PX, PL, names_list, fc["top_species"]).assign(k=K)
    cls = pd.concat([pd.read_csv(Paths(c).interim / "fauna" / "species.csv") for c in cfgs.values()]).drop_duplicates("name").set_index("name").cls
    mem = region_members(PX, PL, names_list, cls, fc, Paths(cfgs[areas[0]]))
    for dd in data.values():
        out = Paths(dd["cfg"]).interim / "faunal"
        ch[["k", "region", "species", "fidelity", "frequency", "score"]].to_csv(out / f"region_species_k{K}.csv", index=False)
        mem.to_csv(out / f"region_members_k{K}.csv", index=False)
        (out / "n_species.txt").write_text(str(int(used.sum())))
    log.info("joint faunal: %d regions (%d groups) written for %s; set assemble.faunal_k = %d, faunal_group_k = %d "
             "and faunal.k_maps = [%d, %d] in each area", K, G, ", ".join(areas), K, G, G, K)
    return dict(k=K, group_k=G, scan=scan)


if __name__ == "__main__":
    from .config import setup_logging

    setup_logging()
    if len(sys.argv) > 1:
        run(sys.argv[1:])
    else:  # self-check: the plateau rule and the assignment
        # steps: +.10 +.10 +.05 (k=6) +.12 (k=7) +.05 +.05 +.02 (k=10), then +.001 per region
        s = pd.DataFrame(dict(k=range(3, 20), explained=[.10, .20, .30, .35, .47, .52, .57, .59, .591, .592, .593, .594,
                                                         .595, .596, .597, .598, .599]))
        assert choose_k(s) == (10, 7), choose_k(s)
        X = sp.csr_matrix(np.array([[1, 1, 0], [0, 0, 1]]))
        freq = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert list(assign(X, np.array([[2, 1], [1, 2]]), freq)) == [1, 2]
        d = np.array([[1, -1, 0], [1, -1, 1], [0, -1, -1], [1, -1, 1], [1, 0, 1], [1, 0, 1]])
        assert list(longest_runs(d)) == [3, 4, 3], longest_runs(d)
        # two groups; the first holds two tight clusters, which become its two regions, numbered before the second's
        pts = np.r_[np.zeros(6), np.full(6, 1.0), np.full(6, 10.0)] + np.tile([0, .01, .02], 6)
        D = np.abs(pts[:, None] - pts[None, :])
        lab, ks = nested(D, np.r_[np.ones(12, int), np.full(6, 2)])
        assert ks[1] == 2 and len(set(lab[:6])) == 1 and lab[0] != lab[6] and lab[12:].min() == 3, (ks, lab)
        print("ok")
