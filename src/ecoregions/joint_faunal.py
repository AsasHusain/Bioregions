"""Faunal Ecoregions clustered once over several areas, so no region ends at the seam between two areas.

Each area's ``fauna`` stage gives species presence on its own 10 km cells. Here the cells whose centres lie inside
each area's mapped countries (no overlap strip, so no cell is counted twice) are pooled into ``bin_km`` equal-area
bins (EPSG:6933; presence = any cell of the bin), and the bins of all areas are clustered once, as in ``faunal``:
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


def choose_k(scan: pd.DataFrame, step=0.01, flat=5, k_min=8) -> tuple[int, int]:
    """Start of the first plateau after ``k_min``, and the key's groups (largest step above 5 regions, below k)."""
    e = scan.set_index("k").explained
    d = e.diff()
    k = next((k for k in e.index if k >= k_min and d.get(k, 0) >= step
              and all(d.get(j, 0) < step for j in range(k + 1, k + 1 + flat) if j in d)), int(e.index.max()))
    g = int(d[(d.index > 5) & (d.index < k)].idxmax())
    return int(k), g


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


def run(areas: list[str], bin_km=25.0) -> dict:
    from .boundaries import load_boundaries, mapped_area

    cfgs = {r: load_config(region=r) for r in areas}
    fc = cfgs[areas[0]]["faunal"]
    countries, disputed = load_boundaries(Paths(cfgs[areas[0]]))
    data, names = {}, {}
    for r, cfg in cfgs.items():
        src = Paths(cfg).interim / "fauna"
        grid = Grid.load(src / "grid.json")
        cells = pd.read_csv(src / "cells.csv")
        spc = pd.read_csv(src / "species.csv")
        X = sp.load_npz(src / f"presence_{fc['presence']}.npz").tocsr()
        X.data[:] = 1
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
    S = len(names)
    names_list = sorted(names, key=names.get)
    for d in data.values():
        d["X"] = d["X"][:, :S].tocsr()

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
    del d
    i, j = np.triu_indices(B.shape[0], 1)
    Z = linkage(dv, method="average")
    scan = pd.DataFrame([dict(k=k, explained=round(explained_turnover(dv, i, j, fcluster(Z, k, criterion="maxclust")), 4))
                         for k in range(fc["k_range"][0], max(fc["k_range"][1], 40) + 1)])
    del dv, i, j
    K, G = choose_k(scan)
    log.info("joint faunal: turnover explained\n%s\n-> %d regions in %d groups", scan.to_string(index=False), K, G)
    lab_b = fcluster(Z, K, criterion="maxclust")
    grp_b = fcluster(Z, G, criterion="maxclust")
    parent = pd.crosstab(lab_b, grp_b).idxmax(axis=1)  # UPGMA nests them exactly
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
        print("ok")
