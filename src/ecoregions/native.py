"""Native status of plant records in the botanical area where each was made.

Source: the World Checklist of Vascular Plants (WCVP, frozen release in ``raw/wcvp``) and the level-3 botanical areas
of the World Geographical Scheme for Recording Plant Distributions (WGSRPD, ``raw/wgsrpd/level3.geojson``).

Each record's name resolves to accepted species in WCVP (synonyms to their accepted name, infraspecific taxa to their
species), and takes one status in the level-3 area containing it:

- ``native``: the species is listed there as native (listings marked extinct count as native: the target is the
  native biota before modern conversion);
- ``native nearby``: not native in that area, but native in another level-3 area within ``native.nearby_km`` (100 km)
  of the record. Level-3 areas mostly follow national borders, and a range edge does not stop at one;
- ``introduced``: listed there only as introduced, and native in no area nearby;
- ``unlisted``: in WCVP, but not listed in that area or nearby. A missing listing is not evidence of introduction;
- ``unmatched``: the name is not in WCVP, or resolves to more than one accepted species.

Only the statuses in ``native.drop`` (default: introduced) are removed, so a gap in the checklist never removes a
record. Level-3 areas mostly follow national borders, so the counts per area are written to
``<interim>/native_status.csv`` to show whether exclusions change at a border.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import Paths

log = logging.getLogger(__name__)
STATUSES = ("native", "native nearby", "introduced", "unlisted", "unmatched")


def _binomial(genus: pd.Series, species: pd.Series) -> pd.Series:
    return genus.fillna("").str.strip() + " " + species.fillna("").str.strip()


def status_table(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(name -> accepted species, (accepted species, area) -> native or introduced), cached beside the release."""
    root = paths.raw / "wcvp"
    names_f, dist_f = root / "names_resolved.csv.gz", root / "status_by_area.csv.gz"
    if names_f.exists() and dist_f.exists():
        return pd.read_csv(names_f), pd.read_csv(dist_f)
    n = pd.read_csv(root / "wcvp_names.csv", sep="|", dtype=str, quoting=3,
                    usecols=["plant_name_id", "taxon_rank", "taxon_status", "genus", "species",
                             "accepted_plant_name_id", "parent_plant_name_id"])
    n = n[n.species.notna() & (n.species != "")]
    n["name"] = _binomial(n.genus, n.species)
    acc = n[n.taxon_status == "Accepted"].set_index("plant_name_id")
    # accepted species of every accepted taxon: itself, or for an infraspecific taxon the species it belongs to
    acc_sp = acc["name"]
    # every name (accepted or not) -> its accepted taxon -> that taxon's species
    target = n.accepted_plant_name_id.where(n.taxon_status != "Accepted", n.plant_name_id)
    n["accepted"] = target.map(acc_sp)
    n = n.dropna(subset=["accepted"])
    # an accepted name resolves to itself even if it is also a synonym elsewhere (homonyms)
    own = n[n.taxon_status == "Accepted"][["name", "accepted"]].drop_duplicates()
    other = n[(n.taxon_status != "Accepted") & ~n.name.isin(own.name)][["name", "accepted"]].drop_duplicates()
    names = pd.concat([own, other])
    k = names.groupby("name").accepted.nunique()
    names = names[names.name.isin(k[k == 1].index)].drop_duplicates("name")  # ambiguous names stay unmatched

    d = pd.read_csv(root / "wcvp_distribution.csv", sep="|", dtype=str, quoting=3,
                    usecols=["plant_name_id", "area_code_l3", "introduced", "location_doubtful"])
    d = d[d.location_doubtful != "1"]
    d["accepted"] = d.plant_name_id.map(acc_sp)
    d = d.dropna(subset=["accepted"])
    d["native"] = d.introduced == "0"
    dist = d.groupby(["accepted", "area_code_l3"]).native.any().reset_index()
    dist["status"] = np.where(dist.native, "native", "introduced")
    dist = dist[["accepted", "area_code_l3", "status"]]
    names.to_csv(names_f, index=False)
    dist.to_csv(dist_f, index=False)
    log.info("native: WCVP resolved %d names to %d accepted species; %d species-area listings", len(names),
             names.accepted.nunique(), len(dist))
    return names, dist


def areas_of(paths: Paths, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Level-3 area code of each point; points just off the outlines (coasts) take the nearest area within 0.1 deg."""
    a = gpd.read_file(paths.raw / "wgsrpd" / "level3.geojson")[["LEVEL3_COD", "geometry"]]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326)
    a = a.to_crs(4326)
    j = gpd.sjoin(pts, a, predicate="within", how="left")
    j = j[~j.index.duplicated()]
    code = j.LEVEL3_COD.reindex(pts.index)
    miss = code.isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss], a, max_distance=0.1, how="left")
        code[miss] = near[~near.index.duplicated()].LEVEL3_COD.reindex(pts[miss].index)
    return code.to_numpy()


def nearby_areas(paths: Paths, lon: np.ndarray, lat: np.ndarray, km: float) -> pd.Series:
    """For each point (by position), the level-3 areas within ``km`` of it."""
    a = gpd.read_file(paths.raw / "wgsrpd" / "level3.geojson")[["LEVEL3_COD", "geometry"]].to_crs("EPSG:6933")
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=4326).to_crs("EPSG:6933")
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=pts.buffer(km * 1000)), a, predicate="intersects")
    return j.groupby(level=0).LEVEL3_COD.agg(list)


def statuses(paths: Paths, species: pd.Series, lon: np.ndarray, lat: np.ndarray, nearby_km: float = 100.0):
    names, dist = status_table(paths)
    acc = species.map(names.set_index("name").accepted).to_numpy()
    area = areas_of(paths, lon, lat)
    st = pd.Series(dist.status.to_numpy(), index=pd.MultiIndex.from_frame(dist[["accepted", "area_code_l3"]]))
    out = st.reindex(pd.MultiIndex.from_arrays([acc, area])).to_numpy(dtype=object)
    out = np.where(pd.isna(out), "unlisted", out)
    # not native here: native in a neighbouring area within nearby_km?
    check = np.flatnonzero(np.isin(out, ["introduced", "unlisted"]) & ~pd.isna(acc))
    if check.size:
        near = nearby_areas(paths, lon[check], lat[check], nearby_km)
        native = set(map(tuple, dist.loc[dist.status == "native", ["accepted", "area_code_l3"]].to_numpy()))
        for k, i in enumerate(check):
            if any((acc[i], c) in native for c in near.get(k, [])):
                out[i] = "native nearby"
    return np.where(pd.isna(acc), "unmatched", out), area


def screen(cfg, recs: pd.DataFrame) -> pd.DataFrame:
    """Plant records (``species``, ``decimalLongitude``, ``decimalLatitude``) without those whose status in their
    botanical area is in ``native.drop``; writes the counts per area and status."""
    paths = Paths(cfg)
    lon = recs.decimalLongitude.astype(float).to_numpy()
    lat = recs.decimalLatitude.astype(float).to_numpy()
    st, area = statuses(paths, recs.species.reset_index(drop=True), lon, lat,
                        cfg.get("native", {}).get("nearby_km", 100.0))
    tab = pd.crosstab(pd.Series(area, name="area_code_l3").fillna("none"), pd.Series(st, name="status"))
    tab = tab.reindex(columns=list(STATUSES), fill_value=0)
    paths.interim.mkdir(parents=True, exist_ok=True)
    tab.to_csv(paths.interim / "native_status.csv")
    drop = set(cfg.get("native", {}).get("drop", ["introduced"]))
    keep = ~np.isin(st, list(drop))
    log.info("native: %s of %d plant records; dropped %d (%s)",
             ", ".join(f"{s} {int((st == s).sum())}" for s in STATUSES), len(st), int((~keep).sum()), "/".join(sorted(drop)))
    return recs[keep]
