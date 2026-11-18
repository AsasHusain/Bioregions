"""Command-line entry point: ``python -m ecoregions <stage> [--region pak-afg] [--config config.yaml]``.

Stages are thin wrappers around module-level ``run(cfg)`` functions so the
Makefile (and notebooks) can call any step in isolation.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import time

from .config import load_config, setup_logging

STAGES = {
    "download": "ecoregions.download",
    "preflight": "ecoregions.preflight",
    "grid": "ecoregions.boundaries",
    "dem": "ecoregions.dem",
    "terrain": "ecoregions.terrain",
    "hydro": "ecoregions.hydro",
    "covariates": "ecoregions.covariates",
    "features": "ecoregions.features",
    "provinces": "ecoregions.provinces",
    "segment": "ecoregions.segment",
    "snap": "ecoregions.snap",
    "attributes": "ecoregions.attributes",
    "communities": "ecoregions.communities",
    "vegcheck": "ecoregions.vegcheck",
    "fauna": "ecoregions.fauna",
    "faunal": "ecoregions.faunal",
    "global_fauna": "ecoregions.global_fauna",
    "biomes": "ecoregions.biomes",
    "assemble": "ecoregions.assemble",
    "evaluate": "ecoregions.evaluate",
    "scorecard": "ecoregions.scorecard",
    "merges": "ecoregions.merges",
    "confirm": "ecoregions.confirm",
    "plots": "ecoregions.plots",
    "audit": "ecoregions.audit",
    "names": "ecoregions.names",
    "validate": "ecoregions.validate",
    "crosscheck": "ecoregions.crosscheck",
    "package": "ecoregions.package",
    "join": "ecoregions.join",
    "bordercheck": "ecoregions.bordercheck",
    "webmap": "ecoregions.webmap",
    "staticmap": "ecoregions.staticmap",
}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ecoregions")
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("--config", default=None)
    p.add_argument("--region", default=None, help="mapped region (regions/<region>.yaml); default pak-afg")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    setup_logging(logging.DEBUG if a.verbose else logging.INFO)
    cfg = load_config(a.config, a.region)
    t0 = time.time()
    status = "failed"
    try:
        importlib.import_module(STAGES[a.stage]).run(cfg)
        status = "ok"
    finally:
        record_run(cfg, a.stage, t0, status)
    logging.getLogger("ecoregions").info("stage %s finished in %.1f min", a.stage, (time.time() - t0) / 60)


def record_run(cfg, stage: str, t0: float, status: str) -> None:
    """One line per stage run in <interim>/runs.jsonl: code commit, uncommitted source files and the hashes of the
    settings and rule files, so every output can be traced to the code and settings that made it."""
    import hashlib
    import json
    import subprocess
    from pathlib import Path

    from .config import Paths

    root = Path(cfg["_root"])
    git = lambda *c: subprocess.run(["git", *c], cwd=root, capture_output=True, text=True).stdout.rstrip("\n")
    files = [root / "config.yaml", root / "regions" / f"{cfg['_region']}.yaml", root / "gazetteer.yaml",
             *sorted(root.glob("communities*.yaml"))]
    rec = {"stage": stage, "region": cfg["_region"], "status": status,
           "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), "minutes": round((time.time() - t0) / 60, 2),
           "commit": git("rev-parse", "HEAD"),
           "uncommitted": [l[3:] for l in git("status", "--porcelain", "--", "src", "*.yaml", "regions").splitlines()],
           "settings": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in files if f.exists()}}
    out = Paths(cfg).interim / "runs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
