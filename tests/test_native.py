import numpy as np
import pandas as pd

from ecoregions import native


def test_statuses(monkeypatch):
    names = pd.DataFrame({"name": ["Pistacia vera", "Ailanthus altissima", "Old name", "Rare plant"],
                          "accepted": ["Pistacia vera", "Ailanthus altissima", "Pistacia vera", "Rare plant"]})
    dist = pd.DataFrame({"accepted": ["Pistacia vera", "Ailanthus altissima", "Rare plant"],
                         "area_code_l3": ["IRN", "IRN", "IRN"], "status": ["native", "introduced", "native"]})
    monkeypatch.setattr(native, "status_table", lambda paths: (names, dist))
    # five records: native in Iran; introduced in Iran; synonym in Iran; just across the border in Turkmenistan;
    # unknown name
    monkeypatch.setattr(native, "areas_of", lambda paths, lon, lat: np.array(["IRN", "IRN", "IRN", "TKM", "IRN"]))
    monkeypatch.setattr(native, "nearby_areas", lambda paths, lon, lat, km: pd.Series({0: ["IRN"], 1: ["TKM", "IRN"]}))
    sp = pd.Series(["Pistacia vera", "Ailanthus altissima", "Old name", "Rare plant", "Nomen nudum"])
    st, _ = native.statuses(None, sp, np.zeros(5), np.zeros(5))
    assert list(st) == ["native", "introduced", "native", "native nearby", "unmatched"]
