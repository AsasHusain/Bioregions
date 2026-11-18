from ecoregions.landsat import pick


def f(date, cloud, tile="1/1"):
    p, r = tile.split("/")
    return {"properties": {"datetime": date, "eo:cloud_cover": cloud, "landsat:wrs_path": p, "landsat:wrs_row": r}}


def test_pick_keeps_one_season_and_spreads_years():
    feats = [f("1985-07-01", 1), f("1985-08-01", 2), f("1986-06-15", 3), f("1987-01-10", 0), f("1988-11-01", 0.5)]
    got = [x["properties"]["datetime"] for x in pick(feats, 2)["11"]]
    assert got == ["1985-07-01", "1986-06-15"]  # winter and November scenes dropped; second year before same year


def test_pick_widens_season_for_tiles_without_two_scenes():
    feats = [f("1985-07-01", 1, "2/2"), f("1986-10-01", 2, "2/2"), f("1987-12-01", 0, "2/2")]
    assert [x["properties"]["datetime"] for x in pick(feats, 5)["22"]] == ["1985-07-01", "1986-10-01"]
