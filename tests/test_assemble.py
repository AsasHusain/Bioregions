import numpy as np

from ecoregions.assemble import belt_zones, block_share


def test_belt_patches_become_one_zone():
    # a belt type (5) as scattered 90 m patches covering about 40 % of the land: as a zone it takes the area
    rng = np.random.default_rng(0)
    cls90 = np.where(rng.random((90, 90)) < 0.4, 5, 1).astype("uint8")
    cls = np.ones((30, 30), "uint8")  # the 270 m mode is the surrounding vegetation (1) everywhere
    assert np.allclose(block_share(cls90, 5, 3).mean(), (cls90 == 5).mean(), atol=1e-6)
    out = belt_zones(cls, cls90, 3, [5], {"window_km": 2, "min_share": 0.3}, 270.0)
    assert (out == 5).mean() > 0.9
    assert (belt_zones(cls, cls90, 3, [5], {"window_km": 2, "min_share": 0.6}, 270.0) == 5).sum() == 0
