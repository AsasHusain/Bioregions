import numpy as np
import rasterio
import rasterio.shutil
from rasterio.transform import from_origin

from ecoregions.natforest import overview


def test_overview_reads_from_leading_bytes_only(tmp_path):
    a = (np.arange(2048 * 2048) % 251).astype("uint8").reshape(2048, 2048)
    src = tmp_path / "full.tif"
    prof = dict(driver="GTiff", height=2048, width=2048, count=1, dtype="uint8", crs="EPSG:32641",
                transform=from_origin(500000, 3500000, 10, 10), nodata=255)
    with rasterio.open(src, "w", **prof) as d:
        d.write(a, 1)
    cog = tmp_path / "cog.tif"
    with rasterio.open(src) as s:
        rasterio.shutil.copy(s, cog, driver="COG", compress="DEFLATE", overview_resampling="nearest", blocksize=256)
    data = cog.read_bytes()
    bounds, i, need = overview(data[: 1 << 16], 8)
    assert need < len(data) // 4
    assert 62.9 < bounds[0] < 63.1 and 31.4 < bounds[1] < 31.5
    with rasterio.MemoryFile(data[:need]) as mf, mf.open(overview_level=i) as s:
        assert s.shape == (256, 256) and s.res == (80, 80)
        assert (s.read(1) == a[::8, ::8]).mean() > 0.99
