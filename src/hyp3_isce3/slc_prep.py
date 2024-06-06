from pathlib import Path
import asf_search
import rasterio
from shapely.geometry import shape

from hyp3lib.get_orb import downloadSentinelOrbitFile

from hyp3_isce3 import utils
from hyp3_isce3.isce3_rtc import download_dem_for_footprint


def download_granule(granule_name: str, output_dir: Path):
    """Download a S1 granule using asf_search.

    Args:
        granule_name: Name of the granule to download
        output_dir: Directory to save the granule in
    """
    result = asf_search.granule_search([granule_name])[0]
    footprint = shape(result.geojson()['geometry']).buffer(0.15)
    result.download(path=output_dir)
    return output_dir / granule_name, footprint


def prep_slc(
    slc_granule: str,
    save_dir: Path,
    dem_name: str = 'dem.tiff',
    esa_username: str = None,
    esa_password: str = None,
) -> None:
    """Download data needed for RTC processing using multiple threads.

    Args:
        slc_granule: slc granule to download.
        save_dir: The directory to save the data to.
        dem_name: The name to give the downloaded DEM.
        esa_username: The ESA CDSE username to use for downloading orbit files.
        esa_password: The ESA CDSE password to use for downloading orbit files.
    """
    if (esa_username is None) or (esa_password is None):
        esa_username, esa_password = utils.get_esa_credentials()
    esa_creds = (esa_username, esa_password)

    save_dir.mkdir(exist_ok=True, parents=True)
    slc_path, slc_footprint = download_granule(slc_granule, save_dir)

    dem_path = save_dir / dem_name
    download_dem_for_footprint(slc_footprint, dem_path)

    orbit_path, _ = downloadSentinelOrbitFile(slc_granule, str(save_dir), esa_credentials=esa_creds)
    orbit_path = Path(orbit_path)

if __name__ == '__main__':
    prep_slc('S1B_IW_SLC__1SDV_20180504T104507_20180504T104535_010770_013AEE_919F-SLC', Path.cwd())
