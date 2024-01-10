"""
ISCE3 burst-based RTC processing for single or multiple bursts.
"""
import argparse
import copy
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import asf_search
import rasterio
from dem_stitcher import stitch_dem
from hyp3lib.get_orb import downloadSentinelOrbitFile
from isce3.product import GeoGridParameters
from rtc import helpers
from rtc.mosaic_geobursts import mosaic_single_output_file
from rtc.rtc_s1_single_job import run_single_job
from rtc.runconfig import (
    RunConfig,
    check_geogrid_dict,
    generate_geogrids,
    load_parameters,
    wrap_namespace,
)
from ruamel.yaml import YAML
from s1reader.s1_reader import load_single_burst
from shapely import unary_union
from shapely.geometry import Polygon, shape

from hyp3_isce3 import __version__, utils


log = logging.getLogger(__name__)


INPUT_DIR = Path('input_dir')
SCRATCH_DIR = Path('scratch_dir')
OUTPUT_DIR = Path('output_dir')


@dataclass
class BurstInfo:
    """Dataclass for storing burst information."""

    granule: str
    slc_granule: str
    polarization: str
    direction: str
    date: datetime
    data_url: Path
    data_path: Path
    metadata_url: Path
    metadata_path: Path
    footprint: Polygon


def download_dem_for_footprint(footprint: Polygon, dem_path: Path) -> None:
    """Download a DEM for the given footprint.

    Args:
        footprint: The footprint to download a DEM for.
        dem_path: The path to download the DEM to.
    """
    X, p = stitch_dem(footprint.bounds, dem_name='glo_30', dst_ellipsoidal_height=False, dst_area_or_point='Point')
    with rasterio.open(dem_path, 'w', **p) as ds:
        ds.write(X, 1)
        ds.update_tags(AREA_OR_POINT='Point')


def get_burst_info(granules: Iterable[str], save_dir: Path) -> List[BurstInfo]:
    """Get burst information from ASF Search.

    Args:
        granules: The burst granules to get information for.
        save_dir: The directory to save the data to.
    Returns:
        A list of BurstInfo objects.
    """
    burst_infos = []
    for granule in granules:
        results = asf_search.search(product_list=[granule])
        if len(results) == 0:
            raise ValueError(f'ASF Search failed to find {granule}.')
        if len(results) > 1:
            raise ValueError(f'ASF Search found multiple results for {granule}.')
        result = results[0]

        burst_granule = result.properties['fileID']
        slc_granule = result.umm['InputGranules'][0].split('-')[0]
        polarization = burst_granule.split('_')[4].upper()
        direction = result.properties['flightDirection'].upper()
        date_format = '%Y%m%dT%H%M%S'
        burst_time_str = burst_granule.split('_')[3]
        burst_time = datetime.strptime(burst_time_str, date_format)
        data_url = result.properties['url']
        data_path = save_dir / f'{burst_granule}.tiff'
        metadata_url = result.properties['additionalUrls'][0]
        metadata_path = save_dir / f'{burst_granule}.xml'
        footprint = shape(result.geojson()['geometry'])

        burst_info = BurstInfo(
            burst_granule,
            slc_granule,
            polarization,
            direction,
            burst_time,
            data_url,
            data_path,
            metadata_url,
            metadata_path,
            footprint,
        )

        burst_infos.append(burst_info)
    return burst_infos


def prep_data(
    burst_infos: List[BurstInfo],
    save_dir: Path,
    dem_name: str = 'dem.tiff',
    esa_username: str = None,
    esa_password: str = None,
    max_workers: int = 6,
) -> None:
    """Download data needed for RTC processing using multiple threads.

    Args:
        burst_infos: The information of the bursts to download.
        save_dir: The directory to save the data to.
        dem_name: The name to give the downloaded DEM.
        esa_username: The ESA CDSE username to use for downloading orbit files.
        esa_password: The ESA CDSE password to use for downloading orbit files.
        max_workers: The maximum number of threads to use for downloading.
    """
    if (esa_username is None) or (esa_password is None):
        esa_username, esa_password = utils.get_esa_credentials()
    esa_creds = (esa_username, esa_password)

    save_dir.mkdir(exist_ok=True, parents=True)
    dem_path = save_dir / dem_name

    full_footprint = unary_union([x.footprint for x in burst_infos]).buffer(0.15)
    slc_granules = list(set([x.slc_granule for x in burst_infos]))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.submit(download_dem_for_footprint, full_footprint, dem_path)

        for info in burst_infos:
            executor.submit(asf_search.download_url, info.data_url, info.data_path.parent, info.data_path.name)
            executor.submit(
                asf_search.download_url, info.metadata_url, info.metadata_path.parent, info.metadata_path.name
            )

        for slc_granule in slc_granules:
            executor.submit(downloadSentinelOrbitFile, slc_granule, str(save_dir), esa_credentials=esa_creds)


def check_group_validity(burst_infos: Iterable[BurstInfo]) -> None:
    """Check that a set of burst products are valid for merging. This includes:
    No more than 30 bursts are being processed.
    All bursts have the same:
        - polarization
        - direction (ascending or descending)
    All bursts were collected with 7 days of each other.
    All bursts are contiguous. (every burst footprint intersects the union of all other burst footprints)
    """
    if len(burst_infos) > 30:
        raise ValueError('Too many bursts to process. Maximum of 30 bursts allowed.')

    polarizations = set([x.polarization for x in burst_infos])
    if len(polarizations) > 1:
        raise ValueError('All bursts must have the same polarization.')

    directions = set([x.direction for x in burst_infos])
    if len(directions) > 1:
        raise ValueError('All bursts must have the same orbit direction (ascending or descending.')

    dates = [x.date for x in burst_infos]
    if max(dates) - min(dates) > timedelta(days=7):
        raise ValueError('All bursts must have been collected within 7 days of each other.')

    footprints = [x.footprint for x in burst_infos]
    for i, footprint in enumerate(footprints):
        other_footprints = footprints[:i] + footprints[i + 1 :]
        union = unary_union(other_footprints)
        if not footprint.intersects(union):
            raise ValueError('All bursts must be contiguous.')


def set_hyp3_defaults(cfg_dict: dict) -> dict:
    """Set hyp3 defaults for RTC processing.

    Args:
        cfg_dict: The config dictionary to update.
    Returns:
        The updated config dictionary.
    """
    cfg_dict['runconfig']['name'] = 'hyp3_job'
    cfg_dict['runconfig']['groups']['input_file_group']['safe_file_path'] = 'single_burst'
    cfg_dict['runconfig']['groups']['input_file_group']['burst_id'] = None
    cfg_dict['runconfig']['groups']['processing']['geocoding']['num_workers'] = 1
    cfg_dict['runconfig']['groups']['processing']['geocoding']['memory_mode'] = 'auto'
    cfg_dict['runconfig']['groups']['processing']['geocoding']['apply_shadow_masking'] = False
    cfg_dict['runconfig']['groups']['processing']['polarization'] = 'co-pol'
    cfg_dict['runconfig']['groups']['processing']['geo2rdr']['numiter'] = 25
    cfg_dict['runconfig']['groups']['processing']['geo2rdr']['threshold'] = 1e-08
    cfg_dict['runconfig']['groups']['processing']['geocoding']['save_nlooks'] = True
    cfg_dict['runconfig']['groups']['processing']['geocoding']['save_dem'] = True
    cfg_dict['runconfig']['groups']['processing']['geocoding']['save_mask'] = True
    cfg_dict['runconfig']['groups']['processing']['geocoding']['save_incidence_angle'] = True
    cfg_dict['runconfig']['groups']['processing']['geocoding']['save_rtc_anf'] = True
    cfg_dict['runconfig']['groups']['processing']['rtc']['dem_upsampling'] = 1
    cfg_dict['runconfig']['groups']['product_group']['output_dir'] = str(OUTPUT_DIR)
    cfg_dict['runconfig']['groups']['product_group']['output_imagery_compression'] = 'ZSTD'
    cfg_dict['runconfig']['groups']['product_group']['output_imagery_format'] = 'GTiff'
    cfg_dict['runconfig']['groups']['product_group']['processing_type'] = 'CUSTOM'
    cfg_dict['runconfig']['groups']['product_group']['product_path'] = '.'
    cfg_dict['runconfig']['groups']['product_group']['product_id'] = 'rtc'
    cfg_dict['runconfig']['groups']['product_group']['scratch_path'] = str(SCRATCH_DIR)
    cfg_dict['runconfig']['groups']['product_group']['output_imagery_nbits'] = 16
    cfg_dict['runconfig']['groups']['product_group']['save_mosaics'] = False
    cfg_dict['runconfig']['groups']['product_group']['save_browse'] = False
    cfg_dict['runconfig']['groups']['product_group']['save_secondary_layers_as_hdf5'] = False
    return cfg_dict


def set_important_options(
    cfg_dict: dict, dem_path: str, orbit_paths: Iterable[str], pixel_size: float, radiometry: str
) -> dict:
    """Set important options for RTC processing.

    Args:
        cfg_dict: The config dictionary to update.
        dem_path: The path to the DEM to use.
        orbit_paths: The paths to the orbit files to use.
        pixel_size: The pixel size to use.
        radiometry: The radiometry to use (gamma0, sigma0, or uncorrected).
    Returns:
        The updated config dictionary.
    """
    cfg_dict['runconfig']['groups']['dynamic_ancillary_file_group']['dem_file'] = dem_path
    cfg_dict['runconfig']['groups']['input_file_group']['orbit_file_path'] = orbit_paths

    bursts_geogrid = cfg_dict['runconfig']['groups']['processing']['geocoding']['bursts_geogrid']
    mosaic_geogrid = cfg_dict['runconfig']['groups']['processing']['mosaicking']['mosaic_geogrid']
    for grid in (bursts_geogrid, mosaic_geogrid):
        for field in ('x_posting', 'y_posting', 'x_snap', 'y_snap'):
            grid[field] = pixel_size

    cfg_dict['runconfig']['groups']['processing']['geocoding']['bursts_geogrid'] = bursts_geogrid
    cfg_dict['runconfig']['groups']['processing']['mosaicking']['mosaic_geogrid'] = mosaic_geogrid

    if radiometry == 'uncorrected':
        cfg_dict['runconfig']['groups']['processing']['apply_rtc'] = False
    else:
        cfg_dict['runconfig']['groups']['processing']['rtc']['output_type'] = radiometry

    return cfg_dict


def get_valid_orbit(burst_granule: str, orbit_paths: Iterable[Path]) -> Path:
    """Select the valid orbit file for the given burst from a list of orbit files.

    Args:
        burst_granule: The burst granule to find the orbit file for.
        orbit_paths: The paths to the orbit files to choose from.
    Returns:
        The path to the valid orbit file.
    """
    date_format = '%Y%m%dT%H%M%S'
    burst_time_str = burst_granule.split('_')[3]
    burst_time = datetime.strptime(burst_time_str, date_format)

    for orbit in orbit_paths:
        orbit_name = orbit.name.split('.')[0]
        start_time_str = orbit_name.split('_')[6][1:]
        start_time = datetime.strptime(start_time_str, date_format)

        stop_time_str = orbit_name.split('_')[7]
        stop_time = datetime.strptime(stop_time_str, date_format)

        if burst_time > start_time and burst_time < stop_time:
            return orbit

    raise ValueError(f'No valid orbit file provided for burst {burst_granule}')


def create_configs(
    burst_infos: Iterable[BurstInfo], orbit_paths: Iterable[Path], dem_path: Path, pixel_size: float, radiometry: str
) -> Tuple[GeoGridParameters, List[RunConfig]]:
    """Create RunConfig objects for each input burst.

    Args:
        burst_infos: The information of the bursts to process.
        orbit_paths: The paths to the orbit files to use.
        dem_path: The path to the DEM to use.
        pixel_size: The pixel size to use.
        radiometry: The radiometry to use (gamma0, sigma0, or uncorrected).
    Returns:
        The mosaic geogrid parameters for merging, and a list of RunConfig objects customized for each burst.
    """
    parser = YAML(typ='safe')
    default_cfg_path = f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/rtc_s1.yaml'
    with open(default_cfg_path, 'r') as f_default:
        cfg = parser.load(f_default)

    cfg = set_hyp3_defaults(cfg)
    orbit_path_strs = [str(x) for x in orbit_paths]
    set_important_options(cfg, str(dem_path), orbit_path_strs, pixel_size, radiometry)

    groups_cfg = cfg['runconfig']['groups']

    mosaic_dict = groups_cfg['processing']['mosaicking']
    check_geogrid_dict(mosaic_dict['mosaic_geogrid'])

    geocoding_dict = groups_cfg['processing']['geocoding']
    check_geogrid_dict(geocoding_dict['bursts_geogrid'])

    sns = wrap_namespace(groups_cfg)

    bursts = {}
    burst_orbits = {}
    for info in burst_infos:
        burst_orbit = str(get_valid_orbit(info.granule, orbit_paths))

        burst_obj = load_single_burst(str(info.data_path), str(info.metadata_path), burst_orbit)
        bursts[str(burst_obj.burst_id)] = {burst_obj.polarization: burst_obj}
        burst_orbits[str(burst_obj.burst_id)] = burst_orbit

    geogrid_all, geogrids = generate_geogrids(bursts, geocoding_dict, mosaic_dict)

    burst_cfgs = []
    for burst_id in bursts:
        sns_copy = copy.deepcopy(sns)
        cfg_copy = copy.deepcopy(cfg)
        burst_cfg = RunConfig(
            cfg_copy['runconfig']['name'],
            sns_copy,
            {burst_id: bursts[burst_id]},
            {},
            None,
            geogrid_all,
            {burst_id: geogrids[burst_id]},
            burst_orbits[burst_id],
        )
        burst_cfgs.append(burst_cfg)

    return geogrid_all, burst_cfgs


def create_file_list(polarization: str = 'VV') -> dict:
    """Create a dictionary of output file types and their paths.

    Args:
        polarization: The polarization to use.
    Returns:
        A dictionary of output file types and their paths.
    """
    file_types = [
        f'rtc_{polarization.upper()}.tif',
        'rtc_number_of_looks.tif',
        'rtc_incidence_angle.tif',
        'rtc_mask.tif',
        'rtc_interpolated_dem.tif',
        'rtc_rtc_anf_gamma0_to_beta0.tif',
    ]
    file_groups = {}
    for file_type in file_types:
        files = sorted([str(x) for x in OUTPUT_DIR.glob(f'./*/{file_type}')])
        if len(files) > 0:
            file_groups[file_type] = files

    return file_groups


def burst_rtc(granules: Iterable[str], pixelsize: float, radiometry: str) -> None:
    """Run RTC processing on a single or multiple bursts.

    Args:
        granules: The burst granules to process.
        pixelsize: The pixel size to use.
        radiometry: The radiometry to use (gamma0, sigma0, or uncorrected).
    """
    burst_infos = get_burst_info(granules, INPUT_DIR)
    check_group_validity(burst_infos)
    dem_path = INPUT_DIR / 'dem.tiff'
    prep_data(burst_infos, INPUT_DIR, dem_path.name)
    orbit_paths = [Path(x) for x in INPUT_DIR.glob('*.EOF')]
    mosaic_geogrid, cfgs = create_configs(burst_infos, orbit_paths, dem_path, pixelsize, radiometry)
    for cfg in cfgs:
        load_parameters(cfg)
        run_single_job(cfg)

    file_groups = create_file_list()
    if len(granules) > 1:
        for name in file_groups:
            mosaic_single_output_file(
                file_groups[name],
                file_groups['rtc_number_of_looks.tif'],
                OUTPUT_DIR / name,
                'first',
                SCRATCH_DIR,
                geogrid_in=mosaic_geogrid,
            )
    else:
        for name in file_groups:
            shutil.copy(file_groups[name][0], OUTPUT_DIR / name)


def main():
    """Entrypoint for burst_rtc command line usage."""
    parser = argparse.ArgumentParser(
        prog='create_rtc',
        description=__doc__,
    )
    parser.add_argument('granules', type=str.split, nargs='+')
    parser.add_argument('--pixelsize', type=float, choices=[10.0, 20.0, 30.0], default=30.0)
    parser.add_argument('--radiometry', choices=['gamma0', 'sigma0', 'uncorrected'], default='gamma0')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    args = parser.parse_args()
    args.granules = [item for sublist in args.granules for item in sublist]

    burst_rtc(args.granules, args.pixelsize, args.radiometry)


if __name__ == '__main__':
    main()
