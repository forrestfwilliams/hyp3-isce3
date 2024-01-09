"""
ISCE3 burst-based RTC workflow
"""
import argparse
import copy
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import asf_search
import rasterio
from dem_stitcher import stitch_dem
from hyp3lib.get_orb import downloadSentinelOrbitFile
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
OUTPUT_DIR = Path('output_dir')


def download_dem_for_footprint(footprint, dem_path):
    X, p = stitch_dem(footprint.bounds, dem_name='glo_30', dst_ellipsoidal_height=False, dst_area_or_point='Point')
    with rasterio.open(dem_path, 'w', **p) as ds:
        ds.write(X, 1)
        ds.update_tags(AREA_OR_POINT='Point')
    return dem_path


@dataclass
class BurstInfo:
    granule: str
    slc_granule: str
    data_url: Path
    data_path: Path
    metadata_url: Path
    metadata_path: Path
    footprint: Polygon


def get_burst_info(granules, save_dir):
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
        data_url = result.properties['url']
        data_path = save_dir / f'{burst_granule}.tiff'
        metadata_url = result.properties['additionalUrls'][0]
        metadata_path = save_dir / f'{burst_granule}.xml'
        footprint = shape(result.geojson()['geometry'])
        burst_info = BurstInfo(burst_granule, slc_granule, data_url, data_path, metadata_url, metadata_path, footprint)
        burst_infos.append(burst_info)
    return burst_infos


def prep_data(burst_infos, save_dir, dem_name='dem.tiff', esa_username=None, esa_password=None, max_workers=6):
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


def set_hyp3_defaults(cfg_dict):
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
    cfg_dict['runconfig']['groups']['product_group']['scratch_path'] = 'scratch_dir'
    cfg_dict['runconfig']['groups']['product_group']['output_imagery_nbits'] = 16
    cfg_dict['runconfig']['groups']['product_group']['save_mosaics'] = False
    cfg_dict['runconfig']['groups']['product_group']['save_browse'] = False
    cfg_dict['runconfig']['groups']['product_group']['save_secondary_layers_as_hdf5'] = False
    return cfg_dict


def set_important_options(cfg_dict, dem_path, orbit_paths, pixel_size, radiometry):
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


def create_configs(burst_infos, orbit_paths, dem_path, pixel_size, radiometry) -> RunConfig:
    """Initialize RunConfig class with options from given yaml file.

    Parameters
    ----------
    yaml_path : str
        Path to yaml file containing the options to load
    """
    # load default runconfig
    parser = YAML(typ='safe')
    default_cfg_path = f'{helpers.WORKFLOW_SCRIPTS_DIR}/defaults/rtc_s1.yaml'
    with open(default_cfg_path, 'r') as f_default:
        cfg = parser.load(f_default)

    cfg = set_hyp3_defaults(cfg)
    orbit_path_strs = [str(x) for x in orbit_paths]
    set_important_options(cfg, str(dem_path), orbit_path_strs, pixel_size, radiometry)

    groups_cfg = cfg['runconfig']['groups']

    # Read mosaic dict
    mosaic_dict = groups_cfg['processing']['mosaicking']
    check_geogrid_dict(mosaic_dict['mosaic_geogrid'])

    # Read geocoding dict
    geocoding_dict = groups_cfg['processing']['geocoding']
    check_geogrid_dict(geocoding_dict['bursts_geogrid'])

    # Convert runconfig dict to SimpleNamespace
    sns = wrap_namespace(groups_cfg)

    # Load burst
    bursts = {}
    burst_orbits = {}
    for info in burst_infos:
        # FIXME: this is a hack to get the orbit path
        burst_orbit = str(orbit_paths[0])

        burst_obj = load_single_burst(str(info.data_path), str(info.metadata_path), burst_orbit)
        bursts[str(burst_obj.burst_id)] = {burst_obj.polarization: burst_obj}
        burst_orbits[str(burst_obj.burst_id)] = burst_orbit

    # Load geogrids
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


def create_file_list(polarization='VV'):
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


def burst_rtc(granules: str, pixelsize: int, radiometry: str) -> None:
    """Create a greeting product

    Args:
    """
    burst_infos = get_burst_info(granules, INPUT_DIR)
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
                file_groups['burst_number_of_looks.tif'],
                OUTPUT_DIR / name,
                'first',
                'scratch_dir',
                geogrid_in=mosaic_geogrid,
            )
    else:
        for name in file_groups:
            shutil.copy(file_groups[name][0], OUTPUT_DIR / name)

    return mosaic_geogrid


def main():
    """process_isce3 entrypoint"""
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
