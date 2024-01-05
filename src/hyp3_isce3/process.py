"""
ISCE3 processing
"""
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
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
from shapely.geometry import shape

from hyp3_isce3 import __version__, utils


log = logging.getLogger(__name__)


def download_dem_for_footprint(footprint, dem_path):
    X, p = stitch_dem(footprint.bounds, dem_name='glo_30', dst_ellipsoidal_height=False, dst_area_or_point='Point')
    with rasterio.open(dem_path, 'w', **p) as ds:
        ds.write(X, 1)
        ds.update_tags(AREA_OR_POINT='Point')
    return dem_path


def prep_data(granule, esa_username=None, esa_password=None, async_download=True):
    if (esa_username is None) or (esa_password is None):
        esa_username, esa_password = utils.get_esa_credentials()
    esa_creds = (esa_username, esa_password)

    input_dir = Path('input_dir')
    input_dir.mkdir(exist_ok=True, parents=True)
    results = asf_search.search(product_list=[granule])
    if len(results) == 0:
        raise ValueError(f'ASF Search failed to find {granule}.')
    if len(results) > 1:
        raise ValueError(f'ASF Search found multiple results for {granule}.')
    result = results[0]

    data_url = result.properties['url']
    metadata_url = result.properties['additionalUrls'][0]
    burst_granule = result.properties['fileID']
    slc_granule = result.umm['InputGranules'][0].split('-')[0]
    footprint = shape(result.geojson()['geometry']).buffer(0.2)
    dem_path = input_dir / 'dem.tiff'

    burst_path = input_dir / f'{burst_granule}.tiff'
    metadata_path = input_dir / f'{burst_granule}.xml'
    if async_download:
        with ThreadPoolExecutor() as executor:
            executor.submit(download_dem_for_footprint, footprint, dem_path)
            executor.submit(downloadSentinelOrbitFile, slc_granule, str(input_dir), esa_credentials=esa_creds)
            executor.submit(asf_search.download_url, data_url, input_dir, burst_path.name)
            executor.submit(asf_search.download_url, metadata_url, input_dir, metadata_path.name)
    else:
        with asf_search.ASFSession() as sess:
            asf_search.download_url(data_url, input_dir, burst_path.name, sess)
            asf_search.download_url(metadata_url, input_dir, metadata_path.name, sess)
        download_dem_for_footprint(footprint, dem_path)
        downloadSentinelOrbitFile(slc_granule, str(input_dir), esa_credentials=esa_creds)

    orbit_path = Path(list(input_dir.glob('*.EOF'))[0])

    return burst_path, metadata_path, orbit_path, dem_path


def set_hyp3_defaults(cfg_dict):
    cfg_dict['runconfig']['name'] = 'hyp3_job'
    cfg_dict['runconfig']['groups']['input_file_group']['safe_file_path'] = 'single_burst'
    cfg_dict['runconfig']['groups']['input_file_group']['burst_id'] = None
    cfg_dict['runconfig']['groups']['processing']['geocoding']['memory_mode'] = 'auto'
    cfg_dict['runconfig']['groups']['processing']['geocoding']['apply_shadow_masking'] = False
    cfg_dict['runconfig']['groups']['processing']['polarization'] = 'co-pol'
    cfg_dict['runconfig']['groups']['processing']['geo2rdr']['numiter'] = 25
    cfg_dict['runconfig']['groups']['processing']['geo2rdr']['threshold'] = 1e-08
    cfg_dict['runconfig']['groups']['processing']['geocoding']['save_mask'] = False
    cfg_dict['runconfig']['groups']['processing']['geocoding']['save_nlooks'] = True
    cfg_dict['runconfig']['groups']['processing']['geocoding']['save_rtc_anf'] = True
    cfg_dict['runconfig']['groups']['processing']['rtc']['dem_upsampling'] = 1
    cfg_dict['runconfig']['groups']['product_group']['output_dir'] = 'output_dir'
    cfg_dict['runconfig']['groups']['product_group']['output_imagery_compression'] = 'ZSTD'
    cfg_dict['runconfig']['groups']['product_group']['output_imagery_format'] = 'GTiff'
    cfg_dict['runconfig']['groups']['product_group']['processing_type'] = 'CUSTOM'
    cfg_dict['runconfig']['groups']['product_group']['product_path'] = '.'
    cfg_dict['runconfig']['groups']['product_group']['product_id'] = 'burst'
    cfg_dict['runconfig']['groups']['product_group']['scratch_path'] = 'scratch_dir'
    cfg_dict['runconfig']['groups']['product_group']['output_imagery_nbits'] = 16
    cfg_dict['runconfig']['groups']['product_group']['save_mosaics'] = False
    cfg_dict['runconfig']['groups']['product_group']['save_secondary_layers_as_hdf5'] = True
    return cfg_dict


def set_important_options(cfg_dict, dem_path, orbit_path, pixel_size, radiometry):
    cfg_dict['runconfig']['groups']['dynamic_ancillary_file_group']['dem_file'] = dem_path
    cfg_dict['runconfig']['groups']['input_file_group']['orbit_file_path'] = [orbit_path]

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


def create_config(burst_path, metadata_path, orbit_path, dem_path, pixel_size, radiometry) -> RunConfig:
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
    set_important_options(cfg, str(dem_path), str(orbit_path), pixel_size, radiometry)

    groups_cfg = cfg['runconfig']['groups']

    # Read mosaic dict
    mosaic_dict = groups_cfg['processing']['mosaicking']
    check_geogrid_dict(mosaic_dict['mosaic_geogrid'])

    # Read geocoding dict
    geocoding_dict = groups_cfg['processing']['geocoding']
    check_geogrid_dict(geocoding_dict['bursts_geogrid'])

    # Convert runconfig dict to SimpleNamespace
    sns = wrap_namespace(groups_cfg)

    # Load bursts
    burst_obj = load_single_burst(str(burst_path), str(metadata_path), str(orbit_path))
    bursts_new = {str(burst_obj.burst_id): {burst_obj.polarization: burst_obj}}
    bursts = bursts_new

    # Load geogrids
    geogrid_all, geogrids = generate_geogrids(bursts, geocoding_dict, mosaic_dict)

    # Empty reference dict for base runconfig class constructor
    empty_ref_dict = {}

    configuration = RunConfig(
        cfg['runconfig']['name'], sns, bursts, empty_ref_dict, None, geogrid_all, geogrids, orbit_path
    )

    return configuration


def create_rtc(granule: str, pixelsize: int, radiometry: str) -> None:
    """Create a greeting product

    Args:
    """
    burst_path, metadata_path, orbit_path, dem_path = prep_data(granule)
    cfg = create_config(burst_path, metadata_path, orbit_path, dem_path, pixelsize, radiometry)
    load_parameters(cfg)
    run_single_job(cfg)


def create_file_list(polarization='VV'):
    output_dir = Path('output_dir')
    rtcs = sorted([str(x) for x in output_dir.glob(f'./*/burst_{polarization.upper()}.tif')])
    auxs = sorted([str(x) for x in output_dir.glob('./*/burst.h5')])
    # HDF5:"output_dir/t064_136231_iw2/burst.h5"://data/numberOfLooks
    n_looks = [f'HDF5:{x}://data/numberOfLooks' for x in auxs]
    return rtcs, n_looks


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
    for granule in args.granules:
        create_rtc(granule, args.pixelsize, args.radiometry)

    if len(args.granules) > 1:
        rtcs, n_looks = create_file_list()
        mosaic_single_output_file(rtcs, n_looks, 'final.tif', 'first', 'scratch_dir')


if __name__ == '__main__':
    main()
