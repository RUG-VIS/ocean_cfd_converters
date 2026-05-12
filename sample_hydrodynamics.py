"""
Author: Dr. Christian Kehl
Date: 06-07-2025
"""
from argparse import ArgumentParser
# from datetime import timedelta
from glob import glob
import math
import datetime
import numpy as np
# from numpy.random import default_rng

import xarray as xr
# import dask.array as da
from netCDF4 import Dataset
import h5py

# import fnmatch
import gc
import os
# import time as ostime

# from scipy.interpolate import interpn, griddata
# from scipy.spatial import Delaunay
# from scipy.interpolate import LinearNDInterpolator
# import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
# import itertools
from polyTEOS10_consts import *

from multiprocessing import Pool

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    import numba
except:
    numba = None
# from numba import njit, prange

with_GC = False
DBG_MSG = False

# a = 9.6 * 1e3 # [a in km -> 10e3]
a = 359.0
# b = 4.8 * 1e3 # [b in km -> 10e3]
b = 179.0
# c = 1.0
c = 2.1 * 1e3  # by definiton: meters
tsteps = 122 # in steps
tstepsize = 6.0 # unitary
tscale = 12.0*60.0*60.0 # in seconds


def interp_weights(xyz, uvw):
    d = 3
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

# Helper function for time-conversion from the calendar format
def convert_timearray(t_array, dt_minutes, ns_per_sec, debug=False, array_name="time array"):
    """

    :param t_array: 2D array of time values in either calendar- or float-time format; dim-0 = object entities, dim-1 = time steps (or 1D with just timesteps)
    :param dt_minutes: expected delta_t als float-value (in minutes)
    :param ns_per_sec: conversion value of number of nanoseconds within 1 second
    :param debug: parameter telling to print debug messages or not
    :param array_name: name of the array (for debug prints)
    :return: converted t_array
    """
    ta = t_array
    while len(ta.shape) > 1:
        ta = ta[0]
    if isinstance(ta[0], datetime.datetime) or isinstance(ta[0], datetime.timedelta) or isinstance(ta[0], np.timedelta64) or isinstance(ta[0], np.datetime64) or np.float64(ta[1]-ta[0]) > (dt_minutes+dt_minutes/2.0):
        if debug:
            print("{}.dtype before conversion: {}".format(array_name, t_array.dtype))
        t_array = (t_array / ns_per_sec).astype(np.float64)
        ta = (ta / ns_per_sec).astype(np.float64)
        if debug:
            print("{0}.range and {0}.dtype after conversion: ({1}, {2}) \t {3}".format(array_name, ta.min(), ta.max(), ta.dtype))
    else:
        if debug:
            print("{0}.range and {0}.dtype: ({1}, {2}) \t {3} \t(no conversion applied)".format(array_name, ta.min(), ta.max(), ta.dtype))
        pass
    return t_array

def convert_timevalue(in_val, t0, ns_per_sec, debug=False):
    """
    :param in_val: input value
    :param t0: reference time value, in format of 'datetime.datetime' or 'np.datetime'
    :param ns_per_sec: float64 value of nanoseconds per second
    :param debug: debug-switch to print debug information
    """
    if debug:
        print("input value: {}".format(in_val))
    tval = in_val
    if isinstance(tval, datetime.datetime) or isinstance(tval, np.datetime64):
        tval = tval - t0
        if debug:
            print("converted timestep to time difference: {}".format(tval))
    if isinstance(tval, datetime.timedelta) or isinstance(tval, np.timedelta64):
        tval = np.array([tval / ns_per_sec], dtype=np.float64)[0]
        if debug:
            print("converted timedelta-value to float value: {}".format(tval))
    return tval

# -------------------------------------------------------------------------------------------------------------------- #
def get_data_of_ndarray_nc(data_array):
    """
    :param data_array: input field
    :return: tuple of data_array np.nanmin, np.nanmax, data0, data_dx
    """
    if data_array is None or data_array.shape[0] == 0:
        return None, None, None, None
    darray = data_array.data
    dmin = np.nanmin(data_array)
    dmax = np.nanmax(data_array)
    d0 = None
    data_dx = None
    if len(data_array.shape) == 1:
        d0 = darray[0]
        if data_array.shape[0] > 1:
            data_dx = darray[1] - darray[0]
    del darray
    return dmin, dmax, d0, data_dx

def get_data_of_ndarray_h5(data_array):
    """
    :param data_array: input field
    :return: tuple of data_array np.nanmin, np.nanmax, data0, data_dx
    """
    if data_array is None or data_array.shape[0] == 0:
        return None, None, None, None
    darray = data_array[()]
    dmin = np.nanmin(data_array)
    dmax = np.nanmax(data_array)
    d0 = None
    data_dx = None
    if len(data_array.shape) == 1:
        d0 = darray[0]
        if data_array.shape[0] > 1:
            data_dx = darray[1] - darray[0]
    del darray
    return dmin, dmax, d0, data_dx

# -------------------------------------------------------------------------------------------------------------------- #

def time_index_value(tx, _ft, periodic, _ft_dt=None):  # , _ft_min=None, _ft_max=None
    # expect ft to be forward-linear
    ft = _ft
    if isinstance(_ft, xr.DataArray):
        ft = ft.data
    f_dt = _ft_dt
    if f_dt is None:
        f_dt = ft[1] - ft[0]
        if type(f_dt) not in [np.float64, np.float32]:
            f_dt = datetime.timedelta(f_dt).total_seconds()
    # f_interp = (f_min + tx) / f_dt if f_dt >= 0 else (f_max + tx) / f_dt
    f_interp = tx / f_dt
    ti = int(math.floor(f_interp))
    if periodic:
        ti = ti % ft.shape[0]
    else:
        ti = max(0, min(ft.shape[0]-1, ti))
    return ti

def time_partion_value(tx, _ft, periodic, _ft_dt=None):  # , _ft_min=None, _ft_max=None
    # expect ft to be forward-linear
    ft = _ft
    if isinstance(_ft, xr.DataArray):
        ft = ft.data
    f_dt = _ft_dt
    if f_dt is None:
        f_dt = ft[1] - ft[0]
        if type(f_dt) not in [np.float64, np.float32]:
            f_dt = datetime.timedelta(f_dt).total_seconds()
    # f_interp = (f_min + tx) / f_dt if f_dt >= 0 else (f_max + tx) / f_dt
    f_interp = abs(tx / f_dt)
    if periodic:
        # print("f_interp = math.fmod({}, {})".format(f_interp, float(ft.shape[0])))
        f_interp = math.fmod(f_interp, float(ft.shape[0]))
    else:
        # print("f_interp = max({}, min({}, {}))".format(ft[0], ft[-1], f_interp))
        f_interp = max(0.0, min(float(ft.shape[0]-1), f_interp))
    f_t = f_interp - math.floor(f_interp)
    return f_t

def lat_index_value(lat, _fl):
    # expect fl to be forward-linear
    fl = _fl
    if isinstance(_fl, xr.DataArray):
        fl = fl.data
    f_dL = fl[1] - fl[0]
    f_interp = lat / f_dL
    lati = int(math.floor(f_interp))
    return lati

def lat_partion_value(lat, _fl):
    # expect ft to be forward-linear
    fl = _fl
    if isinstance(_fl, xr.DataArray):
        fl = fl.data
    f_dL = fl[1] - fl[0]
    f_interp = lat / f_dL
    lat_t = f_interp - math.floor(f_interp)
    return lat_t

def depth_index_value(dx, _fd):
    # expect ft to be forward-linear
    fd = _fd
    if isinstance(_fd, xr.DataArray):
        fd = fd.data
    f_dD = fd[1] - fd[0]
    f_interp = dx / f_dD
    di = int(math.floor(f_interp))
    return di

def depth_partion_value(dx, _fd):
    # expect ft to be forward-linear
    fd = _fd
    if isinstance(_fd, xr.DataArray):
        fd = fd.data
    f_dD = fd[1] - fd[0]
    f_interp = dx / f_dD
    f_d = f_interp - math.floor(f_interp)
    return f_d

# -------------------------------------------------------------------------------------------------------------------- #



def perIterGC():
    gc.collect()

# ====
# start example: python3 density4hydrodynamics.py -d /media/christian/MyPassport/data/hydrodynamic/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/ -o /media/christian/MyPassport/data/hydrodynamic/CMEMS/GLOBAL/ -U mercatorglorys12v1_gl12_mean_2016* -V mercatorglorys12v1_gl12_mean_2016* -T mercatorglorys12v1_gl12_mean_2016* -N mercatorglorys12v1_gl12_mean_2016* --uvar uo --vvar vo --wvar None --tpvar thetao --navar so --xvar longitude --yvar latitude --zvar depth --tvar time -F nc --writeNC
#                python3 density4hydrodynamics.py -d /media/christian/MyPassport/data/hydrodynamic/CMEMS/GLOBAL_REANALYSIS_PHY_001_030_monthly/ -o /media/christian/MyPassport/data/hydrodynamic/CMEMS/GLOBAL/ -U mercatorglorys12v1_gl12_mean_2016* -V mercatorglorys12v1_gl12_mean_2016* -T mercatorglorys12v1_gl12_mean_2016* -N mercatorglorys12v1_gl12_mean_2016* --uvar uo --vvar vo --wvar None --tpvar thetao --navar so --xvar longitude --yvar latitude --zvar depth --tvar time -F nc --writeNC
#                python3 doublegyre_scenario.py -f vis_example/metadata.txt -t 366 -dt 60 -ot 720 -im 'rk4' -N 2**12 -sres 2.5 -gres 5 -sm 'regular_jitter' -fsx 360 -fsy 180
#                python3 doublegyre_scenario.py -f metadata.txt -t 366 -dt 60 -ot 1440 -im 'rk4' -N 2**12 -sres 1.25 -gres 2 -sm 'regular_jitter' -fsx 540 -fsy 270 -fsz 20 -3D
#                python3 density4hydrodynamics.py --hydrodir /media/christian/OneTouch/storage/data/hydrodynamics/CMEMS/2023-3D/currents/ --physdir /media/christian/OneTouch/storage/data/hydrodynamics/CMEMS/2023-3D/physics/ -o /media/christian/OneTouch/storage/data/hydrodynamics/CMEMS/2023-3D/physplus/ -U glo12_rg_1d-m_????????-????????_3D-uovo_* -V glo12_rg_1d-m_????????-????????_3D-uovo_* -W glo12_rg_1d-m_????????-????????_3D-wo_* -T glo12_rg_1d-m_????????-????????_3D-thetao_* -N glo12_rg_1d-m_????????-????????_3D-so_* --uvar uo --vvar vo --wvar wo --tpvar thetao --navar so --xvar longitude --yvar latitude --zvar depth --tvar time -F nc --writeNC -LOm -15.0 -LOM 45.0 -LAm -65.0 -LAM -20.0
#                python3 ./sample_hydrodynamics.py -d /media/christian/DATA/data/hydrodynamics/ENWS/reanalysis2D-2024/currents/ -o /media/christian/DATA/data/hydrodynamics/ENWS/reanalysis2D-2024/currents_resampled/ -U "metoffice_foam1_amm7_NWS_CUR_*" -V "metoffice_foam1_amm7_NWS_CUR_*" --uvar uo --vvar vo --xvar longitude --yvar latitude --zvar depth --tvar time -LOm -7.65 -LOM -6.05 -LAm 57.65 -LAM 59.03 -F nc --writeNC
#                python3 /media/christian/DATA/git/ocean_cfd_converters/sample_hydrodynamics.py -d /media/christian/OneTouch/storage/data/hydrodynamics/CMEMS/2023-3D/currents/ -o /media/christian/OneTouch/storage/data/hydrodynamics/SouthernOcean/currents/ -U "glo12_rg_1d-m_*_3D-uovo_*" -V "glo12_rg_1d-m_*_3D-uovo_*" -W "glo12_rg_1d-m_*_3D-wo_*" --uvar uo --vvar vo --wvar wo --xvar longitude --yvar latitude --zvar depth --tvar time -LOm -15.0 -LOM 45.0 -LAm -65.0 -LAM -20.0 -F nc --writeNC
# ====
if __name__=='__main__':
    parser = ArgumentParser(description="computes the density field for a given OGCM")
    parser.add_argument("-d", "--filedir", dest="filedir", type=str, default="None", help="head directory containing all input data and also are the store target for output files")
    parser.add_argument("--hydrodir", dest="hydrodir", type=str, default="None",
                        help="head directory containing all *hydrodynamic* data are located")
    parser.add_argument("-o", "--outdir", dest="outdir", type=str, default="None", help="head output directory")
    parser.add_argument("-U", "--Upattern", dest="Upattern", type=str, default='*U.nc', help="pattern of U-file(s)")
    parser.add_argument("--uvar", dest="uvar", type=str, default='vozocrtx', help="variable name of U")
    parser.add_argument("-V", "--Vpattern", dest="Vpattern", type=str, default='*V.nc', help="pattern of V-file(s)")
    parser.add_argument("--vvar", dest="vvar", type=str, default='vomecrty', help="variable name of V")
    parser.add_argument("-W", "--Wpattern", dest="Wpattern", type=str, default='*W.nc', help="pattern of W-file(s)")
    parser.add_argument("--wvar", dest="wvar", type=str, default='None', help="variable name of W")
    parser.add_argument("--xvar", dest="xvar", type=str, default="None", help="variable name of x")
    parser.add_argument("--xuvar", dest="xuvar", type=str, default="None", help="variable name of x in field 'U', if differing between fields.")
    parser.add_argument("--xvvar", dest="xvvar", type=str, default="None", help="variable name of x in field 'V', if differing between fields.")
    parser.add_argument("--xwvar", dest="xwvar", type=str, default="None", help="variable name of x in field 'W', if differing between fields.")
    parser.add_argument("--yvar", dest="yvar", type=str, default="None", help="variable name of y")
    parser.add_argument("--yuvar", dest="yuvar", type=str, default="None", help="variable name of y in field 'U', if differing between fields.")
    parser.add_argument("--yvvar", dest="yvvar", type=str, default="None", help="variable name of y in field 'V', if differing between fields.")
    parser.add_argument("--ywvar", dest="ywvar", type=str, default="None", help="variable name of y in field 'W', if differing between fields.")
    parser.add_argument("--zvar", dest="zvar", type=str, default="None", help="variable name of z")
    parser.add_argument("--zuvar", dest="zuvar", type=str, default="None", help="variable name of z in field 'U', if differing between fields.")
    parser.add_argument("--zvvar", dest="zvvar", type=str, default="None", help="variable name of z in field 'V', if differing between fields.")
    parser.add_argument("--zwvar", dest="zwvar", type=str, default="None", help="variable name of z in field 'W', if differing between fields.")
    parser.add_argument("--tvar", dest="tvar", type=str, default="None", help="variable name of t")
    parser.add_argument("--tuvar", dest="tuvar", type=str, default="None", help="variable name of t in field 'U', if differing between fields.")
    parser.add_argument("--tvvar", dest="tvvar", type=str, default="None", help="variable name of t in field 'V', if differing between fields.")
    parser.add_argument("--twvar", dest="twvar", type=str, default="None", help="variable name of t in field 'W', if differing between fields.")
    parser.add_argument("--fixZ", dest="fixZ", action="store_true", default=False, help="transform z-Axis to display height, e.g. depth is negative")
    parser.add_argument("--metric", dest="metric", action='store_true', default=False, help="stores if coordinates are resampled to metric metres (True) or not (False; default).")
    parser.add_argument("-LOm", "--lonmin", dest="lonmin", type=float, default=None, help="min. longitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-LOM", "--lonmax", dest="lonmax", type=float, default=None, help="max. longitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-LAm", "--latmin", dest="latmin", type=float, default=None, help="min. latitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-LAM", "--latmax", dest="latmax", type=float, default=None, help="max. latitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-DM", "--depthmax", dest="depthmax", type=float, default=None, help="max. latitude (in arcdegrees or metres) to be plotted - only effective when interpolating")
    parser.add_argument("-TIm", "--timin", dest="timin", type=int, default=None, help="min. time index to plot - only effective when interpolating")
    parser.add_argument("-TIM", "--timax", dest="timax", type=int, default=None, help="max. time index to plot - only effective when interpolating")
    parser.add_argument("-sT", "--scale_T", dest="scale_T", type=str, default='1.0', help="Scaling the values of T(ime) by value - needed e.g. to recalc fields into the 'seconds' base (default: 1.0).")
    parser.add_argument("-ST", "--shift_T", dest="shift_T", type=str, default="0.0", help="shift (in seconds) of the values of T(ime) by value - set a new base (default: 0.0).")
    parser.add_argument("-F", "--format", dest="format", choices=['nc', 'h5'], default='nc', help="type of field files to evaluate, NetCDF (nc) or HDF5 (h5). Default: nc")
    parser.add_argument("--writeNC", dest="writeNC", action='store_true', default=False, help="write output to NetCDF (default: false)")
    parser.add_argument("--writeH5", dest="writeH5", action='store_true', default=False, help="write output to HDF5 (default: false)")
    args = parser.parse_args()

    filedir = args.filedir
    filedir = eval(filedir) if filedir == "None" else filedir
    hydrodir = args.hydrodir
    hydrodir = eval(hydrodir) if hydrodir == "None" else hydrodir
    outdir = args.outdir
    outdir = eval(outdir) if outdir == "None" else outdir
    use_sep_dirs = False
    assert ((filedir is not None) or (hydrodir is not None))
    if filedir is None:
        filedir = hydrodir
        use_sep_dirs = True
    else:
        hydrodir = filedir
    if outdir is None:
        outdir = filedir
    if outdir is None:
        outdir = filedir
    periodicFlag = True
    hasW = False
    is3D = True
    scale_T = float(eval(args.scale_T))
    shift_T = float(eval(args.shift_T))
    metric_resample = args.metric
    netcdf_write = args.writeNC
    hdf5_write = args.writeH5
    save_single_file = False

    fileformat = args.format
    Upattern = args.Upattern
    if '.' in Upattern:
        p_index = str.rfind(Upattern, '.')
        Upattern = Upattern[0:p_index]
    Vpattern = args.Vpattern
    if '.' in Upattern:
        p_index = str.rfind(Vpattern, '.')
        Vpattern = Vpattern[0:p_index]
    Wpattern = args.Wpattern
    if '.' in Wpattern:
        p_index = str.rfind(Wpattern, '.')
        Wpattern = Wpattern[0:p_index]


    xuvar = None
    xvvar = None
    xwvar = None
    yuvar = None
    yvvar = None
    ywvar = None
    zuvar = None
    zvvar = None
    zwvar = None
    tuvar = None
    tvvar = None
    twvar = None

    xvar = args.xvar
    xvar = eval(xvar) if xvar=="None" else xvar
    if xvar is None:
        xuvar = args.xuvar
        xuvar = eval(xuvar) if xuvar == "None" else xuvar
        xvvar = args.xvvar
        xvvar = eval(xvvar) if xvvar == "None" else xvvar
        xwvar = args.xwvar
        xwvar = eval(xwvar) if xwvar == "None" else xwvar
    else:
        xuvar = xvar
        xvvar = xvar
        xwvar = xvar
    yvar = args.yvar
    yvar = eval(yvar) if yvar == "None" else yvar
    if yvar is None:
        yuvar = args.yuvar
        yuvar = eval(yuvar) if yuvar == "None" else yuvar
        yvvar = args.yvvar
        yvvar = eval(yvvar) if yvvar == "None" else yvvar
        ywvar = args.ywvar
        ywvar = eval(ywvar) if ywvar == "None" else ywvar
    else:
        yuvar = yvar
        yvvar = yvar
        ywvar = yvar
    zvar = args.zvar
    zvar = eval(zvar) if zvar == "None" else zvar
    if zvar is None:
        zuvar = args.zuvar
        zuvar = eval(zuvar) if zuvar == "None" else zuvar
        zvvar = args.zvvar
        zvvar = eval(zvvar) if zvvar == "None" else zvvar
        zwvar = args.zwvar
        zwvar = eval(zwvar) if zwvar == "None" else zwvar
    else:
        zuvar = zvar
        zvvar = zvar
        zwvar = zvar
    if (zuvar is None) or (zvvar is None):
        is3D = False
    tvar = args.tvar
    tvar = eval(tvar) if tvar == "None" else tvar
    if tvar is None:
        tuvar = args.tuvar
        tuvar = eval(tuvar) if tuvar == "None" else tuvar
        tvvar = args.tvvar
        tvvar = eval(tvvar) if tvvar == "None" else tvvar
        twvar = args.twvar
        twvar = eval(twvar) if twvar == "None" else twvar
    else:
        tuvar = tvar
        tvvar = tvar
        twvar = tvar
    uvar = args.uvar
    uvar = eval(uvar) if uvar == "None" else uvar
    assert uvar is not None
    vvar = args.vvar
    vvar = eval(vvar) if vvar == "None" else vvar
    assert vvar is not None
    wvar = args.wvar
    wvar = eval(wvar) if wvar == "None" else wvar
    if wvar is None:
        hasW = False
    else:
        hasW = True

    # ==== temporal- and space resampling will be necessary ==== #
    multifile = False
    time_adaptive = False
    grid_adaptive = False
    plain_write = True
    # ==== spatial conversion   ==== #
    equatorial_a_radius = 6378137.0  # in [m]
    polar_b_radius      = 6356752.3 / 2.0  # [m]
    # ==== time conversion data ==== #
    ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    if DBG_MSG:
        print("ns_per_sec = {}".format((ns_per_sec/np.timedelta64(1, 'ns')).astype(np.float64)))
    sec_per_day = 86400.0
    # ==== ==== ==== ==== ==== ==== ==== #
    xyz_type = np.float64
    get_uvw_bounds = False

    # fU_nc = None
    # fV_nc = None
    # fW_nc = None
    fX_nc = None
    fY_nc = None
    fZ_nc = None
    fX_nc_shape, fX_nc_len, fX_nc_min, fX_nc_max = None, None, None, None
    fY_nc_shape, fY_nc_len, fY_nc_min, fY_nc_max = None, None, None, None
    fZ_nc_shape, fZ_nc_len, fZ_nc_min, fZ_nc_max = None, None, None, None
    fT_nc = None
    speed_nc = None
    fU_ext_nc = None
    fV_ext_nc = None
    fW_ext_nc = None
    f_velmag_ext_nc = None
    extents_nc = None
    uvel_fpath_nc = sorted(glob(os.path.join(hydrodir, Upattern + ".nc")))
    vvel_fpath_nc = sorted(glob(os.path.join(hydrodir, Vpattern + ".nc")))
    wvel_fpath_nc = None
    if hasW:
        wvel_fpath_nc = sorted(glob(os.path.join(hydrodir, Wpattern + ".nc")))
    if "nc" in fileformat:
        # if hasW:
        #     assert len(wvel_fpath_nc) == len(uvel_fpath_nc)
        if len(uvel_fpath_nc) > 1 and len(vvel_fpath_nc) > 1:
            multifile = True
        if len(uvel_fpath_nc) > 0 and os.path.exists(uvel_fpath_nc[0]):
            f_u = xr.open_dataset(uvel_fpath_nc[0], decode_cf=False, engine='netcdf4')
            fT_nc = f_u.variables[tuvar].data
            fX_nc = f_u.variables[xuvar]
            fX_nc_min, fX_nc_max, fX_nc_0, fX_nc_dx = get_data_of_ndarray_nc(fX_nc)
            fX_nc = fX_nc.data
            if len(fX_nc.shape) > 1:
                fX_nc = fX_nc.flatten()
            fX_nc_shape, fX_nc_len = fX_nc.shape, fX_nc.shape[0]
            # xyz_type = fX_nc.dtype
            fY_nc = f_u.variables[yuvar]
            fY_nc_min, fY_nc_max, fY_nc_0, fY_nc_dy = get_data_of_ndarray_nc(fY_nc)
            fY_nc = fY_nc.data
            if len(fY_nc.shape) > 1:
                fY_nc = fY_nc.flatten()
            fY_nc_shape, fY_nc_len = fY_nc.shape, fY_nc.shape[0]
            if is3D:
                fZ_nc = f_u.variables[zuvar] if zuvar in f_u.variables.keys() else None
                fZ_nc_min, fZ_nc_max, fZ_nc_0, fZ_nc_dz = get_data_of_ndarray_nc(fZ_nc)
                fZ_nc = fZ_nc.data
                fZ_nc_shape, fZ_nc_len = fZ_nc.shape, fZ_nc.shape[0]
            if fZ_nc is None:
                is3D = False
            extents_nc = (np.nanmin(fX_nc), np.nanmax(fX_nc), np.nanmin(fY_nc), np.nanmax(fY_nc), np.nanmin(fZ_nc), np.nanmax(fZ_nc)) if is3D else (np.nanmin(fX_nc), np.nanmax(fX_nc), np.nanmin(fY_nc), np.nanmax(fY_nc))
            f_u.close()
            del f_u
            if not multifile:
                uvel_fpath_nc = uvel_fpath_nc[0]
        if len(vvel_fpath_nc) > 0 and os.path.exists(vvel_fpath_nc[0]):
            if not multifile:
                vvel_fpath_nc = vvel_fpath_nc[0]
        if hasW and  len(wvel_fpath_nc) > 0 and os.path.exists(wvel_fpath_nc[0]):
            if not multifile:
                wvel_fpath_nc = wvel_fpath_nc[0]

        # print("fX_nc: {}".format(fX_nc))
        # print("fY_nc: {}".format(fY_nc))
        # print("fZ_nc: {}".format(fZ_nc))
        # print("fT_nc: {}".format(fT_nc))
        print("extends XYZ (NetCDF): {}".format(extents_nc))

    # fU_h5 = None
    # fV_h5 = None
    # fW_h5 = None
    fX_h5 = None
    fY_h5 = None
    fZ_h5 = None
    fX_h5_shape, fX_h5_len, fX_h5_min, fX_h5_max = None, None, None, None
    fY_h5_shape, fY_h5_len, fY_h5_min, fY_h5_max = None, None, None, None
    fZ_h5_shape, fZ_h5_len, fZ_h5_min, fZ_h5_max = None, None, None, None
    fT_h5 = None
    speed_h5 = None
    fU_ext_h5 = None
    fV_ext_h5 = None
    fW_ext_h5 = None
    f_velmag_ext_h5 = None
    extents_h5 = None
    uvel_fpath_h5 = sorted(glob(os.path.join(hydrodir, Upattern + ".h5")))
    vvel_fpath_h5 = sorted(glob(os.path.join(hydrodir, Vpattern + ".h5")))
    wvel_fpath_h5 = None
    if hasW:
        wvel_fpath_h5 = sorted(glob(os.path.join(hydrodir, Wpattern + ".h5")))
    grid_fpath_h5 = os.path.join(hydrodir, 'grid.h5')
    if "h5" in fileformat:
        if hasW:
            assert len(wvel_fpath_h5) == len(uvel_fpath_h5)
        if len(uvel_fpath_h5) > 1 and len(vvel_fpath_h5) > 1:
            multifile |= True
        if len(uvel_fpath_h5) > 0 and os.path.exists(uvel_fpath_h5[0]):
            if not multifile:
                uvel_fpath_h5 = uvel_fpath_h5[0]
        if len(vvel_fpath_h5) > 0 and os.path.exists(vvel_fpath_h5[0]):
            if not multifile:
                vvel_fpath_h5 = vvel_fpath_h5[0]
        if hasW and len(wvel_fpath_h5) > 0 and os.path.exists(wvel_fpath_h5[0]):
            if not multifile:
                wvel_fpath_h5 = wvel_fpath_h5[0]
        if os.path.exists(grid_fpath_h5):
            fZ_h5 = None
            f_grid = h5py.File(grid_fpath_h5, "r")
            fX_h5 = f_grid['longitude']
            fX_h5_min, fX_h5_max, fX_h5_0, fX_h5_dx = get_data_of_ndarray_h5(fX_h5)
            fX_h5 = fX_h5[()]
            fX_h5_shape, fX_h5_len = fX_h5.shape, fX_h5.shape[0]
            # xyz_type = fX_h5.dtype
            fY_h5 = f_grid['latitude']
            fY_h5_min, fY_h5_max, fY_h5_0, fY_h5_dy = get_data_of_ndarray_h5(fY_h5)
            fY_h5 = fY_h5[()]
            fY_h5_shape, fY_h5_len = fY_h5.shape, fY_h5.shape[0]
            if is3D:
                fZ_h5 = f_grid['depths'] if 'depths' in f_grid else (f_grid['depth'] if 'depth' in f_grid else None)
                fZ_h5_min, fZ_h5_max, fZ_h5_0, fZ_h5_dz = get_data_of_ndarray_h5(fZ_h5)
                fZ_h5 = fZ_h5[()]
                fZ_h5_shape, fZ_h5_len = fZ_h5.shape, fZ_h5.shape[0]
            if fZ_h5 is None:
                is3D = False
            fT_h5 = f_grid['times'][()]
            extents_h5 = (np.nanmin(fX_h5), np.nanmax(fX_h5), np.nanmin(fY_h5), np.nanmax(fY_h5), np.nanmin(fZ_h5), np.nanmax(fZ_h5)) if is3D else (np.nanmin(fX_h5), np.nanmax(fX_h5), np.nanmin(fY_h5), np.nanmax(fY_h5))
            f_grid.close()
            del f_grid

        # print("fX_h5: {}".format(fX_h5))
        # print("fY_h5: {}".format(fY_h5))
        # print("fZ_h5: {}".format(fZ_h5))
        # print("fT_h5: {}".format(fT_h5))
        print("extends XYZ (NetCDF): {}".format(extents_h5))

    print("multifile: {}".format(multifile))
    print("is3D: {}".format(is3D))
    timebase = None
    dtime_array = None
    fU_shape, fV_shape, fW_shape = None, None, None
    fT = None
    fT_dt = 0
    t0 = None
    fT_fpath_mapping = []  # stores tuples with (<index of file in all files>, <filepath U>, <local index of ti in fT(f_u)>)
    if "nc" in fileformat:
        fXb_ft_nc = []
        fYb_ft_nc = []
        fZb_ft_nc = []
        fT_ft_nc = []
        fU_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        fV_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        fW_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        if multifile:
            print("U-files: {}".format(uvel_fpath_nc))
            i = 0
            for fpath in uvel_fpath_nc:
                f_u = xr.open_dataset(fpath, decode_cf=False, engine='netcdf4')
                xnc = f_u.variables[xuvar]
                xnc_min, xnc_max, xnc_0, xnc_dx = get_data_of_ndarray_nc(xnc)
                # print("xnc_min = {}, xnc_max = {}, xnc_0 = {}, xnc_dx = {}".format(xnc_min, xnc_max, xnc_0, xnc_dx))
                ync = f_u.variables[yuvar]
                ync_min, ync_max, ync_0, ync_dy = get_data_of_ndarray_nc(ync)
                znc = None
                znc_min, znc_max, znc_0, znc_dz = None, None, None, None
                if is3D:
                    znc = f_u.variables[zuvar] if zuvar in f_u.variables.keys() else None
                    znc_min, znc_max, znc_0, znc_dz = get_data_of_ndarray_nc(znc)
                tnc = f_u.variables[tuvar].data
                if len(fXb_ft_nc) > 0 and fXb_ft_nc[0][0] == xnc_min and fXb_ft_nc[0][1] == xnc_max:
                    grid_adaptive &= False
                fXb_ft_nc.append((xnc_min, xnc_max))
                if len(fYb_ft_nc) > 0 and fYb_ft_nc[0][0] == ync_min and fYb_ft_nc[0][1] == ync_max:
                    grid_adaptive &= False
                fYb_ft_nc.append((ync_min, ync_max))
                if is3D and (znc is not None):
                    if len(fZb_ft_nc) > 0 and fZb_ft_nc[0][0] == znc_min and fZb_ft_nc[0][1] == znc_max:
                        grid_adaptive &= False
                    fZb_ft_nc.append((znc_min, znc_max))
                if t0 is None:
                    t0 = tnc[0] if (isinstance(tnc, list) or isinstance(tnc, np.ndarray)) else tnc
                if np.isclose(fT_dt, 0.0):
                    if len(tnc) > 1:
                        ft_0 = convert_timevalue(tnc[0], t0, ns_per_sec, debug=(i == 0))
                        ft_1 = convert_timevalue(tnc[1], t0, ns_per_sec, debug=(i == 0))
                        fT_dt = ft_1 - ft_0
                    elif len(fT_ft_nc) > 0:
                        ft_0 = (convert_timevalue(tnc[0], t0, ns_per_sec, debug=(i == 0)) if (isinstance(tnc, list) or isinstance(tnc, np.ndarray)) else convert_timevalue(tnc, t0, ns_per_sec, debug=(i == 0)))
                        fT_dt = ft_0 - fT_ft_nc[0]

                if isinstance(tnc, list):
                    tnc = np.ndarray(tnc)
                if isinstance(tnc, np.ndarray):
                    for ti in range(tnc.shape[0]):
                        fT_ft_nc.append(convert_timevalue(tnc[ti], t0, ns_per_sec, debug=(i == 0)))
                        fT_fpath_mapping.append((i, fpath, ti))
                else:
                    fT_ft_nc.append(convert_timevalue(tnc, t0, ns_per_sec, debug=(i == 0)))
                    fT_fpath_mapping.append((i, fpath, 0))

                xi_same = True
                for xG in fXb_ft_nc:
                    xi_same &= ((xnc_min, xnc_max) != xG)
                yi_same = True
                for yG in fYb_ft_nc:
                    yi_same &= ((ync_min, ync_max) != yG)
                zi_same = True
                if is3D and znc is not None and len(fZb_ft_nc) > 0:
                    for zG in fZb_ft_nc:
                        zi_same &= ((znc_min, znc_max) != zG)
                if xi_same and yi_same and zi_same:
                    grid_adaptive &= False
                else:
                    grid_adaptive |= True

                fU_shape = f_u.variables[uvar].shape
                if i == 0:
                    print("fU - shape: {}".format(fU_shape))
                if get_uvw_bounds:
                    fU_nc = f_u.variables[uvar].data
                    max_u_value = np.maximum(np.maximum(np.abs(np.nanmin(fU_nc)), np.abs(np.nanmax(fU_nc))), np.maximum(np.abs(fU_ext_nc[0]), np.abs(fU_ext_nc[1])))
                    fU_ext_nc = (-max_u_value, +max_u_value)
                xnc, ync, znc, fU_nc = None, None, None, None
                f_u.close()
                # del xnc
                # del ync
                # del znc
                # del fU_nc
                del f_u
                i += 1
            if get_uvw_bounds:
                print("fU - ext.: {}".format(fU_ext_nc))
            print("V-files: {}".format(vvel_fpath_nc))
            i = 0
            for fpath in vvel_fpath_nc:
                f_v = xr.open_dataset(fpath, decode_cf=False, engine='netcdf4')
                fV_shape = f_v.variables[vvar].shape
                if i == 0:
                    print("fV - shape: {}".format(fV_shape))
                if get_uvw_bounds:
                    fV_nc = f_v.variables[vvar].data
                    max_v_value = np.maximum(np.maximum(np.abs(np.nanmin(fV_nc)), np.abs(np.nanmax(fV_nc))), np.maximum(np.abs(fV_ext_nc[0]), np.abs(fV_ext_nc[1])))
                    fV_ext_nc = (-max_v_value, +max_v_value)
                fV_nc = None
                f_v.close()
                # del fV_nc
                del f_v
                i += 1
            if get_uvw_bounds:
                print("fV - ext.: {}".format(fV_ext_nc))
            if hasW:
                print("W-files: {}".format(wvel_fpath_nc))
                i = 0
                for fpath in wvel_fpath_nc:
                    f_w = xr.open_dataset(fpath, decode_cf=False, engine='netcdf4')
                    fW_shape = f_w.variables[wvar].shape
                    if i == 0:
                        print("fW - shape: {}".format(fW_shape))
                    if get_uvw_bounds:
                        fW_nc = f_w.variables[wvar].data
                        max_w_value = np.maximum(np.maximum(np.abs(np.nanmin(fW_nc)), np.abs(np.nanmax(fW_nc))), np.maximum(np.abs(fW_ext_nc[0]), np.abs(fW_ext_nc[1])))
                        fW_ext_nc = (-max_w_value, +max_w_value)
                    fW_nc = None
                    f_w.close()
                    # del fW_nc
                    del f_w
                    i += 1
                if get_uvw_bounds:
                    print("fW - ext.: {}".format(fW_ext_nc))
            print("|T| = {}, dt = {}, T = {}".format(len(fT_ft_nc), fT_dt, fT_ft_nc))
            # ==== check for dt consistency ==== #
            if len(fT_ft_nc) > 1:
                for ti in range(1, len(fT_ft_nc)):
                    delta_dt = (fT_ft_nc[ti] - fT_ft_nc[ti-1])
                    time_adaptive |= not np.isclose(delta_dt, fT_dt)
            else:
                time_adaptive = False
            # if grid_adaptive:
            #     fX = None
            #     fY = None
            #     fZ = None
            fT = np.array(fT_ft_nc)
        else:
            print("U-file: {}".format(uvel_fpath_nc))
            # ======== u-velocity ======== #
            f_u = xr.open_dataset(uvel_fpath_nc, decode_cf=False, engine='netcdf4')
            xnc = f_u.variables[xuvar]
            xnc_min, xnc_max, xnc_0, xnc_dx = get_data_of_ndarray_nc(xnc)
            fXb_ft_nc.append((xnc_min, xnc_max))
            ync = f_u.variables[yuvar]
            ync_min, ync_max, ync_0, ync_dy = get_data_of_ndarray_nc(ync)
            fYb_ft_nc.append((ync_min, ync_max))
            znc = None
            znc_min, znc_max, znc_0, znc_dz = None, None, None, None
            if is3D:
                znc = f_u.variables[zuvar].data if zuvar in f_u.variables.keys() else None
                znc_min, znc_max, znc_0, znc_dz = get_data_of_ndarray_nc(znc)
                fZb_ft_nc.append((znc_min, znc_max))
            tnc = f_u.variables[tuvar].data
            grid_adaptive = False

            if t0 is None:
                t0 = tnc[0] if (isinstance(tnc, list) or isinstance(tnc, np.ndarray)) else tnc
            if np.isclose(fT_dt, 0.0):
                if len(tnc) > 1:
                    ft_0 = convert_timevalue(tnc[0], t0, ns_per_sec, debug=True)
                    ft_1 = convert_timevalue(tnc[1], t0, ns_per_sec, debug=True)
                    fT_dt = ft_1 - ft_0
                elif len(fT_ft_nc) > 0:
                    ft_0 = (convert_timevalue(tnc[0], t0, ns_per_sec, debug=True) if (isinstance(tnc, list) or isinstance(tnc, np.ndarray)) else convert_timevalue(tnc, t0, ns_per_sec, debug=True))
                    fT_dt = ft_0 - fT_ft_nc[0]

            if isinstance(tnc, np.ndarray):
                # fT_ft_nc = convert_timearray(tnc, fT_dt*60, ns_per_sec, True).tolist()
                fT_ft_nc = tnc.tolist()
            else:
                fT_ft_nc.append(tnc)
            for ti in range(len(fT_ft_nc)):
                fT_fpath_mapping.append((None, uvel_fpath_nc, ti))
            fT = np.array(fT_ft_nc)

            fU_shape = f_u.variables[uvar].shape
            if get_uvw_bounds:
                fU_nc = f_u.variables[uvar].data
                max_u_value = np.maximum(np.abs(fU_nc.min()), np.abs(fU_nc.max()))
                fU_ext_nc = (-max_u_value, +max_u_value)
                del fU_nc
            f_u.close()
            del xnc
            del ync
            del znc
            del f_u
            if get_uvw_bounds:
                print("fU - ext.: {}".format(fU_ext_nc))
            print("V-file: {}".format(vvel_fpath_nc))
            # ======== v-velocity ======== #
            f_v = xr.open_dataset(vvel_fpath_nc, decode_cf=False, engine='netcdf4')
            fV_shape = f_v.variables[vvar].shape
            if get_uvw_bounds:
                fV_nc = f_v.variables[vvar].data
                max_v_value = np.maximum(np.abs(fV_nc.min()), np.abs(fV_nc.max()))
                fV_ext_nc = (-max_v_value, +max_v_value)
                del fV_nc
            f_v.close()
            del f_v
            if get_uvw_bounds:
                print("fV - ext.: {}".format(fV_ext_nc))
            if hasW:
                print("W-file: {}".format(wvel_fpath_nc))
                # ======== w-velocity ======== #
                f_w = xr.open_dataset(wvel_fpath_nc, decode_cf=False, engine='netcdf4')
                fW_shape = f_w.variables[wvar].shape
                if get_uvw_bounds:
                    fW_nc = f_w.variables[wvar].data
                    max_w_value = np.maximum(np.abs(fW_nc.min()), np.abs(fW_nc.max()))
                    fW_ext_nc = (-max_w_value, +max_w_value)
                    del fW_nc
                f_w.close()
                del f_w
                if get_uvw_bounds:
                    print("fW - ext.: {}".format(fW_ext_nc))

        # ======== Time post-processing ======== #
        time_in_min = np.nanmin(fT, axis=0)
        time_in_max = np.nanmax(fT, axis=0)
        if DBG_MSG:
            print("Times:\n\tmin = {}\n\tmax = {}".format(time_in_min, time_in_max))
        # assert fT.shape[1] == time_in_min.shape[0]
        timebase = time_in_max[0] if (isinstance(time_in_max, list) or isinstance(time_in_max, np.ndarray)) else time_in_max
        dtime_array = fT - timebase
        fT = convert_timearray(fT, fT_dt, ns_per_sec, debug=DBG_MSG, array_name="fT")

        del fXb_ft_nc
        del fYb_ft_nc
        del fZb_ft_nc
        del fT_ft_nc
    else:
        fXb_ft_h5 = []
        fYb_ft_h5 = []
        fZb_ft_h5 = []
        fT_ft_h5 = []
        fU_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        fV_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        fW_ext_nc = (+np.finfo(np.float32).eps, -np.finfo(np.float32).eps)
        assert (isinstance(fT_h5, list) or isinstance(fT_h5, np.ndarray))
        fT_h5 = np.array(fT_h5) if isinstance(fT_h5, list) else fT_h5
        if len(fT_h5.shape) > 1:  # then format: N_grid x N_steps
            fT_h5 = np.max(fT_h5, axis=0)
        # ==== ==== consistent global grid file ==== ==== #
        if os.path.exists(grid_fpath_h5):
            grid_adaptive = False
            fXb_ft_h5.append((fX_h5_min, fX_h5_max))
            fYb_ft_h5.append((fY_h5_min, fY_h5_max))
            if is3D:
                fZb_ft_h5.append((fZ_h5_min, fZ_h5_max))

            if t0 is None:
                t0 = fT_h5[0] if (isinstance(fT_h5, list) or isinstance(fT_h5, np.ndarray)) else fT_h5
            if np.isclose(fT_dt, 0.0):
                if len(fT_h5) > 1 or fT_h5.shape[0] > 1:
                    ft_0 = convert_timevalue(fT_h5[0], t0, ns_per_sec, debug=True)
                    ft_1 = convert_timevalue(fT_h5[1], t0, ns_per_sec, debug=True)
                    fT_dt = ft_1 - ft_0
                elif len(fT_ft_h5) > 0:
                    ft_0 = convert_timevalue(fT_h5[0], t0, ns_per_sec, debug=True)
                    fT_dt = ft_0 - fT_ft_h5[0]

            if isinstance(fT_h5, np.ndarray):
                # fT_ft_nc = convert_timearray(tnc, fT_dt*60, ns_per_sec, True).tolist()
                fT_ft_h5 = fT_h5.tolist()
            else:
                fT_ft_h5.append(fT_h5)
            for ti in range(len(fT_ft_h5)):
                fT_fpath_mapping.append((None, uvel_fpath_nc, ti))
            fT = np.array(fT_ft_h5)

            xi_same = True
            for xG in fXb_ft_h5:
                xi_same &= ((fX_h5_min, fX_h5_max) != xG)
            yi_same = True
            for yG in fYb_ft_h5:
                yi_same &= ((fY_h5_min, fY_h5_max) != yG)
            zi_same = True
            if is3D and fZ_h5 is not None and len(fZb_ft_h5) > 0:
                for zG in fZb_ft_h5:
                    zi_same &= ((fZ_h5_min, fZ_h5_max) != zG)
            if xi_same and yi_same and zi_same:
                grid_adaptive &= False
            else:
                grid_adaptive |= True

            # fT = np.array(fT_ft_h5)
        # ==== ==== consistent global grid file ==== ==== #
        if multifile:
            print("U-files: {}".format(uvel_fpath_h5))
            i = 0
            for fpath in uvel_fpath_h5:
                f_u = h5py.File(fpath, "r")
                fU_shape = f_u[uvar].shape
                if get_uvw_bounds:
                    fU_h5 = f_u[uvar][()]
                    max_u_value = np.maximum(np.abs(fU_h5.min()), np.abs(fU_h5.max()))
                    fU_ext_h5 = (-max_u_value, +max_u_value)
                    del fU_h5
                f_u.close()
                del f_u
                # ==== ==== ==== ==== TODO: interpolate time from time 'attribute' ==== ==== ==== ==== #
                # ti = time_index_value(tx, fT_ft_h5, True, _ft_dt=fT_dt)
                # fpath_triplet = fT_fpath_mapping[ti]
                # fT_fpath_mapping[ti] = (i, fpath_triplet[1], fpath_triplet[2])
                fpath_triplet = fT_fpath_mapping[i]
                fT_fpath_mapping[i] = (i, fpath_triplet[1], 0)
                i += 1
            if get_uvw_bounds:
                print("fU - ext.: {}".format(fU_ext_h5))
            print("V-files: {}".format(vvel_fpath_h5))
            i = 0
            for fpath in vvel_fpath_h5:
                f_v = h5py.File(fpath, "r")
                fV_shape = f_v[vvar].shape
                if get_uvw_bounds:
                    fV_h5 = f_v[vvar][()]
                    max_v_value = np.maximum(np.abs(fV_h5.min()), np.abs(fV_h5.max()))
                    fV_ext_h5 = (-max_v_value, +max_v_value)
                    del fV_h5
                f_v.close()
                del f_v
                i += 1
            if get_uvw_bounds:
                print("fV - ext.: {}".format(fV_ext_h5))
            if hasW:
                print("W-files: {}".format(wvel_fpath_h5))
                i = 0
                for fpath in wvel_fpath_h5:
                    f_w = h5py.File(fpath, "r")
                    fW_shape = f_w[wvar].shape
                    if get_uvw_bounds:
                        fW_h5 = f_w[wvar][()]
                        max_w_value = np.maximum(np.abs(fW_h5.min()), np.abs(fW_h5.max()))
                        fW_ext_h5 = (-max_w_value, +max_w_value)
                        del fW_h5
                    f_w.close()
                    del f_w
                    i += 1
                if get_uvw_bounds:
                    print("fW - ext.: {}".format(fW_ext_h5))
            if DBG_MSG:
                print("|T| = {}, dt = {}, T = {}".format(len(fT_ft_h5), fT_dt, fT_ft_h5))
            # ==== check for dt consistency ==== #
            if len(fT_ft_h5) > 1:
                for ti in range(1, len(fT_ft_h5)):
                    delta_dt = (fT_ft_h5[ti] - fT_ft_h5[ti-1])
                    time_adaptive |= not np.isclose(delta_dt, fT_dt)
            else:
                time_adaptive = False
        else:
            print("U-file: {}".format(uvel_fpath_h5))
            f_u = h5py.File(uvel_fpath_h5, "r")
            fU_shape = f_u[uvar].shape
            if get_uvw_bounds:
                fU_h5 = f_u[uvar][()]
                max_u_value = np.maximum(np.abs(fU_h5.min()), np.abs(fU_h5.max()))
                fU_ext_h5 = (-max_u_value, +max_u_value)
                del fU_h5
            f_u.close()
            del f_u
            if get_uvw_bounds:
                print("fU - ext.: {}".format(fU_ext_h5))
            print("V-files: {}".format(vvel_fpath_h5))
            f_v = h5py.File(vvel_fpath_h5, "r")
            fV_shape = f_v[vvar].shape
            if get_uvw_bounds:
                fV_h5 = f_v[vvar][()]
                max_v_value = np.maximum(np.abs(fV_h5.min()), np.abs(fV_h5.max()))
                fV_ext_h5 = (-max_v_value, +max_v_value)
                del fV_h5
            f_v.close()
            del f_v
            if get_uvw_bounds:
                print("fV - ext.: {}".format(fV_ext_h5))
            if hasW:
                print("W-files: {}".format(wvel_fpath_h5))
                f_w = h5py.File(wvel_fpath_h5, "r")
                fW_shape = f_w[wvar].shape
                if get_uvw_bounds:
                    fW_h5 = f_w[wvar][()]
                    max_w_value = np.maximum(np.abs(fW_h5.min()), np.abs(fW_h5.max()))
                    fW_ext_h5 = (-max_w_value, +max_w_value)
                    del fW_h5
                f_w.close()
                del f_w
                if get_uvw_bounds:
                    print("fW - ext.: {}".format(fW_ext_h5))
            if DBG_MSG:
                print("|T| = {}, dt = {}, T = {}".format(len(fT_ft_h5), fT_dt, fT_ft_h5))

        fT = np.array(fT_ft_h5)
        # ======== Time post-processing ======== #
        time_in_min = np.nanmin(fT, axis=0)
        time_in_max = np.nanmax(fT, axis=0)
        if True:
            print("Times:\n\tmin = {}\n\tmax = {}; \n\tfT: {}, \n\tfT_dt: {}, \n\tns_per_sec: {}".format(time_in_min, time_in_max, fT, fT_dt, ns_per_sec))
        # assert fT.shape[1] == time_in_min.shape[0]
        timebase = time_in_max[0] if (isinstance(time_in_max, list) or isinstance(time_in_max, np.ndarray)) else time_in_max
        dtime_array = fT - timebase
        fT = convert_timearray(fT, fT_dt, ns_per_sec, debug=DBG_MSG, array_name="fT")

        del fXb_ft_h5
        del fYb_ft_h5
        del fZb_ft_h5
        del fT_ft_h5

    fX = None
    fY = None
    fZ = None
    fX_shape, fX_len, fX_min, fX_max = None, None, None, None
    fY_shape, fY_len, fY_min, fY_max = None, None, None, None
    fZ_shape, fZ_len, fZ_min, fZ_max = None, None, None, None
    speed = None
    fU_ext = None
    fV_ext = None
    fW_ext = None
    extents = None
    if "nc" in fileformat:
        fX = fX_nc
        fY = fY_nc
        fZ = fZ_nc
        fX_shape, fX_len, fX_min, fX_max = fX_nc_shape, fX_nc_len, fX_nc_min, fX_nc_max
        fY_shape, fY_len, fY_min, fY_max = fY_nc_shape, fY_nc_len, fY_nc_min, fY_nc_max
        fZ_shape, fZ_len, fZ_min, fZ_max = fZ_nc_shape, fZ_nc_len, fZ_nc_min, fZ_nc_max
        fU_ext, fV_ext, fW_ext = fU_ext_nc, fV_ext_nc, fW_ext_nc
        extents = extents_nc
    elif "h5" in fileformat:
        fX = fX_h5
        fY = fY_h5
        fZ = fZ_h5
        fX_shape, fX_len, fX_min, fX_max = fX_h5_shape, fX_h5_len, fX_h5_min, fX_h5_max
        fY_shape, fY_len, fY_min, fY_max = fY_h5_shape, fY_h5_len, fY_h5_min, fY_h5_max
        fZ_shape, fZ_len, fZ_min, fZ_max = fZ_h5_shape, fZ_h5_len, fZ_h5_min, fZ_h5_max
        fU_ext, fV_ext, fW_ext = fU_ext_h5, fV_ext_h5, fW_ext_h5
        extents = extents_h5
    else:
        exit()

    print("multifile: {}".format(multifile))
    print("grid adaptive: {}".format(grid_adaptive))
    print("time adaptive: {}".format(time_adaptive))
    print("fX - shape: {}; |fX|: {}".format(fX_shape, len(fX)))
    print("fY - shape: {}; |fY|: {}".format(fY_shape, len(fY)))
    print("fZ - shape: {}; |fZ|: {}".format(fZ_shape, len(fZ)))
    print("fT - shape: {}; |fT|: {}".format(fT.shape, len(fT)))
    print("fU - shape: {}".format(fU_shape))
    print("fV - shape: {}".format(fV_shape))
    if hasW:
        print("fW - shape: {}".format(fW_shape))
    fX_ext = (fX_min, fX_max)
    fY_ext = (fY_min, fY_max)
    fZ_ext = (fZ_min, fZ_max)
    fT_ext = (fT.min(), fT.max())
    if get_uvw_bounds:
        print("fU ext. - {}".format(fU_ext))
        print("fV ext. - {}".format(fV_ext))
        print("fW ext. - {}".format(fW_ext))
    print("fX ext. (in) - {}".format(fX_ext))
    print("fY ext. (in) - {}".format(fY_ext))
    print("fZ ext. (in) - {}".format(fZ_ext))
    print("fT ext. (in) - {}".format(fT_ext))
    print("fT: {}".format(fT))
    sX = fX_ext[1] - fX_ext[0]
    sY = fY_ext[1] - fY_ext[0]
    sZ = fZ_ext[1] - fZ_ext[0]
    sT = fT_ext[1] - fT_ext[0]

    # ==== scale subsetting ==== #
    loni_min = 0
    loni_max = fX_shape[0]-1
    loni_range = loni_max-loni_min
    lati_min = 0
    lati_max = fY_shape[0]-1
    lati_range = lati_max-lati_min
    depthi_max = fZ_shape[0]-1
    depthi_range = depthi_max-0
    clip = False
    sX = fX_ext[1] - fX_ext[0]
    if args.lonmin is not None or args.lonmax is not None:
        clip = True
        lonmin = fX_min if args.lonmin is None else max(fX_min, args.lonmin)
        loni_min = np.min(np.nonzero(fX >= lonmin))
        if DBG_MSG:
            print("fX_min: {}, args.lonmin: {}, lonmin: {}, loni_min: {}, fx_loni_min: {}".format(fX_min, args.lonmin, lonmin, loni_min, fX[loni_min]))
        lonmax = fX_max if args.lonmax is None else min(fX_max, args.lonmax)
        loni_max = np.max(np.nonzero(fX <= lonmax))
        if DBG_MSG:
            print("fX_max: {}, args.lonmax: {}, lonmax: {}, loni_max: {}, fx_loni_max: {}".format(fX_max, args.lonmax, lonmax, loni_max, fX[loni_max]))
        # if resample_x == 1 or resample_x == 2:
        #     lonmin = (lonmin / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
        #     lonmax = (lonmax / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
        fX_ext = (max(fX_ext[0], fX[loni_min]), min(fX_ext[1], fX[loni_max]))
        loni_range = loni_max-loni_min
        sX = fX_ext[1] - fX_ext[0]
    sY = fY_ext[1] - fY_ext[0]
    if args.latmin is not None or args.latmax is not None:
        clip = True
        latmin = fY_min if args.latmin is None else max(fY_min, args.latmin)
        lati_min = np.min(np.nonzero(fY >= latmin))
        if DBG_MSG:
            print("fY_min: {}, args.latmin: {}, latmin: {}, lati_min: {}, fy_lati_min: {}".format(fY_min, args.latmin, latmin, lati_min, fY[lati_min]))
        latmax = fY_max if args.latmax is None else min(fY_max, args.latmax)
        lati_max = np.max(np.nonzero(fY <= latmax))
        if DBG_MSG:
            print("fY_max: {}, args.latmax: {}, latmax: {}, lati_max: {}, fy_lati_max: {}".format(fY_max, args.latmax, latmax, lati_max, fY[lati_max]))
        # if resample_y == 1 or resample_y == 2:
        #     latmin = (latmin / 90.0) * ((2.0 * np.pi * polar_b_radius) / 2.0)
        #     latmax = (latmax / 90.0) * ((2.0 * np.pi * polar_b_radius) / 2.0)
        fY_ext = (max(fY_ext[0], fY[lati_min]), min(fY_ext[1], fY[lati_max]))
        lati_range = lati_max-lati_min
        sY = fY_ext[1] - fY_ext[0]
    sZ = fZ_ext[1] - fZ_ext[0]
    if args.depthmax is not None and is3D:
        clip = True
        depthi_max = np.max(np.nonzero(fZ <= args.depthmax))
        fZ_ext = (fZ_ext[0], min(fZ_ext[1], fZ[depthi_max]))
        depthi_range = depthi_max-0
        if DBG_MSG:
            print("fZ_max: {}. args.depthmax: {}, fZ_ext: {}, depthi_max: {}, fz_depthi_max: {}".format(fZ_max, args.depthmax, fZ_ext, depthi_max, fZ[depthi_max]))
        sZ = fZ_ext[1] - fZ_ext[0]
    if args.fixZ and is3D:
        fZ_ext = (fZ_ext[1] * -1.0, fZ_ext[0] * -1.0)
        sZ = fZ_ext[1] - fZ_ext[0]
    print("clip: {}".format(clip))
    if clip:
        print("sX (clip) - {}".format(sX))
        print("sY (clip) - {}".format(sY))
        if is3D:
            print("sZ (clip) - {}".format(sZ))
        print("fX ext. (clip) - {}".format(fX_ext))
        print("fY ext. (clip) - {}".format(fY_ext))
        if is3D:
            print("fZ ext. (clip) - {}".format(fZ_ext))
        print("fT ext. (clip) - {}".format(fT_ext))
    print("Indices: \n\tfXi: [{} - {}], |fXi| = {}\n\tfYi: [{} - {}], |fYi| = {}\n\tfZi: [{} - {}], |fZi| = {}".format(loni_min, loni_max, loni_range, lati_min, lati_max, lati_range, 0, depthi_max, depthi_range))

    resample_x = 0
    resample_y = 0
    if metric_resample:
        if (fX_ext[0] >= -180.1) and (fX_ext[1] <= 180.1):
            fX = (fX / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
            fX_ext = (np.nanmin(fX), np.nanmax(fX))
            sX = fX_ext[1] - fX_ext[0]
            resample_x = 1
        elif (fX_ext[0] >= 0.0) and (fX_ext[1] <= 360.1):
            # fX = (fX / 360.0) * (2.0 * np.pi * equatorial_a_radius)
            fX = ((fX-180.0) / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)
            fX_min, fX_max = fX_min-180.0, fX_max-180.0
            fX_ext = (np.nanmin(fX), np.nanmax(fX))
            sX = fX_ext[1] - fX_ext[0]
            resample_x = 2
        if (fY_ext[0] >= -90.1) and (fY_ext[1] <= 90.1):
            fY = (fY / 90.0) * ((2.0 * np.pi  * polar_b_radius) / 2.0)
            fY_ext = (np.nanmin(fY), np.nanmax(fY))
            sY = fY_ext[1] - fY_ext[0]
            resample_y = 1
        elif (fY_ext[0] >= 0.0) and (fY_ext[1] <= 180.1):
            # fY = (fY / 180.0) * (np.pi * polar_b_radius)
            fY = ((fY-90.0) / 90.0) * ((2.0 * np.pi * polar_b_radius) / 2.0)
            fY_min, fY_max = fY_min-90.0, fY_max-90.0
            fY_ext = (np.nanmin(fY), np.nanmax(fY))
            sY = fY_ext[1] - fY_ext[0]
            resample_y = 2
    fX_dx_1 = fX[1:]
    fX_dx_0 = fX[0:-1]
    fX_dx = abs(fX_dx_1-fX_dx_0)
    fx_dx_valid = np.nonzero(fX_dx)[0]
    fX_dx_min = np.nanmin(fX_dx[fx_dx_valid])
    # fX_dx_mid = np.nanmedian(fX_dx[fx_dx_valid])
    fX_dx_mid = np.nanmean(fX_dx[fx_dx_valid])
    fX_dx_max = np.nanmax(fX_dx[fx_dx_valid])
    del fX_dx_0
    del fX_dx_1
    # # del fX_dx
    fY_dy_1 = fY[1:]
    fY_dy_0 = fY[0:-1]
    fY_dy = abs(fY_dy_1-fY_dy_0)
    fy_dy_valid = np.nonzero(fY_dy)[0]
    fY_dy_min = np.nanmin(fY_dy[fy_dy_valid])
    # fY_dy_mid = np.nanmedian(fY_dy[fy_dy_valid])
    fY_dy_mid = np.nanmean(fY_dy[fy_dy_valid])
    fY_dy_max = np.nanmax(fY_dy[fy_dy_valid])
    del fY_dy_0
    del fY_dy_1
    # # del fY_dy
    fZ_dz_1 = fZ[1:]
    fZ_dz_0 = fZ[0:-1]
    fZ_dz = abs(fZ_dz_1-fZ_dz_0)
    fz_dz_valid = np.nonzero(fZ_dz)[0]
    fZ_dz_min = np.nanmin(fZ_dz[fz_dz_valid])
    # fZ_dz_mid = np.nanmedian(fZ_dz[fz_dz_valid])
    fZ_dz_mid = np.nanmean(fZ_dz[fz_dz_valid])
    fZ_dz_max = np.nanmax(fZ_dz[fz_dz_valid])
    del fZ_dz_0
    del fZ_dz_1
    # # del fZ_dz
    lateral_gres = min(fX_dx_mid, fY_dy_mid)
    # lateral_gres = max(lateral_gres, (0.25 / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0)) if resample_x > 0 or resample_y > 0 else lateral_gres
    # # lateral_gres = max(min(fX_dx_mid, fY_dy_mid), (0.25 / 180.0) * ((2.0 * np.pi * equatorial_a_radius) / 2.0))
    vertical_gres = fZ_dz_mid

    # ======= fT - scale-shift ======= #
    fT = (fT * scale_T) + shift_T
    fT_dt = fT_dt * scale_T
    fT_ext = ((fT_ext[0] * scale_T) - shift_T, (fT_ext[1] * scale_T) + shift_T)
    sT = fT_ext[1] - fT_ext[0]
    # ==== end - fT - scale-shift ==== #

    # del fX
    # del fY
    # del fZ
    print("sX - {}".format(sX))
    print("sY - {}".format(sY))
    print("sZ - {}".format(sZ))
    print("sT - {}".format(sT))
    print("fX ext. (out) - {}".format(fX_ext))
    print("fY ext. (out) - {}".format(fY_ext))
    print("fZ ext. (out) - {}".format(fZ_ext))
    print("fT ext. (out) - {}".format(fT_ext))
    print("fX_dx - min: {}; max {}".format(fX_dx_min, fX_dx_max))
    print("fY_dy - min: {}; max {}".format(fY_dy_min, fY_dy_max))
    print("fZ_dz - min: {}; max {}".format(fZ_dz_min, fZ_dz_max))
    print("lateral gres: {}; vertical gres: {}".format(lateral_gres, vertical_gres))
    dt = fT[1] - fT[0]
    print("dT: {}; fT_dt: {}".format(dt, fT_dt))
    gc.collect()

    # ==== time interpolation ==== #
    ti_min = 0
    ti_max = fT.shape[0]-1
    # idt = math.copysign(1.0 * 86400.0, fT_dt)
    idt = fT_dt
    iT = fT
    cap_min = fT[ti_min]
    cap_max = fT[ti_max]
    iT_max = np.max(fT)
    iT_min = np.min(fT)
    # calcsteps = (fT_ext[1]-fT_ext[0])/fT_dt
    # tsteps = int(math.floor(calcsteps))
    tsteps = (ti_max+1)-ti_min  # here, we can use the full range of timestamps because our calculation always calculates step-by-step (i.e. no linear interpolation involved)
    print("ti_min: {}; ti_max: {}; idt: {}; iT: {}; cap_min: {}; cap_max: {}; iT_min: {}; iT_max: {}; tsteps: {}".format(ti_min, ti_max, idt, iT, cap_min, cap_max, iT_min, iT_max, tsteps))

    if hdf5_write:
        grid_file = h5py.File(os.path.join(outdir, "grid.h5"), "w")
        grid_lon_ds = grid_file.create_dataset("longitude",
                                               data=fX[loni_min:loni_max],
                                               compression="gzip",
                                               compression_opts=4)
        grid_lon_ds.attrs['unit'] = "arc degree"
        grid_lon_ds.attrs['name'] = 'longitude'
        grid_lon_ds.attrs['min'] = np.nanmin(fX[loni_min:loni_max])
        grid_lon_ds.attrs['max'] = np.nanmax(fX[loni_min:loni_max])
        grid_lat_ds = grid_file.create_dataset("latitude",
                                               data=fY[lati_min:lati_max],
                                               compression="gzip",
                                               compression_opts=4)
        grid_lat_ds.attrs['unit'] = "arc degree"
        grid_lat_ds.attrs['name'] = 'latitude'
        grid_lat_ds.attrs['min'] = np.nanmin(fY[lati_min:lati_max])
        grid_lat_ds.attrs['max'] = np.nanmax(fY[lati_min:lati_max])
        grid_lat_ds = grid_file.create_dataset("depth",
                                               data=fZ[0:depthi_range],
                                               compression="gzip",
                                               compression_opts=4)
        grid_lat_ds.attrs['unit'] = "metres"
        grid_lat_ds.attrs['name'] = 'depth'
        grid_lat_ds.attrs['min'] = np.nanmin(fZ[0:depthi_range])
        grid_lat_ds.attrs['max'] = np.nanmax(fZ[0:depthi_range])
        grid_time_ds = grid_file.create_dataset("times", data=iT,
                                                compression="gzip",
                                                compression_opts=4)
        grid_time_ds.attrs['unit'] = "seconds"
        grid_time_ds.attrs['name'] = 'time'
        grid_time_ds.attrs['min'] = np.nanmin(iT)
        grid_time_ds.attrs['max'] = np.nanmax(iT)
        grid_file.close()

    us_minmax = [0., 0.]
    us_statistics = [0., 0.]
    us_file, us_file_ds = None, None
    us_nc_file, us_nc_tdim, us_nc_zdim, us_nc_ydim, us_nc_xdim, us_nc_uvel = None, None, None, None, None, None
    us_nc_tvar, us_nc_zvar, us_nc_yvar, us_nc_xvar = None, None, None, None

    if save_single_file:
        if hdf5_write:
            us_file = h5py.File(os.path.join(outdir, "hydrodynamic_U.h5"), "w")
            us_file_ds = us_file.create_dataset("uo",
                                                shape=(1, depthi_range, lati_range, loni_range),
                                                maxshape=(iT.shape[0], depthi_range, lati_range, loni_range),
                                                dtype=np.float32,
                                                compression="gzip", compression_opts=4)
            us_file_ds.attrs['unit'] = "m/s"
            us_file_ds.attrs['name'] = 'meridional_velocity'
        if netcdf_write:
            us_nc_file = Dataset(os.path.join(outdir, "hydrodynamic_U.nc"), mode='w', format='NETCDF4_CLASSIC')
            us_nc_xdim = us_nc_file.createDimension('lon', loni_range)
            us_nc_ydim = us_nc_file.createDimension('lat', lati_range)
            us_nc_zdim = us_nc_file.createDimension('depth', depthi_range)
            us_nc_tdim = us_nc_file.createDimension('time', None)
            us_nc_xvar = us_nc_file.createVariable('lon', np.float32, ('lon', ))
            us_nc_yvar = us_nc_file.createVariable('lat', np.float32, ('lat', ))
            us_nc_zvar = us_nc_file.createVariable('depth', np.float32, ('depth', ))
            us_nc_tvar = us_nc_file.createVariable('time', np.float32, ('time', ))
            us_nc_file.title = "hydrodynamic-3D-U"
            us_nc_file.subtitle = "365d-daily"
            us_nc_xvar.units = "arcdegree_eastwards"
            us_nc_xvar.long_name = "longitude"
            us_nc_yvar.units = "arcdegree_northwards"
            us_nc_yvar.long_name = "latitude"
            us_nc_zvar.units = "metres_down"
            us_nc_zvar.long_name = "depth"
            us_nc_tvar.units = "seconds"
            us_nc_tvar.long_name = "time"
            us_nc_xvar[:] = fX[loni_min:loni_max]
            us_nc_yvar[:] = fY[lati_min:lati_max]
            us_nc_zvar[:] = fZ[0:depthi_range]
            if args.fix3D:
                # still TODO
                pass
            us_nc_uvel = us_nc_file.createVariable('uo', np.float32, ('time', 'depth', 'lat', 'lon'))
            us_nc_uvel.units = "m/s"
            us_nc_uvel.standard_name = "eastwards longitudinal zonal velocity"

    vs_minmax = [0., 0.]
    vs_statistics = [0., 0.]
    vs_file, vs_file_ds = None, None
    vs_nc_file, vs_nc_tdim, vs_nc_zdim, vs_nc_ydim, vs_nc_xdim, vs_nc_vvel = None, None, None, None, None, None
    vs_nc_tvar, vs_nc_zvar, vs_nc_yvar, vs_nc_xvar = None, None, None, None
    if save_single_file:
        if hdf5_write:
            vs_file = h5py.File(os.path.join(outdir, "hydrodynamic_V.h5"), "w")
            vs_file_ds = vs_file.create_dataset("vo",
                                                shape=(1, depthi_range, lati_range, loni_range),
                                                maxshape=(iT.shape[0], depthi_range, lati_range, loni_range),
                                                dtype=np.float32,
                                                compression="gzip", compression_opts=4)
            vs_file_ds.attrs['unit'] = "m/s"
            vs_file_ds.attrs['name'] = 'zonal_velocity'
        if netcdf_write:
            vs_nc_file = Dataset(os.path.join(outdir, "hydrodynamic_V.nc"), mode='w', format='NETCDF4_CLASSIC')
            vs_nc_xdim = vs_nc_file.createDimension('lon', loni_range)
            vs_nc_ydim = vs_nc_file.createDimension('lat', lati_range)
            vs_nc_zdim = vs_nc_file.createDimension('depth', depthi_range)
            vs_nc_tdim = vs_nc_file.createDimension('time', None)
            vs_nc_xvar = vs_nc_file.createVariable('lon', np.float32, ('lon', ))
            vs_nc_yvar = vs_nc_file.createVariable('lat', np.float32, ('lat', ))
            vs_nc_zvar = vs_nc_file.createVariable('depth', np.float32, ('depth', ))
            vs_nc_tvar = vs_nc_file.createVariable('time', np.float32, ('time', ))
            vs_nc_file.title = "hydrodynamic-3D-V"
            vs_nc_file.subtitle = "365d-daily"
            vs_nc_xvar.units = "arcdegree_eastwards"
            vs_nc_xvar.long_name = "longitude"
            vs_nc_yvar.units = "arcdegree_northwards"
            vs_nc_yvar.long_name = "latitude"
            vs_nc_zvar.units = "metres_down"
            vs_nc_zvar.long_name = "depth"
            vs_nc_tvar.units = "seconds"
            vs_nc_tvar.long_name = "time"
            vs_nc_xvar[:] = fX[loni_min:loni_max]
            vs_nc_yvar[:] = fY[lati_min:lati_max]
            vs_nc_zvar[:] = fZ[0:depthi_range]
            if args.fix3D:
                # still TODO
                pass
            vs_nc_vvel = vs_nc_file.createVariable('vo', np.float32, ('time', 'depth', 'lat', 'lon'))
            vs_nc_vvel.units = "m/s"
            vs_nc_vvel.standard_name = "northwards latitudinal meridional velocity"

    if hasW:
        ws_minmax = [0., 0.]
        ws_statistics = [0., 0.]
        ws_file, ws_file_ds = None, None
        ws_nc_file, ws_nc_tdim, ws_nc_zdim, ws_nc_ydim, ws_nc_xdim, ws_nc_vvel = None, None, None, None, None, None
        ws_nc_tvar, ws_nc_zvar, ws_nc_yvar, ws_nc_xvar = None, None, None, None
        if save_single_file:
            if hdf5_write:
                ws_file = h5py.File(os.path.join(outdir, "hydrodynamic_W.h5"), "w")
                ws_file_ds = ws_file.create_dataset("wo",
                                                    shape=(1, depthi_range, lati_range, loni_range),
                                                    maxshape=(iT.shape[0], depthi_range, lati_range, loni_range),
                                                    dtype=np.float32,
                                                    compression="gzip", compression_opts=4)
                ws_file_ds.attrs['unit'] = "m/s"
                ws_file_ds.attrs['name'] = 'vertical_velocity'
            if netcdf_write:
                ws_nc_file = Dataset(os.path.join(outdir, "hydrodynamic_W.nc"), mode='w', format='NETCDF4_CLASSIC')
                ws_nc_xdim = ws_file_ds.createDimension('lon', loni_range)
                ws_nc_ydim = ws_file_ds.createDimension('lat', lati_range)
                ws_nc_zdim = ws_file_ds.createDimension('depth', depthi_range)
                ws_nc_tdim = ws_file_ds.createDimension('time', None)
                ws_nc_xvar = ws_file_ds.createVariable('lon', np.float32, ('lon', ))
                ws_nc_yvar = ws_file_ds.createVariable('lat', np.float32, ('lat', ))
                ws_nc_zvar = ws_file_ds.createVariable('depth', np.float32, ('depth', ))
                ws_nc_tvar = ws_file_ds.createVariable('time', np.float32, ('time', ))
                ws_file_ds.title = "hydrodynamic-3D-W"
                ws_file_ds.subtitle = "365d-daily"
                ws_nc_xvar.units = "arcdegree_eastwards"
                ws_nc_xvar.long_name = "longitude"
                ws_nc_yvar.units = "arcdegree_northwards"
                ws_nc_yvar.long_name = "latitude"
                ws_nc_zvar.units = "metres_down"
                ws_nc_zvar.long_name = "depth"
                ws_nc_tvar.units = "seconds"
                ws_nc_tvar.long_name = "time"
                ws_nc_xvar[:] = fX[loni_min:loni_max]
                ws_nc_yvar[:] = fY[lati_min:lati_max]
                ws_nc_zvar[:] = fZ[0:depthi_range]
                if args.fix3D:
                    # still TODO
                    pass
                ws_nc_wvel = ws_nc_file.createVariable('wo', np.float32, ('time', 'depth', 'lat', 'lon'))
                ws_nc_wvel.units = "m/s"
                ws_nc_wvel.standard_name = "downwards depth vertical velocity"

    print("Sampling 3D velocity data ...")
    idx_x, idx_y, idx_z = np.meshgrid(range(loni_min,loni_max), range(lati_min,lati_max),range(0, depthi_max), sparse=False,
                                      indexing='ij', copy=False)
    idx_x = idx_x.flatten()
    idx_y = idx_y.flatten()
    idx_z = idx_z.flatten()

    items = np.column_stack((idx_x, idx_y, idx_z))

    # items = zip(idx_x, idx_y, idx_z)
    # items = []
    # for _indices in range(0, idx_x.shape[0]):
    #     items.append((idx_x[_indices], idx_y[_indices], idx_z[_indices]))
    # print("Indices: {}\n".format(items[0].shape[0], items[1].shape[0], items[2].shape[0]))

    print("Indices: {}\n".format(items))

    # exit()

    # u = np.zeros((depthi_range, lati_range, loni_range))
    # v = np.zeros((depthi_range, lati_range, loni_range))
    # v = np.zeros((depthi_range, lati_range, loni_range))
    seconds_per_day = 24.0*60.0*60.0
    total_items = (ti_max + 1) - ti_min
    total_values = total_items * loni_range * lati_range * depthi_range
    current_item = 0
    current_values = 0
    for ti in range(ti_min, ti_max + 1):
        tx0 = iT_min + float(ti) * idt
        uvw_ti0 = ti
        fpath_idx_ti0 = fT_fpath_mapping[uvw_ti0][0]
        local_ti0 = fT_fpath_mapping[uvw_ti0][2]
        if DBG_MSG:
            print("path ti0: {} (local index: {})".format(fpath_idx_ti0, local_ti0))
        uvel_fpath_ti0 = None
        vvel_fpath_ti0 = None
        wvel_fpath_ti0 = None
        # tp_fpath_ti0 = None
        # na_fpath_ti0 = None
        if multifile:
            uvel_fpath_ti0 = uvel_fpath_nc[fpath_idx_ti0]
            vvel_fpath_ti0 = vvel_fpath_nc[fpath_idx_ti0]
            if hasW:
                wvel_fpath_ti0 = wvel_fpath_nc[fpath_idx_ti0]
        else:
            uvel_fpath_ti0 = uvel_fpath_nc
            vvel_fpath_ti0 = vvel_fpath_nc
            if hasW:
                wvel_fpath_ti0 = wvel_fpath_nc
        if DBG_MSG:
            print("ti0 - file index: {}, filepath: {}, local ti-index: {}".format(fpath_idx_ti0, uvel_fpath_ti0, local_ti0))
        if DBG_MSG:
            print("Loading files timestep ti0")
        # ---- load ti0 ---- #
        # if DBG_MSG:
        #     print("Loaded XYZ data (ti=0).")
        f_u_0 = xr.open_dataset(uvel_fpath_ti0, decode_cf=True, engine='netcdf4')
        fU0 = f_u_0.variables[uvar]
        fU = np.array(fU0[local_ti0]).squeeze()
        f_v_0 = xr.open_dataset(vvel_fpath_ti0, decode_cf=True, engine='netcdf4')
        fV0 = f_v_0.variables[vvar]
        fV = np.array(fV0[local_ti0]).squeeze()
        fW0 = None
        if hasW:
            f_w_0 = xr.open_dataset(wvel_fpath_ti0, decode_cf=True, engine='netcdf4')
            fW0 = f_w_0.variables[wvar]
            fW = np.array(fW0[local_ti0]).squeeze()

        if not save_single_file:
            us_minmax = [0., 0.]
            us_statistics = [0., 0.]
            if hdf5_write:
                u_filename = "hydrodynamic_U_d%03d.h5" % (ti, )
                us_file = h5py.File(os.path.join(outdir, u_filename), "w")
                us_file_ds = us_file.create_dataset("uo",
                                                    shape=(1, depthi_range, lati_range, loni_range),
                                                    maxshape=(iT.shape[0], depthi_range, lati_range, loni_range),
                                                    dtype=np.float32,
                                                    compression="gzip", compression_opts=4)
                us_file_ds.attrs['unit'] = "m/s"
                us_file_ds.attrs['time'] = tx0
                us_file_ds.attrs['time_unit'] = "s"
                us_file_ds.attrs['name'] = 'meridional_velocity'
            if netcdf_write:
                u_filename = "hydrodynamic_U_d%03d.nc" % (ti, )
                us_nc_file = Dataset(os.path.join(outdir, u_filename), mode='w', format='NETCDF4_CLASSIC')
                us_nc_xdim = us_nc_file.createDimension('lon', loni_range)
                us_nc_ydim = us_nc_file.createDimension('lat', lati_range)
                us_nc_zdim = us_nc_file.createDimension('depth', depthi_range)
                us_nc_tdim = us_nc_file.createDimension('time', 1)
                us_nc_xvar = us_nc_file.createVariable('lon', np.float32, ('lon',))
                us_nc_yvar = us_nc_file.createVariable('lat', np.float32, ('lat',))
                us_nc_zvar = us_nc_file.createVariable('depth', np.float32, ('depth',))
                us_nc_tvar = us_nc_file.createVariable('time', np.float32, ('time',))
                us_nc_file.title = "hydrodynamic-3D-U"
                us_nc_file.subtitle = "365d-daily"
                us_nc_xvar.units = "arcdegree_eastwards"
                us_nc_xvar.long_name = "longitude"
                us_nc_yvar.units = "arcdegree_northwards"
                us_nc_yvar.long_name = "latitude"
                us_nc_zvar.units = "metres_down"
                us_nc_zvar.long_name = "depth"
                us_nc_tvar.units = "seconds"
                us_nc_tvar.long_name = "time"
                us_nc_xvar[:] = fX[loni_min:loni_max]
                us_nc_yvar[:] = fY[lati_min:lati_max]
                us_nc_zvar[:] = fZ[0:depthi_range]
                us_nc_uvel = us_nc_file.createVariable('uo', np.float32, ('time', 'depth', 'lat', 'lon'))
                us_nc_uvel.units = "m/s"
                us_nc_uvel.standard_name = "eastwards longitudinal zonal velocity"

            vs_minmax = [0., 0.]
            vs_statistics = [0., 0.]
            if hdf5_write:
                v_filename = "hydrodynamic_V_d%03d.h5" %(ti, )
                vs_file = h5py.File(os.path.join(outdir, v_filename), "w")
                vs_file_ds = vs_file.create_dataset("vo",
                                                    shape=(1, depthi_range, lati_range, loni_range),
                                                    maxshape=(iT.shape[0], depthi_range, lati_range, loni_range),
                                                    dtype=np.float32,
                                                    compression="gzip", compression_opts=4)
                vs_file_ds.attrs['unit'] = "m/s"
                vs_file_ds.attrs['time'] = tx0
                vs_file_ds.attrs['time_unit'] = "s"
                vs_file_ds.attrs['name'] = 'zonal_velocity'
            if netcdf_write:
                v_filename = "hydrodynamic_V_d%03d.nc" % (ti, )
                vs_nc_file = Dataset(os.path.join(outdir, v_filename), mode='w', format='NETCDF4_CLASSIC')
                vs_nc_xdim = vs_nc_file.createDimension('lon', loni_range)
                vs_nc_ydim = vs_nc_file.createDimension('lat', lati_range)
                vs_nc_zdim = vs_nc_file.createDimension('depth', depthi_range)
                vs_nc_tdim = vs_nc_file.createDimension('time', 1)
                vs_nc_xvar = vs_nc_file.createVariable('lon', np.float32, ('lon', ))
                vs_nc_yvar = vs_nc_file.createVariable('lat', np.float32, ('lat', ))
                vs_nc_zvar = vs_nc_file.createVariable('depth', np.float32, ('depth', ))
                vs_nc_tvar = vs_nc_file.createVariable('time', np.float32, ('time', ))
                vs_nc_file.title = "hydrodynamic-3D-V"
                vs_nc_file.subtitle = "365d-daily"
                vs_nc_xvar.units = "arcdegree_eastwards"
                vs_nc_xvar.long_name = "longitude"
                vs_nc_yvar.units = "arcdegree_northwards"
                vs_nc_yvar.long_name = "latitude"
                vs_nc_zvar.units = "metres_down"
                vs_nc_zvar.long_name = "depth"
                vs_nc_tvar.units = "seconds"
                vs_nc_tvar.long_name = "time"
                vs_nc_xvar[:] = fX[loni_min:loni_max]
                vs_nc_yvar[:] = fY[lati_min:lati_max]
                vs_nc_zvar[:] = fZ[0:depthi_range]
                vs_nc_vvel = vs_nc_file.createVariable('vo', np.float32, ('time', 'depth', 'lat', 'lon'))
                vs_nc_vvel.units = "m/s"
                vs_nc_vvel.standard_name = "northwards latitudinal meridional velocity"

            if hasW:
                ws_minmax = [0., 0.]
                ws_statistics = [0., 0.]
                if hdf5_write:
                    w_filename = "hydrodynamic_W_d%03d.h5" %(ti, )
                    ws_file = h5py.File(os.path.join(outdir, w_filename), "w")
                    ws_file_ds = vs_file.create_dataset("wo",
                                                        shape=(1, depthi_range, lati_range, loni_range),
                                                        maxshape=(iT.shape[0], depthi_range, lati_range, loni_range),
                                                        dtype=np.float32,
                                                        compression="gzip", compression_opts=4)
                    ws_file_ds.attrs['unit'] = "m/s"
                    ws_file_ds.attrs['time'] = tx0
                    ws_file_ds.attrs['time_unit'] = "s"
                    ws_file_ds.attrs['name'] = 'vertical_velocity'
                if netcdf_write:
                    w_filename = "hydrodynamic_W_d%03d.nc" % (ti, )
                    ws_nc_file = Dataset(os.path.join(outdir, w_filename), mode='w', format='NETCDF4_CLASSIC')
                    ws_nc_xdim = ws_nc_file.createDimension('lon', loni_range)
                    ws_nc_ydim = ws_nc_file.createDimension('lat', lati_range)
                    ws_nc_zdim = ws_nc_file.createDimension('depth', depthi_range)
                    ws_nc_tdim = ws_nc_file.createDimension('time', 1)
                    ws_nc_xvar = ws_nc_file.createVariable('lon', np.float32, ('lon', ))
                    ws_nc_yvar = ws_nc_file.createVariable('lat', np.float32, ('lat', ))
                    ws_nc_zvar = ws_nc_file.createVariable('depth', np.float32, ('depth', ))
                    ws_nc_tvar = ws_nc_file.createVariable('time', np.float32, ('time', ))
                    ws_nc_file.title = "hydrodynamic-3D-V"
                    ws_nc_file.subtitle = "365d-daily"
                    ws_nc_xvar.units = "arcdegree_eastwards"
                    ws_nc_xvar.long_name = "longitude"
                    ws_nc_yvar.units = "arcdegree_northwards"
                    ws_nc_yvar.long_name = "latitude"
                    ws_nc_zvar.units = "metres_down"
                    ws_nc_zvar.long_name = "depth"
                    ws_nc_tvar.units = "seconds"
                    ws_nc_tvar.long_name = "time"
                    ws_nc_xvar[:] = fX[loni_min:loni_max]
                    ws_nc_yvar[:] = fY[lati_min:lati_max]
                    ws_nc_zvar[:] = fZ[0:depthi_range]
                    ws_nc_wvel = ws_nc_file.createVariable('wo', np.float32, ('time', 'depth', 'lat', 'lon'))
                    ws_nc_wvel.units = "m/s"
                    ws_nc_wvel.standard_name = "downwards depth vertical velocity"
            # ==== === files created. === ==== #

            uo = fU[0:depthi_max, lati_min:lati_max, loni_min:loni_max]
            us_minmax = [min(us_minmax[0], np.min(uo)), max(us_minmax[1], np.max(uo))]
            us_statistics[0] += uo.mean()
            us_statistics[1] += uo.std()
            vo = fV[0:depthi_max, lati_min:lati_max, loni_min:loni_max]
            vs_minmax = [min(vs_minmax[0], np.min(vo)), max(vs_minmax[1], np.max(vo))]
            vs_statistics[0] += vo.mean()
            vs_statistics[1] += vo.std()
            if hasW:
                wo = fW[0:depthi_max, lati_min:lati_max, loni_min:loni_max]
                ws_minmax = [min(ws_minmax[0], np.min(wo)), max(ws_minmax[1], np.max(wo))]
                ws_statistics[0] += wo.mean()
                ws_statistics[1] += wo.std()

            if save_single_file:
                if hdf5_write:
                    us_file_ds.resize((ti+1), axis=0)
                    us_file_ds[ti, :, :, :] = uo
                    vs_file_ds.resize((ti+1), axis=0)
                    vs_file_ds[ti, :, :, :] = vo
                    if hasW:
                        ws_file_ds.resize((ti+1), axis=0)
                        ws_file_ds[ti, :, :, :] = wo
                if netcdf_write:
                    # salt_nc_value[ti, :, :, :] = salt
                    us_nc_uvel[ti, :, :, :] = uo
                    vs_nc_vvel[ti, :, :, :] = vo
                    if hasW:
                        ws_nc_wvel[ti, :, :, :] = wo
            else:
                if hdf5_write:
                    us_file_ds[0, :, :, :] = uo
                    vs_file_ds[0, :, :, :] = vo
                    if hasW:
                        ws_file_ds[0, :, :, :] = wo
                if netcdf_write:
                    us_nc_uvel[0, :, :, :] = uo
                    vs_nc_vvel[0, :, :, :] = vo
                    if hasW:
                        ws_nc_wvel[0, :, :, :] = wo
            if not save_single_file:
                if hdf5_write:
                    us_file_ds.attrs['min'] = us_minmax[0]
                    us_file_ds.attrs['max'] = us_minmax[1]
                    us_file_ds.attrs['mean'] = us_statistics[0]
                    us_file_ds.attrs['std'] = us_statistics[1]
                    us_file.close()
                    vs_file_ds.attrs['min'] = vs_minmax[0]
                    vs_file_ds.attrs['max'] = vs_minmax[1]
                    vs_file_ds.attrs['mean'] = vs_statistics[0]
                    vs_file_ds.attrs['std'] = vs_statistics[1]
                    vs_file.close()
                    if hasW:
                        ws_file_ds.attrs['min'] = ws_minmax[0]
                        ws_file_ds.attrs['max'] = ws_minmax[1]
                        ws_file_ds.attrs['mean'] = ws_statistics[0]
                        ws_file_ds.attrs['std'] = ws_statistics[1]
                        ws_file.close()
                if netcdf_write:
                    us_nc_tvar[0] = tx0
                    us_nc_file.close()
                    vs_nc_tvar[0] = tx0
                    vs_nc_file.close()
                    if hasW:
                        ws_nc_tvar[0] = tx0
                        ws_nc_file.close()

        fU0 = None
        fU = None
        fV0 = None
        fV = None
        if hasW:
            fW0 = None
            fW = None
        f_u_0.close()
        f_v_0.close()
        if hasW:
            f_w_0.close()
        if DBG_MSG:
            print("Finished timestep {} of {}.".format(ti, ti_max))
        current_item = ti
        workdone = current_item / total_items
        print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
        # current_item = ti
        # workdone = current_item / total_items
        # print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)
    print("\nFinished UV(W) sampling.")

    if save_single_file:
        if hdf5_write:
            us_file_ds.attrs['min'] = us_minmax[0]
            us_file_ds.attrs['max'] = us_minmax[1]
            us_file_ds.attrs['mean'] = us_statistics[0]
            us_file_ds.attrs['std'] = us_statistics[1]
            us_file.close()
            vs_file_ds.attrs['min'] = vs_minmax[0]
            vs_file_ds.attrs['max'] = vs_minmax[1]
            vs_file_ds.attrs['mean'] = vs_statistics[0]
            vs_file_ds.attrs['std'] = vs_statistics[1]
            vs_file.close()
            if hasW:
                ws_file_ds.attrs['min'] = ws_minmax[0]
                ws_file_ds.attrs['max'] = ws_minmax[1]
                ws_file_ds.attrs['mean'] = ws_statistics[0]
                ws_file_ds.attrs['std'] = ws_statistics[1]
                ws_file.close()
        if netcdf_write:
            us_nc_tvar[:] = iT[0]
            vs_nc_tvar[:] = iT[0]
            if hasW:
                ws_nc_tvar[:] = iT[0]
            us_nc_file.close()
            vs_nc_file.close()
            if hasW:
                ws_nc_file.close()

