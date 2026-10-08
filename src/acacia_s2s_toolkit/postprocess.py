# SPDX-FileCopyrightText: 2024 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os,sys,glob
import xesmf as xe
import pooch

#this is for readthedocs to work correctly. These packages cannot be installed by pip, so one has to mock import them
autodoc_mock_imports = ["esmpy", "xesmf", "netCDF4"]

#this determines with functions will be publicly visible in the installed package
__all__=["postprocess_forecast",
         "get_example_data",
         "load_raw_data", 
         "read_netcdf", 
         "add_enslag",
         "deaccumulate",
         "align_grid",
         "set_verbose",
         "align_time",
         "organize_by_leadtime",
         "_log",
         "aggregate",
         "leadtime_to_time",
         "organize_by_validtime"
        ]

#######################################################################################

#
# helper functions
#
#######################################################################################

VERBOSE = True

def set_verbose(v):
    global VERBOSE
    VERBOSE = v
    print("verbosity is {}".format(v))


def _log(msg, force=False):
    if VERBOSE | force:
        print(msg)
        
def _check_grid_overlap(obs, hindcast, raise_if_missing=True, threshold=0.9):
    """
    Check spatial overlap between two lat/lon grids.

    Parameters
    ----------
    obs : xarray.DataArray or xarray.Dataset
        Observational grid containing `lat` and `lon`.
    hindcast : xarray.DataArray or xarray.Dataset
        Hindcast grid containing `lat` and `lon`.
    raise_if_missing : bool, default True
        If True, raises a ValueError if observations do not fully cover the hindcast/forecast grid.
    threshold : float, optional
        Minimum fraction of overlap relative to the smaller domain
        required to proceed (default: 0.9).

    Raises
    ------
    ValueError
        If overlap fraction is smaller than the threshold.
    """

    obs_lat_min = float(obs.lat.min())
    obs_lat_max = float(obs.lat.max())
    obs_lon_min = float(obs.lon.min())
    obs_lon_max = float(obs.lon.max())

    hc_lat_min = float(hindcast.lat.min())
    hc_lat_max = float(hindcast.lat.max())
    hc_lon_min = float(hindcast.lon.min())
    hc_lon_max = float(hindcast.lon.max())

    # intersection box
    lat_min = max(obs_lat_min, hc_lat_min)
    lat_max = min(obs_lat_max, hc_lat_max)
    lon_min = max(obs_lon_min, hc_lon_min)
    lon_max = min(obs_lon_max, hc_lon_max)

    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("Observation and hindcast grids do not overlap.")

    overlap_area = (lat_max - lat_min) * (lon_max - lon_min)

    obs_area = (obs_lat_max - obs_lat_min) * (obs_lon_max - obs_lon_min)
    hc_area  = (hc_lat_max - hc_lat_min) * (hc_lon_max - hc_lon_min)

    smaller_area = min(obs_area, hc_area)
    overlap_fraction = overlap_area / smaller_area

    if overlap_fraction < threshold:
        if raise_if_missing:
            raise ValueError(
                "Grid overlap too small: {} ".format(np.round(overlap_fraction,2)),
                "f(threshold {threshold:.0%})"
            )
        else:
            _log("Grid overlap : {} ".format(np.round(overlap_fraction,2)))
            return True
    else:
        return True


def _harmonize_coords(ds):
    """
    Harmonize coordinate names to standard names: lat, lon, time.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset with potentially non-standard coordinate names.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset with coordinates renamed to `lat`, `lon`, `time`.

    Raises
    ------
    ValueError
        If latitude or longitude coordinate cannot be identified.
    """

    coord_map = {}

    lat_names = ["lat", "latitude", "Latitude", "nav_lat", "y"]
    lon_names = ["lon", "longitude", "Longitude", "nav_lon", "x"]
    time_names = ["time", "Time", "valid_time", "date"]

    for name in ds.coords:
        lname = name.lower()

        if name in lat_names or lname in lat_names:
            coord_map[name] = "lat"

        elif name in lon_names or lname in lon_names:
            coord_map[name] = "lon"

        elif name in time_names or lname in time_names:
            coord_map[name] = "time"

    if len(coord_map)>0:
        _log("Renaming coordinates: {}".format(coord_map))
    else:
        _log("Nothing to do")
        
    ds = ds.rename(coord_map)

    if "lat" not in ds.coords:
        raise ValueError("Latitude coordinate not found.")
    if "lon" not in ds.coords:
        raise ValueError("Longitude coordinate not found.")

    return ds




    
#######################################################################################
#
# main functions
#
#######################################################################################


def read_netcdf(file, variable):
    """
    Read a dataset and return the selected variable.

    The function opens a NetCDF file (or collection of files) using xarray,
    verifies that the requested variable exists, renames variables to the adopted convention (time,lat,lon) and checks that the time
    coordinate represents daily data.

    Parameters
    ----------
    file : str or list[str]
        Path or pattern pointing to NetCDF file(s). Passed to
        ``xarray.open_mfdataset``.
    variable : str
        Name of the variable to extract from the dataset.

    Returns
    -------
    xarray.DataArray
        DataArray containing the requested variable.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    OSError
        If the file cannot be opened.
    KeyError
        If the requested variable is not present in the dataset.
    ValueError
        If the dataset does not appear to contain daily data.
    """
    
    _log("\n" + "="*60)
    _log(" START: read_netcdf ".center(60, "="))
    _log("="*60 + "\n")
    
    _log("reading {} from {}".format(variable,file))
    
            
    # Check file existence (for simple paths; glob patterns handled by xarray)
    if isinstance(file, str) and not any(c in file for c in "*?[]"):
        if not os.path.exists(file):
            raise FileNotFoundError(f"data file not found: {file}")

    try:
        ds = xr.open_dataset(file)
        _log("success")
    except Exception as e:
        raise OSError("Unable to open dataset: {}".format(file)) from e

    _log("looking for {}".format(variable))
    
    if variable not in ds:
        raise KeyError("Variable '{}' not found in dataset.".format(variable))
    else:
        _log("found")
        
    # Make sure lat,lon and time are there, rename if necessary
    _log("Harmonizing coordinates...")
    ds = _harmonize_coords(ds)

    #and we return a data array rather than a dataset
    da = ds[variable]

    
    
    # just an idiot-check
    _log("check if data are daily...")    
    if da.time.size > 2:
        dt = pd.Series(da.time.values).diff().dropna()

        # dominant timestep
        median_dt = dt.median()

        if median_dt < pd.Timedelta(hours=23):
            raise ValueError(
                "Dataset appears to be sub-daily (median timestep < 1 day)."
            )        
        _log("daily with data time range: {} to {}".format(str(da.time.values[0]),str(da.time.values[-1])))
    else:
        raise ValueError(
                "Dataset appears to only one time step. This is not what is expected."
            )
        
    _log("all done")
    _log("\n" + "="*60+"\n")
    
    return da







def deaccumulate(accum):
    """
    Convert accumulated values (e.g. precipitation) to daily totals.
    
    Midnight (00:00) accumulation values are differenced to obtain daily
    totals. Because the value at 00:00 represents accumulation over the
    *previous* day, the first value is reinserted and timestamps are
    shifted back by one day so that each value corresponds to the correct
    accumulation period.
    
    For hindcasts, the first day of each hindcast cycle is removed from 
    daily totals series to avoid artificial negative differences that 
    arise when differencing across hindcast boundaries.
    
    Parameters
    ----------
    accum : xarray.DataArray or xarray.Dataset
        Accumulated values with a ``time`` coordinate.
    
    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Daily totals with timestamps representing the day of accumulation.
    """
    
    _log("\n" + "="*60)
    _log(" START: accumulated_to_daily ".center(60, "="))
    _log("="*60 + "\n")
    
    # Select midnight values (00:00) in case data are sub=daily, would it work for staggered initializations?
    _log("Selecting midnight accumulation values...")
    
    midnight = accum.sel(time=accum.time.dt.hour == 0)
    
    # Calculate day-to-day differences
    _log("Computing day-to-day differences...")
    daily = midnight.diff("time")

    #to get rid of negative values
    dailypositveonly=daily.where(daily>=0,0)

    #but have to bring nans back
    daily=dailypositveonly.where(~np.isnan(daily))
    
    #have to add the first day back
    daily = daily.reindex(time=midnight.time, fill_value=np.nan)
    
    _log("Detecting cycle starts...")
    
    # this catches first days of each cycle. It's important if only one initialization time, i.e. a non-staggered hindcast. 
    # In daily data for staggered hindasts these days will be NaNs anyway.
    # using diff here
    time_gap = midnight.time.diff("time") > np.timedelta64(1, "D")
    
    #need to add the first day back
    time_gap = time_gap.reindex(time=midnight.time, fill_value=True)
        
    #dropping time gaps - these are days that are "initialization days", i.e. for which accumulated rainfall was not given, i.e. where it was 0
    #in staggered hindcasts - these will be nans already, because diff() gives nan if the previous day was nan
    daily=daily.isel(time=np.where(~time_gap)[0])
    
    # Shift timestamps back by one day - this is done so that a value for a particular date represents the total over that date. The time stamp
    # is kept to be at 00:00, but this is no longer the previous day accumulation.
    _log("Shifting timestamps to represent accumulation day...")    
    daily["time"] = daily.indexes["time"] - pd.DateOffset(days=0.5)
    
    daily=daily.transpose("time","member","lat","lon")
    _log("all done")
    _log("\n" + "="*60+"\n")
    
    return daily





def align_grid(finegrid, coarsegrid, direction="fine_to_coarse", method="conservative", raise_if_missing=True, threshold=0.9):
    """
    Align two datasets spatially using xESMF, preserving time and member dimensions.

    Parameters
    ----------
    finegrid : xarray.DataArray
        High-resolution dataset with dimensions including `time`, `member`, `lat`, `lon`.
    coarsegrid : xarray.DataArray
        Low-resolution dataset with dimensions including `time`, `member`, `lat`, `lon`.
    direction : {"fine_to_coarse", "coarse_to_fine"}, default "fine_to_coarse"
        Direction of spatial alignment:
        - "fine_to_coarse": aggregate finegrid to coarsegrid (conservative regridding)
        - "coarse_to_fine": interpolate coarsegrid to finegrid (method specifies interpolation)
    method : str, optional
        Regridding method when interpolating coarse to fine. Ignored for fine to coarse.

    Returns
    -------
    finealigned : xarray.DataArray
        finegrid dataset aligned to the chosen spatial grid, dims (time, member, lat, lon)
    coarsealigned : xarray.DataArray
        coarsegrid dataset aligned to the chosen spatial grid, dims (time, member, lat, lon)

    Notes
    -----
    - NaN masking is applied along spatial dimensions only; all time steps and members are preserved.
    """
    _log("\n" + "="*60)
    _log(" START: align_spatial ".center(60, "="))
    _log("="*60 + "\n")


    #while using xESMF, only these make sense. But perhaps it is worth thinking about switch to rioxarray regridded for 
    # coarse_to_fine, as it offers more options than just bilinear
    
    allowed={"coarse_to_fine":["bilinear"],
             "fine_to_coarse":["conservative"]
            }
    
    if direction not in allowed.keys():
        raise ValueError("direction must be 'fine_to_coarse' or 'coarse_to_fine'")

    if method not in allowed[direction]:
        raise ValueError("Only {} methods available for {}".format(",".join(allowed[direction]),direction))
        
    #checking if grids overlap
    overlap_check=_check_grid_overlap(finegrid,coarsegrid,raise_if_missing, threshold)
        
    # select one timestep for weight generation
    fine_step = finegrid.isel(time=0, drop=True)
    coarse_step = coarsegrid.isel(time=0, drop=True)
    if 'member' in fine_step.dims:
        fine_step = fine_step.isel(member=0)
    if 'member' in coarse_step.dims:
        coarse_step = coarse_step.isel(member=0)

    # Regrid
    if direction == "fine_to_coarse":
        _log("Regridding finegrid → coarsegrid (conservative)...")
        regridder = xe.Regridder(fine_step, coarse_step, "conservative")
        finealigned = regridder(finegrid, skipna=True, na_thres=0.5)
        coarsealigned = coarsegrid

    elif direction == "coarse_to_fine":
        _log("Regridding coarsegrid → finegrid ({} )...".format(method))
        regridder = xe.Regridder(coarse_step, fine_step, method, extrap_method="inverse_dist")
        coarsealigned = regridder(coarsegrid, skipna=True)
        finealigned = finegrid

    else:
        raise ValueError("direction must be 'fine_to_coarse' or 'coarse_to_fine'")

    # Apply spatial masks using the first timestep (time=0)
    coarsealigned = coarsealigned.where(~np.isnan(finealigned.isel(time=0)))
    finealigned = finealigned.where(~coarsealigned.isnull().all(["time","member"]))


    _log("done")
    _log("\n" + "="*60+"\n")


    return finealigned, coarsealigned
    







def align_time(obs, hindcast, add_members=False, raise_if_missing=True):
    """
    Align two datasets in time by selecting timestamps from the coarse grid dataset.

    Parameters
    ----------
    obs : xarray.DataArray
        Observational dataset with a `time` dimension without gaps.
    hindcast : xarray.DataArray
        Dataset with a `time` dimension, possibly sparse coverage, initialized once per year.
    raise_if_missing : bool, default True
        Raise an error if obs time does not fully cover hindcast time range.

    Returns
    -------
    obsaligned : xarray.DataArray
        Observations dataset aligned with hindcast. This includes replication of the ensemble members structure, which 
        is necessary if hindcast includes more than one initialization time
    hindcastaligned : xarray.DataArray
        Hindcast dataset clipped to the extent of obs (if raise_if_missing is False).
    """
    _log("\n" + "="*60)
    _log(" START: align_time ".center(60, "="))
    _log("="*60 + "\n")

    # Normalize time coordinates
    _log("Normalizing time coordinates...")
    obs = obs.copy()
    obs["time"] = obs.indexes["time"].normalize()

    hindcast = hindcast.copy()
    hindcast["time"] = hindcast.indexes["time"].normalize()

    # Log ranges
    _log("Selecting obs time coordinates matching hindcast...")
    _log("obs: {} to {}\nhindcast: {} to {}".format(
        obs.time.dt.strftime("%Y-%m-%d")[0].data,
        obs.time.dt.strftime("%Y-%m-%d")[-1].data,
        hindcast.time.dt.strftime("%Y-%m-%d")[0].data,
        hindcast.time.dt.strftime("%Y-%m-%d")[-1].data
    ))

    # Select timestamps in obs that match hindcast
    sel = obs.time.isin(hindcast.time)
    obsaligned = obs.sel(time=sel)
    hindcastaligned = hindcast.sel(time=obsaligned.time)

    # Check for coverage: all hindcast times should be in obs
    missing = np.invert(hindcast.time.isin(obs.time)).sum()
    if missing > 0:
        if raise_if_missing:
            raise ValueError("obs does not cover the full hindcast period: missing {} days".format(missing))
        else:
            _log("WARNING: obs does not fully cover hindcast period")
            _log("obs: {} to {}\nhindcast: {} to {}".format(
                obs.time.dt.strftime("%Y-%m-%d")[0].data,
                obs.time.dt.strftime("%Y-%m-%d")[-1].data,
                hindcast.time.dt.strftime("%Y-%m-%d")[0].data,
                hindcast.time.dt.strftime("%Y-%m-%d")[-1].data
            ))
    else:
        _log("Time coverage verified: obs fully covers hindcast period.")

    if add_members:
        #construct pseudo-members in obs data corresponding to hindcast members
        pseudo_members=[]
        for m in hindcastaligned.member.values:
            member_data = hindcastaligned.sel(member=m)
    
            # valid timestep if any gridcell is non-nan
            valid = ~np.isnan(member_data).all(("lat", "lon"))    
            #select obs values for valid data
            pseudo_member=obsaligned.sel(time=valid)
            pseudo_member=pseudo_member.expand_dims(dim={"member":[m]})
            pseudo_members.append(pseudo_member)
    
        obsaligned = xr.concat(
            pseudo_members,
            dim="member",
            join="outer",
            coords=["enslag"]
        )
        obsaligned=obsaligned.transpose("time","member","lat","lon")
        
    
    _log("\n" + "="*60+"\n")


    return obsaligned, hindcastaligned





def leadtime_to_time(data, drop_member=False):
    
    _log("finding first date for each member")
    
    enslag=data["enslag"] #in days
    member=data["member"]
    nominal_time=data["nominal_time"]
    lead_time=data["lead_time"]
    
    enslag_by_member = data.isel(nominal_time=0)["enslag"]
     
    _log("combining members with nominal_time")
    
    result = []
    for m in member.values:
        lag = pd.to_timedelta(enslag.sel(member=m), "D")
        # original lag is calculated as init_time-nominal_time  
        #- so dates before nominal_time have lag of -1, dates after have lag of 1
        # to recover init_time from nominal_time - it is nominal_time + lag
        init_time = nominal_time + lag
        times = np.array([x + pd.to_timedelta(lead_time.data, "D") for x in init_time.data]).flatten()
        
        timedata = data.sel(member=m).stack(time=("nominal_time", "lead_time"))
        timedata = timedata.reset_index("time", drop=True)
        timedata = timedata.drop_vars(
            [c for c in timedata.coords if c not in ("lat", "lon", "time")],
            errors="ignore"
        )
        timedata = timedata.assign_coords(time=times)
        #memdata = memdata.rename(memdata="valid_time")
        
        result.append(timedata)
    
    combined = xr.concat(result, dim="member", join="outer")
    combined = combined.assign_coords(member=member.values)
    
    if drop_member:
        if "member" in combined.coords:
            combined=combined.isel(member=0).drop_vars("member").transpose("time","lat","lon")
    else:
        # reattach enslag as a clean (member,) coordinate
        combined = combined.assign_coords(enslag=("member", enslag_by_member.values)).transpose("time","member","lat","lon")
    return combined




def organize_by_validtime(data, trim=True, drop_member=False):
    """
    data: dataarray in time format
    """
    if trim:
        #finding time steps that do not have all data
        sel=np.isnan(data).all(["lat","lon"]).sum("member")==0
        
        data=data.sel(time=sel)
    member=data["member"]
    time=data["time"]
        
    #find initialization days
    isinit=(data.time.diff("time")> np.timedelta64(1, "D")).reindex(time=data.time, fill_value=True)
    
    isend=~isinit & isinit.shift(time=-1, fill_value=True)
    
    init_indices = np.where(isinit)[0]
    
    valid_time = data.time.isel(time=init_indices.data)
    
    #indices of member's last days
    end_indices = np.where(isend)[0]


    #iterating through cycles/blocks
    blocks=[]
    for i in range(len(init_indices)):
        init_idx=init_indices[i]
        end_idx=end_indices[i]+1
        block = data.isel(time=slice(init_idx, end_idx))
        block = block.rename({"time": "lead_time"})
        block = block.assign_coords(
            lead_time=range(block.sizes["lead_time"])
        )
        block["lead_time"].attrs={"units":"day"}
        
        # Use init_year from enumeration
        block = block.expand_dims(dim={"valid_time":[valid_time.data[i]]})
    
        blocks.append(block)
        
    combined=xr.concat(blocks, "valid_time").transpose("valid_time","lead_time","member","lat","lon")
    if drop_member:
        combined=combined.isel(member=0).drop_vars("member").transpose("valid_time","lead_time","lat","lon")

    return combined





def organize_by_leadtime(data,nominal_time=None,is_observed=False):
    """
    Reshape data into (nominal_time, member, lead_time, lat, lon),
    supporting multiple initializations inside a single file.
    
    Initialization times are inferred independently for each member
    from transitions in the spatial NaN mask.
    """
    
    _log("\n" + "="*60)
    _log(" START: organize_by_leadtime ".center(60, "="))
    _log("="*60 + "\n")
    
    data = data.sortby("time")
    
    #obs data will come without member dimension, so just for consistency of further processing this dimension is added. We will drop it later, though.
    if not "member" in data.dims:
        data=data.expand_dims(dim={"member":[0]})
        
    _log("finding initialization date for each member")
    #find initialization days
    isinit_time=(data.time.diff("time") > np.timedelta64(1, "D")).reindex(time=data.time, fill_value=True)
    
    
    if nominal_time is None:
        # Assign time to each cycle - that is the time of the earliest initialization
        nominal_times = data.time[isinit_time].data
    else:
        nominal_time=pd.to_datetime(nominal_time)
        
        nominal_times = data.time[isinit_time].data
        nominal_times = [pd.to_datetime(d).replace(month=nominal_time.month, day=nominal_time.day) for d in nominal_times]
        
    #iterate through members
    members=[]
    for m in data.member.values:
        member_data = data.sel(member=m)
        #initialization days when valid data start after a stagger gap
        # this cannot be inferred from time, as it might be different for each member
        # valid timesteps when any gridcell is non-nan
        valid = ~np.isnan(member_data).all(("lat", "lon"))
        
        #this is first valid day after the gap
        isinit_gap = valid & ~valid.shift(time=1, fill_value=False)
        
        #this will be index of the last valid
        isend_gap=valid & ~valid.shift(time=-1, fill_value=False)
    
        # now for non-staggered hindcasts, there are no gaps, so we need to use time-based init
        if isinit_gap.sum()>1:
            isinit=isinit_gap
            isend=isend_gap
        else:
            isinit=isinit_time
            isend=~isinit_time & isinit_time.shift(time=-1, fill_value=True)
        
        #indices of member's initialization days
        init_indices = np.where(isinit)[0]
        
        #indices of member's last days
        end_indices = np.where(isend)[0]
            
        #iterating through cycles/blocks
        blocks=[]
        for i in range(len(init_indices)):
            init_idx=init_indices[i]
            end_idx=end_indices[i]+1
            block = member_data.isel(time=slice(init_idx, end_idx))
            block = block.rename({"time": "lead_time"})
            #block = block.assign_coords(
            #    lead_time=pd.to_timedelta(range(block.sizes["lead_time"]), unit="D")
            #)
            block = block.assign_coords(
                lead_time=range(block.sizes["lead_time"])
            )
            block["lead_time"].attrs={"units":"day"}
            
            # Use init_year from enumeration
            block = block.expand_dims(nominal_time=[nominal_times[i]], member=[m])
    
            blocks.append(block)
    
        members.append(xr.concat(blocks,
        		dim="nominal_time",
        		join="outer",
        		coords=["enslag"]
        		)
        	)
    _log("Concatenating member blocks...")
    
    reshaped = xr.concat(
        members,
        dim="member",
        join="outer",
        coords=["enslag"]
    )
    reshaped=reshaped.transpose("nominal_time","lead_time","member","lat","lon")
    
    return reshaped
    





def load_raw_data(nominal_date,
                        download_dir,
                        target_domain,
                        fcst_model,
                        fcst_var,
                        obs_file,
                        obs_var,
                        verbose=True,
                       ):
    """
    reading raw forecast, hindcast, and observations.

    This function reads forecast, hindcast, and observation NetCDF files.

    Parameters
    ----------
    nominal_date : str or datetime-like
        Nominal initialization date of the forecast. Used to identify
        the forecast and hindcast files and to align hindcasts with the
        forecast cycle.

    download_dir : str
        Directory containing forecast and hindcast NetCDF files.

    target_domain : str
        Spatial domain identifier used in the input filenames.

    fcst_model : str
        Forecast model name (e.g., "ECMWF").

    fcst_var : str
        Forecast variable name in the NetCDF files.

    Returns
    -------
    hindcast : xarray.DataArray 
        Hindcast data in original structure with dimensions:

        (lead_time, init_date, member, lat, lon)

    forecast : xarray.DataArray
        Forecast data in original structure.

    obs : xarray.DataArray
        Observations data in original structure

        (date, lat, lon)

    Notes
    -----
    * ECMWF precipitation (`tp`) is provided as running accumulation and
      is automatically converted to daily totals.
    
    """

    #setting up verbosity
    set_verbose(verbose)
    _log("logging is on")

    
    #feedback
    _log("nominal forecast date: {}".format(nominal_date), force=True)
    _log("download dir: {}".format(download_dir), force=True)
    _log("domain: {}".format(target_domain), force=True)
    _log("model: {}".format(fcst_model), force=True)
    _log("forecast variable: {}".format(fcst_var), force=True)
    _log("observed file: {}".format(obs_file), force=True)
    _log("observed variable: {}".format(obs_var), force=True)

    
    #*******
    # setting things up
    #*******
    #this is also validation of nominaldate
    try:
        fcst_date=pd.to_datetime(nominal_date)
    except:
        raise TypeError("Forecast date does not appear to be valid")
        
    fcst_date_str=fcst_date.strftime("%Y%m%d")
    fcst_day=fcst_date.day
    fcst_mon=fcst_date.month
    fcst_year=fcst_date.year
    
    # download function aligns hindcast date with forecast date, so
    hcst_date=fcst_date

    #*******
    # processing
    #*******
    
    # forecast data file
    fcst_file="{}/{}_{}_{}_{}_fc.nc".format(download_dir,fcst_var,fcst_model, fcst_date_str, target_domain)
    
    # hindcast data file
    hcst_file="{}/{}_{}_{}_{}_hc.nc".format(download_dir,fcst_var,fcst_model, fcst_date_str, target_domain)
    
    #reading forecast data
    forecast=read_netcdf(fcst_file, fcst_var)
    
    #reading hindast
    hindcast=read_netcdf(hcst_file, fcst_var)
    
    # reading observed data
    obs=read_netcdf(obs_file, obs_var)

    #returning arrays
    return hindcast,forecast,obs
    






def postprocess_forecast(nominal_date,
                        download_dir,
                        target_domain,
                        fcst_model,
                        fcst_var,
                        agg_window,
                        agg_method,
                        obs_file,
                        obs_var,
                        verbose=True,
                        time_alignment_kwargs=None,
                        grid_alignment_kwargs=None
                       ):
    """
    End-to-end postprocessing pipeline for forecast, hindcast, and observations.

    This function performs all postprocessing steps required to prepare
    forecast, hindcast, and observational datasets for verification or
    downstream analysis. The workflow includes:

    1. Reading forecast, hindcast, and observation NetCDF files.
    2. Converting accumulated variables to daily totals (when required).
    3. Spatial grid alignment between model data and observations.
    4. Temporal alignment of observations with hindcast structure.
    5. Reshaping datasets into a lead-time oriented format.
    6. Temporal aggregation over user-defined windows (e.g. 5-day, 7-day).

    The output datasets share a consistent structure that allows direct
    comparison between hindcasts, forecasts, and observations.

    Parameters
    ----------
    nominal_date : str or datetime-like
        Nominal initialization date of the forecast. Used to identify
        the forecast and hindcast files and to align hindcasts with the
        forecast cycle.

    download_dir : str
        Directory containing forecast and hindcast NetCDF files.

    target_domain : str
        Spatial domain identifier used in the input filenames.

    fcst_model : str
        Forecast model name (e.g., "ECMWF").

    fcst_var : str
        Forecast variable name in the NetCDF files.

    agg_window : int
        Length of the temporal aggregation window in days
        (e.g., 1 for daily values, 5 for pentads, 7 for weekly).

    agg_method : {"mean", "sum"}
        Aggregation method applied within each aggregation window.

    obs_file : str
        Path to the observation NetCDF file.

    obs_var : str
        Variable name in the observation file.

    verbose : bool, default False
        If True, enables logging output during postprocessing.

    time_alignment_kwargs : dict, optional
        Additional keyword arguments passed to `align_time()`.

    grid_alignment_kwargs : dict, optional
        Additional keyword arguments passed to `align_grid()`.


    Returns
    -------
    hindcast_leadtime_window_aggregate : xarray.DataArray 
        Hindcast data reshaped into lead-time structure with dimensions:

        (lead_time, init_date, member, lat, lon)

        and aggregated over the requested temporal window.

    forecast_leadtime_window_aggregate : xarray.DataArray
        Forecast data in the same lead-time structure as hindcasts.

    obs_leadtime_window_aggregate : xarray.DataArray
        Observations aligned with hindcast cycles and reshaped into
        lead-time structure. The member dimension is removed:

        (lead_time, init_date, lat, lon)

    Notes
    -----
    * ECMWF precipitation (`tp`) is provided as running accumulation and
      is automatically converted to daily totals.

    * Spatial alignment ensures that forecast and hindcast data are
      restricted to grid cells where observations are available.

    * Temporal alignment duplicates the hindcast time structure for
      observations, enabling direct comparison across lead times.

    * Lead-time restructuring allows analysis of aggregated forecast
      periods such as weekly or pentad averages/totals.
    """

    #setting up verbosity
    set_verbose(verbose)
    _log("logging is on")

    
    #reading kwargs
    grid_alignment_kwargs = grid_alignment_kwargs or {}
    time_alignment_kwargs = time_alignment_kwargs or {}


    #feedback
    _log("nominal forecast date: {}".format(nominal_date), force=True)
    _log("download dir: {}".format(download_dir), force=True)
    _log("domain: {}".format(target_domain), force=True)
    _log("model: {}".format(fcst_model), force=True)
    _log("forecast variable: {}".format(fcst_var), force=True)
    _log("aggregation window: {}".format(agg_window), force=True)    
    _log("observed file: {}".format(obs_file), force=True)
    _log("observed variable: {}".format(obs_var), force=True)

    
    #*******
    # setting things up
    #*******
    #this is also validation of nominaldate
    try:
        fcst_date=pd.to_datetime(nominal_date)
    except:
        raise TypeError("Forecast date does not appear to be valid")
        
    fcst_date_str=fcst_date.strftime("%Y%m%d")
    fcst_day=fcst_date.day
    fcst_mon=fcst_date.month
    fcst_year=fcst_date.year
    
    # download function aligns hindcast date with forecast date, so
    hcst_date=fcst_date

    #*******
    # processing
    #*******
    
    # forecast data file
    fcst_file="{}/{}_{}_{}_{}_fc.nc".format(download_dir,fcst_var,fcst_model, fcst_date_str, target_domain)
    
    # hindcast data file
    hcst_file="{}/{}_{}_{}_{}_hc.nc".format(download_dir,fcst_var,fcst_model, fcst_date_str, target_domain)
    
    #reading forecast data
    tpfc=read_netcdf(fcst_file, fcst_var)
    
    #reading hindast
    tphc=read_netcdf(hcst_file, fcst_var)
    
    # reading observed data
    obs=read_netcdf(obs_file, obs_var)
    
    if fcst_model=="ECMWF" and fcst_var in ["tp"]:
        #ECMWF provides running accumulation, and it needs to be converted to daily totals     
        #converting to daily
        forecast=accumulated_to_daily(tpfc)
        
        hindcast=accumulated_to_daily(tphc)
        
    else:
        #further processig uses forecast and hincast, so we need to create them for consistency
        forecast=tpfc.copy()
        hindcast=tphc.copy()
    

    #aligning spatial grids of hindcast and obs
    
    #these ones are to be used when we want to do analyses at the grid of forecast/hindcast, i.e. at a coarse resolution
    # here, observations are aggregated to crate data at the spatial resolution/grid of the forecast model
    # note that both hindcast and forecast have to be processed, because this function also masks out forecast 
    # and hindcast data so that they only cover the area for which observations are available.
    obsaligned,forecastaligned=align_grid(obs, forecast, **grid_alignment_kwargs)
    obsaligned,hindcastaligned=align_grid(obs, hindcast, **grid_alignment_kwargs)
    
        
    # aligning time of obs and hindcast
    # this function duplicates the structure of hindcast data, i.e. the dimensions of obs_tslice are the same as those of hindcastaligned
    # i.e. time,member,lat,lon. This will neatly facilitate futher processing.
    # obviously, we cannot do this for forecast as we do not know observations yet
    obs_tslice,hindcast_tslice=align_time(obsaligned, hindcastaligned, **time_alignment_kwargs)
    
    #restructurng into lead-time oriented structure
    # this funtion reshapes data into structure that will allow working with, say, 5-day or 7-day averages
    # this is done for all: hindcast, forecast and (hindcast-aligned) observations
    # the output for hindcast and forecast has the following dimensions: lead_time,init_date, member,lat,lon
    # for obs - lead_time,init_date,lat,lon
    # for forecast - the init_date is the date of the first day of forecast,
    # for hindcast - it is the first date of the ealierst lagged hindcast. If hindcast is not staggered, then it is the date of the first day of that hindcast. 
    # for obs - it is the date corresponding to hindcast init_date

    hindcast_leadtime_window_aggregate=organize_by_leadtime(hindcast_tslice, agg_window, agg_method)
    forecast_leadtime_window_aggregate=organize_by_leadtime(forecastaligned, agg_window, agg_method)
    #obs_tslice has member dimension, this dimension can now be dropped
    obs_leadtime_window_aggregate=organize_by_leadtime(obs_tslice, agg_window, agg_method,drop_members=True)

    _log("postprocessing done", force=True)
    
    return hindcast_leadtime_window_aggregate,forecast_leadtime_window_aggregate,obs_leadtime_window_aggregate





    

def get_example_data(data_dir):
    """
    Download example datasets for toolkit-extension.

    Downloads observation, hindcast, forecast, and domain GeoJSON files
    from remote storage to a local directory. Files are verified against
    known checksums. Skips downloading files that already exist locally.

    Args:
        data_dir (str): Path to the local directory where files will be
            saved. Created automatically if it does not exist.

    Returns:
        bool: True if all files are available locally.

    Raises:
        OSError: If the directory cannot be created.

    Example:
        >>> import toolkit_extension as te
        >>> te.get_example_data("./data")
        downloading obs file
        downloading hindcast file
        downloading forecast file
        downloading domain_geojson file
        True
    """
    
    file_urls = {"obs":["https://web.csag.uct.ac.za/~wolski/acacia/toolkit/PRCPTOT_day_CHC_CHIRPS-2.0-0p25_merged.nc","9a3ac038915fbf9f956f11991f33356276a55351f1dd1f55f617eb2c814b5e1c"],
    "hindcast":["https://web.csag.uct.ac.za/~wolski/acacia/toolkit/tp_ECMWF_20260301_madagascar_hc.nc","87f2cd4e064ff7a4d144b9ac265f001e97bdea619e38a613f0e68ff0661bb48b"],
    "forecast":["https://web.csag.uct.ac.za/~wolski/acacia/toolkit/tp_ECMWF_20260301_madagascar_fc.nc","deeff561fa3ff0bea1e2874a74cf27f0c5aae90225e52e17b083df7e9262917a"],
    "domain_geojson":["https://web.csag.uct.ac.za/~wolski/acacia/toolkit/madagascar.geojson","39ebb562fa1b4b93e972f3657005ab834c0604d465d71916055fe45d8c1e75d4"]}
             
    #create this directory if this does not exist
    if not os.path.exists(data_dir):
        try:
            os.makedirs(data_dir)
        except:
            raise OSError(
                "Cannot create directory {}".format(data_dir)
                )
    for data_type, [url, known_hash] in file_urls.items():
        file_name = "{}/{}".format(data_dir, os.path.basename(url))
        if not os.path.exists(file_name):
            print("downloading {} file".format(data_type))
            try:
                file_name = pooch.retrieve(
                    url=url,
                    known_hash=known_hash,
                    fname=os.path.basename(url),
                    path=data_dir
                )
            except ValueError as e:
                raise RuntimeError(
                    "Failed to download {} file from {}.\n"
                    "The file may no longer be available on the remote server.\n"
                    "Original error: {}".format(data_type, url, e)
                )
            except OSError as e:
                raise ConnectionError(
                    "Failed to download {} file — no internet connection.\n"
                    "Original error: {}".format(data_type, e)
                )


        else:
            print("file already exists locally {}".format(file_name))
    return True






# registry of available aggregation functions
AGG_FUNCTIONS = {}

def register_agg(name):
    """Decorator to register an aggregation function under a name."""
    def wrapper(func):
        AGG_FUNCTIONS[name] = func
        return func
    return wrapper


# --- built-ins ---

@register_agg("mean")
def _mean(da, dim, **kwargs):
    return da.mean(dim=dim)

@register_agg("sum")
def _sum(da, dim, **kwargs):
    return da.sum(dim=dim)

@register_agg("count_zero")
def _count_zero(da, dim, **kwargs):
    return (da == 0).sum(dim=dim)

@register_agg("count_above")
def _count_above(da, dim, threshold=0, **kwargs):
    return (da > threshold).sum(dim=dim)

@register_agg("max")
def _max(da, dim, **kwargs):
    return da.max(dim=dim)
    





def aggregate(
    data,
    window_size,
    method="mean",
    dim="lead_time",
    **kwargs):
    """
    Resample `data` over `dim` into `agg_window`-sized bins, applying
    the named aggregation function, and masking incomplete trailing windows.
    
    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
    window_size : int, e.g. 5
    method : str, name of a function registered in AGG_FUNCTIONS
    dim : str, dimension being resampled (default "lead_time")
    **kwargs : passed through to the aggregation function
    """
    
    _log(f"Aggregating lead_time to {window_size}-D blocks using ({method})...")

    #arbitrary choice
    agg_windows=list(range(5,11))

    #checking if method is available
    if method not in AGG_FUNCTIONS:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Available: {list(AGG_FUNCTIONS)}"
        )
        
    #picking up requested method
    func = AGG_FUNCTIONS[method]
    
    # validating aggregation window
    if window_size not in agg_windows:
        raise ValueError(
            "agg_window must be one of {} got '{}'".format(" ".join(agg_windows),agg_window)
        )
    
    #agg_window is just window in the xarray format
    agg_window=f"{window_size}D"

    # this converts integer index to timedelta - and this is necessary for the aggregator function
    data = data.assign_coords({dim: pd.to_timedelta(data[dim], "D")})

    # truncate to a multiple of window_days before resampling
    n_total = data.sizes[dim]
    n_full = (n_total // window_size) * window_size
    
    if n_full < n_total:
        dropped = n_total - n_full
        print(
                f"resample_agg: dropping last {dropped} step(s) of '{dim}' "
                f"({n_total} not divisible by window_size={window_size}); "
                f"keeping {n_full} steps."
        )
        data = data.isel({dim: slice(0, n_full)})
    
    resampler = data.resample(
        **{dim: agg_window},
        label="left",
        closed="left"
    )
    
    #aggregating done here!
    aggdata = resampler.map(lambda da: func(da, dim=dim, **kwargs))
    
    
    #finding nans
    coords=["lead_time"]

    for v in ["member","nominal_time","valid_time","init_time"]:
        if v in data.coords:
            coords.append(v)
            
    spatial_nan_mask = np.isnan(data).all(coords)
    
    #removing all nans
    aggdata = aggdata.where(~spatial_nan_mask)    
    
    #bringing the lead_time to days from timedelta
    aggdata[dim]=(aggdata[dim]/np.timedelta64(1, 'D')).astype(int)
    
    return aggdata


#this is temporary function
def add_enslag(data, nominal_date):
    isinit_time=(data.time.diff("time") > np.timedelta64(1, "D")).reindex(time=data.time, fill_value=True)
    
    if nominal_date is None:
        # Assign time to each cycle - that is the time of the earliest initialization
        nominal_dates = data.time[isinit_time].data
    else:
        nominal_date=pd.to_datetime(nominal_date)
        
        nominal_dates = data.time[isinit_time].data
        nominal_dates = [pd.to_datetime(d).replace(month=nominal_date.month, day=nominal_date.day) for d in nominal_dates]
        
    #iterate through members
    enslags=[]
    ensdates=[]
    for m in data.member.values:
        member_data = data.sel(member=m)
        #initialization days when valid data start after a stagger gap
        # this cannot be inferred from time, as it might be different for each member
        # valid timesteps when any gridcell is non-nan
        valid = ~np.isnan(member_data).all(("lat", "lon"))
        
        #this is first valid day after the gap
        isinit_gap = valid & ~valid.shift(time=1, fill_value=False)
        
        #this will be index of the last valid
        isend_gap=valid & ~valid.shift(time=-1, fill_value=False)
    
        # now for non-staggered hindcasts, there are no gaps, so we need to use time-based init
        if isinit_gap.sum()>1:
            isinit=isinit_gap
            isend=isend_gap
        else:
            isinit=isinit_time
            isend=~isinit_time & isinit_time.shift(time=-1, fill_value=True)
        
        #indices of member's initialization days
        init_indices = np.where(isinit)[0]
        
        #indices of member's last days
        end_indices = np.where(isend)[0]
        
        enslag=int((data.time.data[init_indices[0]]-nominal_dates[0])/np.timedelta64(1, 'D'))
    
        enslags.append(enslag)
        ensdates.append(data.time.data[init_indices[0]])
    
    data=data.assign_coords(enslag=("member",enslags))
    return data
