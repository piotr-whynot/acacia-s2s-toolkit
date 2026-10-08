# SPDX-FileCopyrightText: 2024 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

# output suitable ECDS variables in light of requested forecasts.
from acacia_s2s_toolkit import variable_dict, argument_check
from datetime import datetime, timedelta
from importlib import resources
import requests
import numpy as np
import pandas as pd
import ast
import calendar
import os
from pathlib import Path
import yaml

def read_lookup_table(fcdate='20250828'):
    # Open public ecbox contain .csv files with lookup tables.
    url = "https://sites.ecmwf.int/ecbox/acacia_s2s_toolkit/s/api/v2/pub/files/lookup_tables"
    r = requests.get(url)
    lookup_tables_list = r.json()

    # get date components of all lookup_tables
    file_dates = np.sort([datetime.strptime(f["path"].split("_")[-1].replace(".csv", ""),"%Y%m%d") for f in lookup_tables_list["files"] if f["path"].startswith("lookup_table_")])

    # get date of first look up table
    first_lookup_table_time = file_dates[0]
    # convert fcdate to a time field. 
    fc_dt = datetime.strptime(fcdate,'%Y%m%d')

    ###### Select the most recent lookup table compared to the requested forecast date #####
    if fc_dt < first_lookup_table_time:
        first_lookup_table_time_str = first_lookup_table_time.strftime('%Y%m%d')
        csv_file = f'lookup_table_{first_lookup_table_time_str}.csv'
    else:
        # select valid dates (all less than fcdate)
        valid_dates = [date for date in file_dates if date <= fc_dt]
        if not valid_dates:
            raise FileNotFoundError(f"No lookup table found before {fcdate}")

        # pick the most recent one, then read that lookup table
        chosen_date = max(valid_dates)
        chosen_date_str = chosen_date.strftime('%Y%m%d')
        csv_file = f'lookup_table_{chosen_date_str}.csv'

    # read lookup table
    df = pd.read_csv(f'{url}/{csv_file}') # read csv file (lookup table). 

    columns_to_eval = ["fcFreq", "dayfcLags", "rfRange", "rfLagDetail"]

    for col in columns_to_eval:
        # if the option is a list of numbers or zero. line is ensuring lists and numbers are not strings
        df[col] = df[col].apply(lambda x: int(x) if x == "0" else (ast.literal_eval(x) if isinstance(x, str) and (x.startswith("[") or x.startswith("(")) else x)) 
    return df

def convert_dayofweek_to_leadtime(fcdate, start_lt):
    if isinstance(start_lt, int):
        return start_lt

    weekdays = {
        'monday': 0,
        'tuesday': 1,
        'wednesday': 2,
        'thursday': 3,
        'friday': 4,
        'saturday': 5,
        'sunday': 6,
    }

    fc_dt = datetime.strptime(str(fcdate), "%Y%m%d")

    days_ahead = (weekdays[start_lt.lower()] - fc_dt.weekday()) % 7
    print (f'days ahead: {days_ahead}')

    return days_ahead * 24

def get_single_parameter(origin_id,fcdate,parameter):
    # first read lookup table
    df = read_lookup_table(fcdate)
    # find forecast length for originID
    match = df.loc[df["Origin"] == origin_id, parameter]

    if match.empty:
        print (f"[ERROR] could not find {parameter} for originID '{origin_id}'.")
        return None

    return match.iloc[0]

def get_timeresolution(variable):
    # first find which sub-category the variable sits in
    time_resolution=None
    for category_name, category_dict in variable_dict.s2s_variables.items():
        for subcategory_name, subcategory_vars in category_dict.items():
            if variable in subcategory_vars:
                time_resolution = subcategory_name
                break # found correct time resolution
        if time_resolution:
            break # break outer loop

    if time_resolution is None:
        print (f"[ERROR] could not find variable '{variable}'.")
        return None
    return time_resolution

def output_leadtime_hour(variable,origin_id,start_lt,end_lt,period,filter_accumulation=True,):
    """
    Given variable, output suitable lead times between start_lt and end_lt.

    If filter_accumulation=True:
        accumulated variables only retain the end of each accumulation period.

    If filter_accumulation=False:
        return the full native model lead-time grid.
    """

    time_resolution = get_timeresolution(variable)

    # determine native step size
    if time_resolution.endswith("6hrly"):
        step_hours = 6
    else:
        step_hours = 24

    # build lead-time array
    leadtime_hour = np.arange(start_lt,end_lt + 1,step_hours,dtype=int,)

    # for accumulations, keep only period endpoints
    if (filter_accumulation and time_resolution.startswith("accumulated")):
        if isinstance(period, str) and period.endswith("D"):
            accum_days = int(period[:-1])
        else:
            accum_days = 1

        accum_hours = accum_days * 24

        leadtime_hour = leadtime_hour[((leadtime_hour-start_lt) % accum_hours) == 0]

    # JMA does not provide 0-hour instantaneous daily forecasts
    is_jma = (origin_id == "rjtd")
    is_instant_24 = (time_resolution.startswith("instantaneous") and step_hours == 24)

    if is_jma and is_instant_24:
        leadtime_hour = leadtime_hour[leadtime_hour != 0]

    # MJO initial state not saved
    if variable == "MJO":
        leadtime_hour = leadtime_hour[leadtime_hour != 0]

    return leadtime_hour

def output_sfc_or_plev(variable):
    '''
    Given variable (variable abbreivation), output whether variable is sfc level or on pressure levels?
    return: level_type
    '''
    # Flatten all variables from nested dictionary
    level_type=None
    for category_name, category_dict in variable_dict.s2s_variables.items():
        for subcategory_vars in category_dict.values():
            if variable in subcategory_vars:
                level_type = category_name
                return level_type
    if level_type == None:
        print (f"[ERROR] No leveltype found for '{variable}'.")
        return level_type

def output_webapi_variable_name(variable):
    ''' 
    Given variable abbreviation, output webAPI paramID.
    return webAPI paramID.

    '''
    for variable_abb, webapi_code in variable_dict.webAPI_params.items():
        if variable == variable_abb:
            return webapi_code
    print (f"[ERROR] No webAPI paramID found for '{variable}'.")
    return None

def output_originID(model,fcdate):
    '''
    Given model name, output originID.
    return originID.

    '''
    # first read lookup table
    df = read_lookup_table(fcdate)
    # find forecast length for originID
    match = df.loc[df["Model"] == model, "Origin"]

    if match.empty:
        print (f"[ERROR] could not find for originID for model '{model}'.")
        return None

    return match.iloc[0]

def output_ECDS_variable_name(variable):
    '''
    Given variable name, output the matching ECDS variable name
    
    return ECDS_varname (ECMWF Data Store)
    '''
    ECDS_varname='10m_uwind'
    return ECDS_varname

def output_plevs(variable):
    '''
    Output suitable plevs, if q, (1000, 925, 850, 700, 500, 300, 200) else add 100, 50 and 10 hPa. 
    '''
    all_plevs=[1000,925,850,700,500,300,200,100,50,10]
    if variable == 'q':
        plevs=all_plevs[:-3] # if q is chosen, don't download stratosphere
    else:
        plevs=all_plevs
    print (f"Selected the following pressure levels: {plevs}")
    
    return plevs

def output_fc_lags(origin_id,fcdate):
    '''
    Given origin_id, output lagged ensemble forecasts.
    return array with day lag positions, i.e. [0,-1,-2].
    '''
    fclags = get_single_parameter(origin_id,fcdate,'dayfcLags')

    # Special handling for CPTEC (sbsj). Initialisation only given for Wednesday and Thursday.
    if origin_id == 'sbsj':
        date_obj = datetime.strptime(fcdate, '%Y%m%d')
        weekday = date_obj.weekday()+1  # Monday = 1, ..., Sunday = 7
        if weekday == 4:  # Thursday
            return [0, -1]
        else:
            return [0]
    # Return the list from the dictionary
    return fclags

def output_hc_lags(origin_id,fcdate):
    '''
    Given origin_id, outputted the best lags for downloading reforecasts
    '''
    lag_type = get_single_parameter(origin_id,fcdate,'rfLagType')
    rf_freq_info = get_single_parameter(origin_id,fcdate,'rfLagDetail')

    fcdate_obj = datetime.strptime(fcdate, '%Y%m%d')
    weekday = fcdate_obj.weekday()+1  # Monday = 1, ..., Sunday = 7
    dayofmonth = fcdate_obj.day

    # ---- Nearest DOM (single closest) -------
    if "nearestDOM" == lag_type: # DOMs. get closest lags
        # Build candidate reforecast dates in prev/current/next month
        candidates = []
        for offset_month in [-1, 0, 1]:
            # Month/year shift
            month = (fcdate_obj.month - 1 + offset_month) % 12 + 1
            year = fcdate_obj.year + ((fcdate_obj.month - 1 + offset_month) // 12)
            for dom in rf_freq_info:
                try:
                    candidates.append(datetime(year, month, dom))
                except ValueError:
                    continue  # skip invalid dates (e.g., Feb 30)

        # Find the one with smallest abs(lag)
        closest_rf = min(candidates, key=lambda d: abs((d - fcdate_obj).days))
        lag = (closest_rf - fcdate_obj).days
        return lag
    # ------ Before/after DOM -----------------
    if "before_after_DOM" == lag_type:
        # Build candidate reforecast dates in prev/current/next month
        candidates = []
        for offset_month in [-1, 0, 1]:
            # Month/year shift
            month = (fcdate_obj.month - 1 + offset_month) % 12 + 1
            year = fcdate_obj.year + ((fcdate_obj.month - 1 + offset_month) // 12)
            for dom in rf_freq_info:
                try:
                    candidates.append(datetime(year, month, dom))
                except ValueError:
                    continue  # skip invalid dates (e.g., Feb 30)

        lags = [(c-fcdate_obj).days for c in candidates]

        neg_lags = [l for l in lags if l < 0]
        pos_lags = [l for l in lags if l >= 0]

        largest_neg = max(neg_lags) if neg_lags else None
        smallest_pos = min(pos_lags) if pos_lags else None

        return [largest_neg,smallest_pos] # before and after lags
    # ------ daily reforecast initialisations ------
    if "daily_lagged" == lag_type:
        return rf_freq_info

    # ------ Weekday based (i.e. only Monday and Thursday reforecasts)-----
    if "weekday_based" == lag_type:
        # two options depending on whether fcdate is a Monday or Thursday. 
        if fcdate_obj >= datetime(2023,6,27) and origin_id == 'ecmf': # for ecmwf between 20230627 and 20241112, ECMWF ran daily forecasts but MoThMo reforecasts. 
            return two_Monday_Thursday_rfdates(fcdate_obj)
        else:
            if weekday == 4: # Thursday
                return [-3,0]
            elif weekday == 1: # Monday
                return [0,3]
            else:
                raise ValueError(f"[ERROR] For origin_id '{origin_id}' forecasts are only every Monday and Thursday, therefore a Monday or Thursday forecast date must be selected.")

    # ---- Unique modes  ------
    if "unique" == lag_type:
        # ECMWF ---- odd day reforecasts.
        if rf_freq_info == 'odddates':
            if (dayofmonth % 2 == 0) or (dayofmonth == 29 and fcdate_obj.month == 2):
                return [-1,1] # for even fcdates (or 29th Feb) chosen rfdate before and after forecast date.
            elif (dayofmonth == 1) and (fcdate_obj.month == 1): # for 1st Jan, choose 31st Dec and 1st Jan
                return [-1,0]
            else:
                return [-2,0] # for odd dates, choosen current day, minus 2. 
        if rf_freq_info == 'CNRevery5days': # roughly five days
            # make an array of dates from 2020-01-01 to 2020-12-27
            DOM = {1:[1,6,11,16,21,26,31],2:[5,10,15,20,25],3:[2,7,12,17,22,27],4:[1,6,11,16,21,26],5:[1,6,11,16,21,26,31],6:[5,10,15,20,25,30],7:[5,10,15,20,25,30],8:[4,9,14,19,24,29],9:[3,8,13,18,23,28],10:[3,8,13,18,23,28],11:[2,7,12,17,22,27],12:[2,7,12,17,22,27]}
            CNR_rf_dates = []
            for year in (2020,2021):
                for month, days in DOM.items():
                    for day in days:
                        CNR_rf_dates.append(datetime(year,month,day))
            # change year of fcdate to 2020-fcdate(MM)-fcdate(DD)
            fc_date_2020 = datetime.strptime(f"2020{fcdate[4:]}",'%Y%m%d')
            # nearest date to altered fcdate is rf_date
            closest_day=min(CNR_rf_dates,key=lambda x:abs(fc_date_2020-x))
            lag = (closest_day - fc_date_2020).days
            return lag
        if rf_freq_info == 'JMAtwicepermonth': # twice per month
            # make an array of dates from 2020-01-01 to 2020-12-27
            DOM = {1:[16,31],2:[10,25],3:[12,27],4:[11,26],5:[16,31],6:[15,30],7:[15,30],8:[14,29],9:[13,28],10:[13,28],11:[12,27],12:[12,27]}
            JMA_rf_dates = []
            for year in (2019,2020,2021):
                for month, days in DOM.items():
                    for day in days:
                        JMA_rf_dates.append(datetime(year,month,day))
            # change year of fcdate to 2020-fcdate(MM)-fcdate(DD)
            fc_date_2020 = datetime.strptime(f"2020{fcdate[4:]}",'%Y%m%d')

            lags = [(c-fc_date_2020).days for c in JMA_rf_dates]
            neg_lags = [l for l in lags if l < 0]
            pos_lags = [l for l in lags if l >= 0]

            largest_neg = max(neg_lags) if neg_lags else None
            smallest_pos = min(pos_lags) if pos_lags else None
            
            # need to change lag if leap year. the lags are calculated with year 2020 in mind but it may be the case that the chosen forecasted year is not a leap year, hence day lags will be out by 1.
            fc_is_leap_year = calendar.isleap(datetime.strptime(fcdate,'%Y%m%d').year)
            leap_day_2020 = datetime(2020,2,29)
            # Adjust largest_neg
            if (largest_neg is not None) and (not fc_is_leap_year):
                ref_dt_neg = fc_date_2020 + timedelta(days=largest_neg)
                if crosses_leap_day(ref_dt_neg, fc_date_2020, leap_day_2020):
                    # Remove the extra day: shrink |lag| by 1.
                    # Since it's negative, add +1 (e.g., -10 -> -9)
                    largest_neg = largest_neg + 1
        
            # Adjust smallest_pos
            if (smallest_pos is not None) and (not fc_is_leap_year):
                ref_dt_pos = fc_date_2020 + timedelta(days=smallest_pos)
                if crosses_leap_day(ref_dt_pos, fc_date_2020, leap_day_2020):
                    # Remove the extra day: shrink |lag| by 1.
                    # Since it's non-negative, subtract 1 (e.g., 10 -> 9)
                    smallest_pos = smallest_pos - 1
        
            return [largest_neg,smallest_pos]
        if rf_freq_info == 'JMAoncepermonth': # twice per month
            # make an array of dates from 2020-01-01 to 2020-12-27
            DOM = {1:[16,31],2:[10,25],3:[12,27],4:[11,26],5:[16,31],6:[15,30],7:[15,30],8:[14,29],9:[13,28],10:[13,28],11:[12,27],12:[12,27]}
            JMA_rf_dates = []
            for year in (2019,2020,2021):
                for month, days in DOM.items():
                    for day in days:
                        JMA_rf_dates.append(datetime(year,month,day))
            # change year of fcdate to 2020-fcdate(MM)-fcdate(DD)
            fc_date_2020 = datetime.strptime(f"2020{fcdate[4:]}",'%Y%m%d')

            lags = [(c-fc_date_2020).days for c in JMA_rf_dates]
            closest = min(lags,key=abs) # get closest value
            
            # need to change lag if leap year. the lags are calculated with year 2020 in mind but it may be the case that the chosen forecasted year is not a leap year, hence day lags will be out by 1.
            
            fc_is_leap_year = calendar.isleap(datetime.strptime(fcdate,'%Y%m%d').year)
            leap_day_2020 = datetime(2020,2,29)
            
            if (closest is not None) and (not fc_is_leap_year):
                ref_dt = fc_date_2020 + timedelta(days=closest)
                if crosses_leap_day(ref_dt, fc_date_2020, leap_day_2020):
                    # shrink magnitude by 1
                    if closest < 0:
                        closest += 1
                    else:
                        closest -= 1

            return closest

def crosses_leap_day(ref_dt, anchor_dt, leap_day):
    """True if the closed-open interval between ref_dt and anchor_dt crosses leap_day."""
    # (ref <= leap_day < anchor) or (anchor <= leap_day < ref)
    return (ref_dt <= leap_day < anchor_dt) or (anchor_dt <= leap_day < ref_dt)

def two_Monday_Thursday_rfdates(fcdate):
    '''
    Given a fcdate, output the appropriate three reforecast start dates. Relevant for ECMWF operations between 27th June 2023 and 12th November 2024
    '''
    # create a list of dates +- 14 days around fcdate
    pos_dates = [fcdate + timedelta(days=delta) for delta in range(-14,15)]
    # keep only Mondays and Thursdays
    mon_thurs = [date for date in pos_dates if date.weekday() in (0,3)]
    days_delta = [(md-fcdate).days for md in mon_thurs]
    # then sort by abs value
    days_delta.sort(key=lambda x: (abs(x)))

    return days_delta[:2]


def get_hindcast_model_date(origin_id,fcdate):
    ''' Given origin_id, output appropriate date for reforecast dataset. This is the hindcast model version, not the set of reforecast dates.
    '''
    rf_model_freq = get_single_parameter(origin_id,fcdate,'rfModelFreq')
    rf_model_date = get_single_parameter(origin_id,fcdate,'rfModelDate')
    
    if rf_model_freq == "fixed":
        mrf_date = str(rf_model_date)
    else:
        mrf_date = fcdate

    return mrf_date
        
def get_hindcast_year_span(origin_id,fcdate):
    ''' Given origin_id, output appropriate set of years for reforecasts.
    '''
    rfType = get_single_parameter(origin_id,fcdate,'rfType')
    rf_yrs = get_single_parameter(origin_id,fcdate,'rfRange')

    if rfType == "fixed":
        start, end = rf_yrs
        rf_years = np.arange(start,end+1) # give full set of years, i.e. 1981, 1982, ..., 2013.
    elif rfType == "dynamic":
        fc_year = int(fcdate[:4]) # get year component of date.
        rf_years = np.arange(fc_year - int(rf_yrs),fc_year)
    else:
        raise ValueError(f"[ERROR] Couldn't compute appropriate reforecast years for origin_id '{origin_id}'.")

    return rf_years

def output_formatted_leadtimes(leadtime_hour, fcdate, variable, origin_id,period='1D',lag=0,fc_enslags=0):
    print(fc_enslags)

    # ------------ NORMALISE INPUTS ------------
    leadtime_hour_arr = np.atleast_1d(leadtime_hour).astype(int)
    fc_enslags_arr = np.atleast_1d(fc_enslags) if np.size(fc_enslags) else np.array([0])

    # ---------- SHIFTED FORECAST DATE ----------
    new_fcdate = datetime.strptime(fcdate, '%Y%m%d') + timedelta(days=float(lag))
    convert_fcdate = new_fcdate.strftime('%Y-%m-%d')

    # ---------- TIME RESOLUTION ----------
    time_resolution = get_timeresolution(variable)

    if time_resolution.endswith("6hrly"):
        step_hours = 6
    else:
        step_hours = 24

    # ---------- ALL POSSIBLE LEADTIMES ----------
    # full model grid for checking lagged lead times
    fclength = int(get_single_parameter(origin_id, fcdate, 'fcLength'))

    leadtime_hour_ALL = output_leadtime_hour(variable,origin_id,0,fclength,period,filter_accumulation=False,)

    if leadtime_hour_ALL.size == 0:
        raise ValueError("[ERROR] No leadtimes available.")

    grid_max = int(leadtime_hour_ALL.max())

    # Refine step from actual array
    if len(leadtime_hour_ALL) > 1:
        diffs = np.diff(leadtime_hour_ALL)
        pos = diffs[diffs > 0]
        if pos.size > 0:
            step_hours = int(pos.min())

    # ---------- MODEL-SPECIFIC MINIMUM VALID START ----------
    is_jma = (origin_id == "rjtd")
    is_instant_24 = (step_hours == 24 and not time_resolution.startswith("aver"))
    model_min_start_h = 24 if (is_jma and is_instant_24) else 0

    # ---------- USER VALID TIME WINDOW ----------
    VT_start_req = leadtime_hour_arr.min()
    VT_end_req   = leadtime_hour_arr.max()

    # Apply minimum valid start
    VT_start = max(VT_start_req, model_min_start_h)

    # ---------- COMMON VALID-TIME WINDOW FOR ALL LAGS ----------
    max_lag_days = np.max(np.abs(fc_enslags_arr.astype(int)))
    
    fclength_days = fclength // 24
    overlap_days = fclength_days - max_lag_days
    # Final valid-time endpoint available to every lag member
    overlap_end_hour = (fclength_days - max_lag_days) * 24
    
    if time_resolution.startswith("aver"):
        # Each lead time h represents the interval h to h + step_hours
        VT_filtered = leadtime_hour_arr[
            (leadtime_hour_arr >= VT_start)
            & ((leadtime_hour_arr + step_hours) <= overlap_end_hour)]
    else:
        VT_filtered = leadtime_hour_arr[
            (leadtime_hour_arr >= VT_start)
            & (leadtime_hour_arr <= overlap_end_hour)]
    
    if VT_filtered.size == 0:
        raise ValueError(
            "No requested valid times remain within the common "
            "lagged-ensemble window."
        )
    
    # Accumulated fields need the initial value for differencing
    if ('accumulated' in time_resolution and VT_filtered.size > 0 and VT_filtered[0] > 0):
        VT_filtered = np.insert(VT_filtered, 0, 0)

    # Convert common valid times to model lead times for this lag
    lag_offset_hours = abs(int(lag)) * 24
    aligned_leadtimes = VT_filtered + lag_offset_hours
    
    # ---------- STRICT GRID CHECK (Option B) ----------
    matched = np.intersect1d(aligned_leadtimes, leadtime_hour_ALL)
    if matched.size != aligned_leadtimes.size:
        missing = aligned_leadtimes[
            np.isin(aligned_leadtimes, leadtime_hour_ALL, invert=True)
        ]
        raise ValueError(
            f"[ERROR] One or more aligned leadtimes are not available:\n"
            f"Requested valid times: {leadtime_hour_arr}\n"
            f"Aligned leadtimes:     {aligned_leadtimes}\n"
            f"Variables inputted: {leadtime_hour} {fcdate} {variable} {origin_id} {lag} {fc_enslags}\n"
            f"Missing:               {missing}"
        )

    leadtime_hour_copy = matched

    # ---------- BUILD OUTPUT STRING ----------
    if time_resolution.startswith("aver"):
        leadtimes = "/".join(f"{h}-{h+24}" for h in leadtime_hour_copy)
    else:
        leadtimes = "/".join(str(int(h)) for h in leadtime_hour_copy)

    print(
        f"fcLength={fclength_days}d, "
        f"max_lag={max_lag_days}d, "
        f"overlap={overlap_days}d, "
        f"overlap_end={overlap_end_hour / 24}d"
    )
    
    return leadtimes, convert_fcdate

def create_reforecast_dates(rfyears,rfdate):
    ''' function that produces a list of reforecast dates given set of years and chosen reforecast date
    '''
    if np.size(rfdate) == 1: # for a single reforecast date that is then repeated for all reforecast years
        MM = rfdate[4:6]
        DD = rfdate[6:]
        # select all years
        rf_dates = ','.join(f'"{int(year)}-{MM}-{DD}"' for year in rfyears)
    else:
        stored_dates = []
        for date in rfdate:
            YY = date[:4]
            MM = date[4:6]
            DD = date[6:]
            stored_dates.append(f"{YY}-{MM}-{DD}")
        rf_dates = ','.join(f'"{date}"' for date in stored_dates) 
    return rf_dates
 
def check_and_output_all_fc_arguments(variable,model,fcdate,area,data_format,grid,plevs,leadtime_hour,start_lt,end_lt,period,fc_enslags):
    # check variable name. Is the variable name one of the abbreviations?
    argument_check.check_requested_variable(variable)
    # is it a sfc or pressure level field. # output sfc or level type
    level_type = output_sfc_or_plev(variable)

    # if level_type == plevs and plevs=None, output_plevs. Will only give troposphere for q. 
    # work out appropriate pressure levels
    if level_type == 'pressure':
        if plevs is None:
            plevs = output_plevs(variable)
        else:
            print (f"Downloading the requested pressure levels: {plevs}") # if not, use request plevs.
        # check plevs
        argument_check.check_plevs(plevs,variable)
    else:
        print (f"Downloading the following level type: {level_type}")
        plevs=None

    # get ECDS version of variable name. - WILL WRITE UP IN OCTOBER 2025!
    #ecds_varname = variable_output.output_ECDS_variable_name(variable)
    ecds_varname=None

    # get webapi param
    webapi_param = output_webapi_variable_name(variable) # temporary until move to ECDS (Aug - Oct).

    # check model is in acceptance list and get origin code!
    argument_check.check_model_name(model,fcdate)
    # get origin id
    origin_id = output_originID(model,fcdate)

    # get fc_enslags
    # get lagged ensemble details
    if fc_enslags is None:
        fc_enslags = output_fc_lags(origin_id,fcdate)
    # after gathering fc_enslags, check all ensemble lags are negative or zero and whole numbers as they can be user-inputted.
    argument_check.check_fc_enslags(fc_enslags)

    # if leadtime_hour = None, get leadtime_hour (output all hours).
    if leadtime_hour is None:
        if end_lt == None:
            end_lt = int(get_single_parameter(origin_id, fcdate, 'fcLength'))
        print (f'start_lt {start_lt}')
        leadtime_hour = output_leadtime_hour(variable,origin_id,start_lt,end_lt,period,filter_accumulation=True)
    else:
        leadtime_hour = np.array(leadtime_hour) # make leadtime hour an array
    print (f"For the following variable '{variable}' using the following leadtimes '{leadtime_hour}'.")

    # check leadtime_hours (as individuals can choose own leadtime_hours).
    argument_check.check_leadtime_hours(leadtime_hour,variable,origin_id,fcdate)

    # check fcdate.
    argument_check.check_fcdate(fcdate,origin_id)

    # check dataformat
    argument_check.check_dataformat(data_format)

    # check area selection
    argument_check.check_area_selection(area)

    return level_type, plevs, webapi_param, ecds_varname, origin_id, leadtime_hour, fc_enslags

def check_and_output_all_hc_arguments(variable,origin_id,fcdate,rfyears=None):
    ''' Function that will output all the necessary arguments to download reforecast data
    '''
    # get the date of the reforecast model
    rf_model_date = get_hindcast_model_date(origin_id,fcdate)

    # get the reforecast years
    if rfyears is None:
        rfyears = get_hindcast_year_span(origin_id,fcdate)
    # after computing reforecast years, check the chosen set
    argument_check.check_requested_reforecast_years(rfyears,origin_id,fcdate)

    return rf_model_date, rfyears


