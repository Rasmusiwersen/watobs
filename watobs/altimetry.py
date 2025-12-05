import calendar
import glob
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import copernicusmarine
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray

from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.dates as mdates
from watobs.cmems.utils import get_catalogue_info

logger = logging.getLogger(__name__)


class APIAuthenticationFailed(Exception):
    pass


class InvalidSatelliteName(Exception):
    pass


class AltimetryData:
    """Class returned by DHIAltimetryRepository's get_altimetry_data() method

    Examples
    ========
    >>> repo = DHIAltimetryRepository(api_key="...")
    >>> data = repo.get_altimetry_data("lon=10.9&lat=55.9&radius=10.0", start_time="2021")
    Succesfully retrieved 133 records from API in 0.69 seconds
    >>> data.satellites
    ['j3', '3a', 'c2', 'sa']
    >>> data.df.columns
    Index(['longitude', 'latitude', 'water_level', 'significant_wave_height',
       'wind_speed', 'distance_from_land', 'water_depth', 'satellite',
       'quality', 'absolute_dynamic_topography', 'water_level_rms',
       'significant_wave_height_raw', 'significant_wave_height_rms',
       'wind_speed_raw', 'wind_speed_rads'],
      dtype='object')
    >>> data.df.water_level.head(3)
    date
    2021-01-04 15:30:45.051    0.0531
    2021-01-04 15:30:46.070    0.0394
    2021-01-04 15:30:47.088    0.0294
    Name: water_level, dtype: float64
    >>> data.to_dfs0('alti_data.dfs0')
    """

    def __init__(self, df, area=None, query_params=None):
        self.df = df
        self.area = area
        self.query_params = query_params

    @property
    def satellites(self):
        """Satellites for this data"""
        return list(self.df.satellite.unique())

    @property
    def start_time(self):
        """Start time for this data"""
        return self.df.index[0]

    @property
    def end_time(self):
        """End time for this data"""
        return self.df.index[-1]

    @property
    def n_points(self):
        """Number of points in this dataset"""
        return len(self.df)

    def to_dfs0(self, filename, satellite=None, quality=0):
        """Save altimetry data to dfs0 file.

        Parameters
        ----------
        filename : str
            path to new dfs0 file
        satellite : str, optional
            short name of satellite to be saved, by default all
        quality : int, optional
            highest quality flag to include: 0=good, 1=acceptable, 2=bad,
            if 1 is given as argument data with flag 0 and 1 will be written to
            file, by default 0 (i.e. only good data)
        """
        from mikeio import eum

        df = self.df
        if satellite is not None:
            df = df[df.satellite == satellite]
        if quality is not None:
            df = df[df.quality <= quality]

        if len(df) < 1:
            raise Exception("No data in data frame")

        cols = [
            "longitude",
            "latitude",
            "water_level",
            "significant_wave_height",
            "wind_speed",
        ]
        items = []
        items.append(eum.ItemInfo("Longitude", eum.EUMType.Latitude_longitude))
        items.append(eum.ItemInfo("Latitude", eum.EUMType.Latitude_longitude))
        items.append(eum.ItemInfo("Water Level", eum.EUMType.Water_Level))
        items.append(
            eum.ItemInfo("Significant Wave Height", eum.EUMType.Significant_wave_height)
        )
        items.append(eum.ItemInfo("Wind Speed", eum.EUMType.Wind_speed))

        df[cols].to_dfs0(filename, items=items)

    def plot_map(self, fig_size=(9, 9), markersize=10):
        """plot map of altimetry data

        Parameters
        ----------
        fig_size : Tuple(float), optionally
            size of figure, by default (12,10)
        """
        df = self.df

        if "seaborn-whitegrid" in plt.style.available:
            plt.style.use("seaborn-whitegrid")
        plt.figure(figsize=fig_size)
        markers = ["o", "x", "+", "v", "^", "<", ">", "s", "d", ",", "."]
        j = 0
        for sat in self.satellites:
            dfsub = df[df.satellite == sat]
            plt.plot(
                dfsub.longitude,
                dfsub.latitude,
                markers[j],
                label=sat,
                markersize=markersize,
            )
            j = j + 1
        plt.legend(numpoints=1)
        plt.title(f"Altimetry data between {self.start_time} and {self.end_time}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

    @staticmethod
    def from_csv(filename):
        """read altimetry data from csv file instead of api

        Parameters
        ----------
        filename : str
            path to csv file

        Returns
        -------
        DataFrame
            With datetime index containing the altimetry data
        """
        df = pd.read_csv(filename, parse_dates=True, index_col="datetime")
        print(f"Succesfully read {len(df)} rows from file {filename}")
        return AltimetryData(df)

    def get_dataframe_per_satellite(self, df=None):
        if df is None:
            df = self.df
        res = {}
        sats = self.satellites
        for sat in sats:
            dfsub = df[df.satellite == sat]
            res[sat] = dfsub  # .drop(['satellite'], axis=1)
        return res

    def assign_track_id(self, data=None, max_jump=3.0, verbose=True):
        """Identify individual passings by finding gaps in data for each satellite.

        The track_id will be numbered 0, 1, ... for each satellite.

        Parameters
        ----------
        data : pd.DataFrame, optional
            altimetry data (assumed to have a "satellite" column), by default None
        max_jump : float, optional
            split passings if jump larger than this number of seconds, by default 3.0
        verbose : bool, optional
            print status information?, by default True

        Returns
        -------
        pd.DataFrame
            as input data but with a new column "track_id"
        """
        if data is None:
            data = self.df
        sats = self.satellites

        # 1 step (=1second = 7.2km)

        df = data.copy()

        if "track_id" not in df.columns:
            ids = np.zeros((len(df),), dtype=int)
            df.insert(len(df.columns), "track_id", ids, True)

        # find tracks for each satellite
        tot_tracks = 0
        for sat in sats:
            dfsub = df[df.satellite == sat]
            if len(dfsub) == 0:
                continue

            tt = dfsub.index
            tvec = (tt - tt[0]).total_seconds().values

            nt = len(tvec)
            dtvec = np.zeros(nt)
            dtvec[1:] = np.diff(tvec)

            ids = np.zeros(nt, dtype=int)
            ijump = np.where(dtvec > max_jump)
            ids[ijump] = 1
            ids = np.cumsum(ids)

            tot_tracks = tot_tracks + len(ijump) + 1
            df.loc[df.satellite == sat, "track_id"] = ids

        if verbose:
            print(f"Identified {tot_tracks} individual passings")

        return df

    def print_records_per_satellite(self, df=None, details=1):
        if df is None:
            df = self.df
        sats = self.satellites
        print(f"For the selected area between {self.start_time} and {self.end_time}:")
        for sat in sats:
            dfsub = df[df.satellite == sat]
            print(f"Satellite {sat} has {len(dfsub)} records")
            if details > 1:
                print(dfsub.drop(["longitude", "latitude"], axis=1).describe())


class _DHISatMixin:
    def get_observation_stats(self):
        """Get a summary of the data per satellite missions

        Returns
        -------
        pd.DataFrame
            min and max date and observation count per satellite
        """
        raise NotImplementedError("Subclasses must implement get_observation_stats()")

    def plot_observation_stats(self):
        """Plot graph showing temporal coverage for all satellites

        Examples
        --------
        >>> repo.plot_observation_stats()
        """
        df = self.get_observation_stats()[["min_date", "max_date"]]
        df = df.sort_values("min_date", ascending=False)

        nsats = len(df)
        ysize = max(2.0, 0.45 * nsats)
        figsize = (10, ysize)

        fig, ax = plt.subplots(figsize=figsize)
        y = np.repeat(0.0, 2)
        labels = []

        for row in df.itertuples():
            y += 1.0
            plt.plot([row.min_date, row.max_date], y)
            labels.append(row.Index)

        plt.yticks(np.arange(nsats) + 1, labels)

        end_date = datetime.now() + relativedelta(months=3)
        end_date = (
            end_date.replace(day=1) + relativedelta(months=1) - relativedelta(days=1)
        )
        yearly = pd.date_range(start="1984-1-1", end=end_date, freq="2AS")
        plt.xticks(yearly, labels=yearly.year)
        fmt_year = mdates.YearLocator()
        ax.xaxis.set_minor_locator(fmt_year)
        plt.grid(True, which="both")
        fig.autofmt_xdate()
        ax.set_xlim([df.min_date.min(), df.max_date.max()])
        ax.set_title("Satellite lifespan")
        return ax

    def _validate_area(self, area):
        # polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
        # bbox=115.0,28.5,150.2,52.1
        # lon=10.9&lat=55.9&radius=10.0
        parsed = False
        message = None
        if area[0:5] == "bbox=":
            if area.count(",") == 3:
                parsed = True
            else:
                message = "bbox area should be provided as bbox=115.0,28.5,150.2,52.1"

        elif area[0:8] == "polygon=":
            if area.count(",") >= 5:
                parsed = True
            else:
                message = "polygon area should be provided as polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993"

        elif area[0:4] == "lon=":
            if (area.count("&lat=") == 1) & (area.count("&radius=") == 1):
                parsed = True
            else:
                message = (
                    "circle area should be provided as lon=10.9&lat=55.9&radius=10.0"
                )
        else:
            message = "area must be given as bbox=115.0,28.5,150.2,52.1 or polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993 or lon=10.9&lat=55.9&radius=10.0"

        if not parsed:
            raise Exception(f"Failed to parse area {area}! {message}")
        return self._area_str_to_dict(area)

    @staticmethod
    def _parse_datetime(date: str | None, utc=None) -> pd.Timestamp | None:
        if date is None:
            return None

        return pd.to_datetime(date, format="ISO8601", utc=utc)

    @staticmethod
    def _area_str_to_dict(area):
        dd = {}
        for token in area.split("&"):
            key, val = token.split("=")
            dd[key] = val
        return dd


class DHIAltimetryRepository(_DHISatMixin):
    """Get altimetry observations from DHI

    Notes
    =====
    Get a API key by contacting https://www.dhi-gras.com/
    API documentation: https://altimetry-shop-data-api.dhigroup.com/apidoc

    Examples
    ========
    >>> repo = DHIAltimetryRepository(api_key="...")
    >>> repo.satellites
    ['gs',
     'e1',
     'tx',
     'pn',
     'e2',
     'g1',
     'j1',
     'n1',
     'j2',
     'c2',
     'sa',
     'j3',
     '3a',
     '3b']
    >>> df = repo.get_daily_count("lon=10.9&lat=55.9&radius=10.0", start_time="2021")
    >>> data = repo.get_altimetry_data("lon=10.9&lat=55.9&radius=10.0", start_time="2021")
    >>> data.to_dfs0('alti_data.dfs0')
    """

    API_URL = "https://altimetry-shop-data-api.dhigroup.com/"
    HEADERS = None
    api_key = None
    NA_VALUE = -9999.0

    def __init__(self, api_key):
        self.api_key = api_key
        self.HEADERS = {"authorization": api_key}
        self._api_conf = None
        self._satellites = None
        self._sat_long_names = None

    @property
    def _conf(self):
        if self._api_conf is None:
            self._api_conf = self._get_config()
        return self._api_conf

    def _get_config(self):
        r = requests.get(self.API_URL + "/config", headers=self.HEADERS)
        r.raise_for_status()
        return r.json()

    @property
    def satellites(self):
        """List of avaiable satellites (short names)"""
        if self._satellites is None:
            df = self.get_satellites()
            self._satellites = list(df.index)
            self._sat_long_names = df.long_name.values
        return self._satellites

    def get_satellites(self):
        """Get short and long names for available satellites

        Returns
        -------
        pd.DataFrame
            short and long satellite names
        """
        sats = self._conf.get("satellites")
        df = pd.DataFrame(sats).set_index("short_name")
        return df[["long_name"]]

    def get_quality_filters(self):
        """Get a list of available quality filters with descriptions.

        Returns
        -------
        pd.DataFrame
            available quality filters with descriptions
        """
        qf = self._conf.get("quality_filters")
        return pd.DataFrame(qf).set_index("short_name")

    def get_observation_stats(self):
        """Get a summary of the data per satellite missions

        Returns
        -------
        pd.DataFrame
            min and max date and observation count per satellite
        """
        r = requests.get(
            (self.API_URL + ("/observations-stats")),
            headers=self.HEADERS,
        )
        r.raise_for_status()
        stats = r.json()["stats"]
        df = pd.DataFrame(stats).set_index("short_name")
        df["min_date"] = pd.to_datetime(df["min_date"], format="ISO8601")
        df["max_date"] = pd.to_datetime(df["max_date"], format="ISO8601")
        return df

    def plot_observation_stats(self):
        """Plot graph showing temporal coverage for all satellites

        Examples
        --------
        >>> repo.plot_observation_stats()
        """
        import matplotlib.dates as mdates

        df = self.get_observation_stats()[["min_date", "max_date"]]
        df = df.sort_values("min_date", ascending=False)

        nsats = len(df)
        ysize = max(2.0, 0.45 * nsats)
        figsize = (10, ysize)

        fig, ax = plt.subplots(figsize=figsize)
        y = np.repeat(0.0, 2)
        labels = []

        for row in df.itertuples():
            y += 1.0
            plt.plot([row.min_date, row.max_date], y)
            labels.append(row.Index)

        plt.yticks(np.arange(nsats) + 1, labels)

        yearly = pd.date_range(start="1984-1-1", end="2026-1-1", freq="2AS")
        plt.xticks(yearly, labels=yearly.year)
        fmt_year = mdates.YearLocator()
        ax.xaxis.set_minor_locator(fmt_year)
        plt.grid(True, which="both")
        fig.autofmt_xdate()
        ax.set_xlim([df.min_date.min(), df.max_date.max()])
        ax.set_title("Satellite lifespan")
        return ax

    @property
    def time_of_newest_data(self):
        """Time of the latest data in the altimetry database."""
        df = self.get_observation_stats()[["max_date"]]
        return df.max_date.max()

    def get_daily_count(
        self, area, start_time="20200101", end_time=None, satellites=""
    ):
        """Get total number of daily observations for a given area

        Parameters
        ----------
        area : str
            area specification in one of three allowed formats:
                - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
                - bbox=115,28,150,52
                - lon=10.9&lat=55.9&radius=100
        start_time : str or datetime, optional
            start of time interval, by default "2020-01-01"
        end_time : str or datetime, optional
            end of time interval, by default datetime.now()
        satellites : str, optional
            Satellites to be downloaded, e.g. '', '3a', 'j3, by default '' (=all)

        Returns
        -------
        pd.DataFrame
            number of observations per day
        """
        url = self.API_URL + "temporal-coverage"
        payload = self._area_time_sat_payload(area, start_time, end_time, satellites)
        r = requests.get(url, params=payload, headers=self.HEADERS)
        if r.status_code != 200:
            print(r.text)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data["temporal_coverage"])
        df["date"] = pd.to_datetime(df["date"], format="ISO8601")
        return df.set_index("date")

    def get_spatial_coverage(
        self, area, start_time="20200101", end_time=None, satellites=""
    ):
        """Get spatial observation coverage as count per spatial bin in a
        rectangle covering the specified area

        Parameters
        ----------
        area : str
            area specification in one of three allowed formats:
                - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
                - bbox=115,28,150,52
                - lon=10.9&lat=55.9&radius=100
        start_time : str or datetime, optional
            start of time interval, by default "2020-01-01"
        end_time : str or datetime, optional
            end of time interval, by default datetime.now()
        satellites : str, optional
            Satellites to be downloaded, e.g. '', '3a', 'j3, by default '' (=all)

        Returns
        -------
        geopandas.GeoDataFrame
            count per spatial bin
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "The geopandas package is required by the get_spatial_coverage() method. Install it with 'conda install geopandas' or 'pip install geopandas'"
            )

        url = self.API_URL + "spatial-coverage"
        payload = self._area_time_sat_payload(area, start_time, end_time, satellites)
        r = requests.get(url, params=payload, headers=self.HEADERS)
        if r.status_code != 200:
            print(r.text)
        r.raise_for_status()
        data = r.json()
        if data and "coverage" in data:
            gdf = gpd.GeoDataFrame.from_features(data["coverage"], crs="epsg:4326")
            return gdf

    def get_altimetry_data(
        self,
        area,
        start_time="20200101",
        end_time=None,
        satellites="",
        qual_filters=None,
    ):
        """Main function that retrieves altimetry data from api

        Parameters
        ----------
        area : str
            String specifying location of desired data.  The three forms allowed by the API are:
                - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
                - bbox=115,28,150,52
                - lon=10.9&lat=55.9&radius=100
            A few named domains can also be used:
                - GS_NorthSea, GS_BalticSea, GS_SouthChinaSea
        start_time : str, datetime, optional
            Start of data to be retrieved, by default '20200101'
        end_time : str, datetime, optional
            End of data to be retrieved, by default datetime.now()
        satellites : str, list of str, optional
            Satellites to be downloaded, e.g. '', '3a', 'j3, by default ''
        qual_filters : int, list[int], optional
            Accepted qualities 0=god, 1=acceptable, 2=bad, e.g. [0, 1],
            by default None meaning no filter (=all data)

        Examples
        --------
        >>> repo = DHIAltimetryRepository(api_key="...")
        >>> data = repo.get_altimetry_data("lon=10.9&lat=55.9&radius=10.0", start_time="2021")
        Succesfully retrieved 133 records from API in 0.69 seconds

        Returns
        -------
        DataFrame
            With columns 'longitude', 'latitude', 'water_level', ...
        """
        if end_time is None:
            end_time = datetime.now()
        payload = self._create_query_payload(
            area=area,
            start_time=start_time,
            end_time=end_time,
            qual_filters=qual_filters,
            satellites=satellites,
        )
        df = self.get_altimetry_data_raw(payload)
        return AltimetryData(df, area=area, query_params=payload)

    def _area_time_sat_payload(
        self,
        area=None,
        start_time=None,
        end_time=None,
        satellites=None,
    ) -> dict:
        d = self._validate_area(area)

        start_time = self._parse_datetime(start_time)
        d["start_date"] = start_time.strftime("%Y%m%d")

        if end_time:
            end_time = self._parse_datetime(end_time)
        else:
            end_time = datetime.now()
        d["end_date"] = end_time.strftime("%Y%m%d")

        if start_time > end_time:
            raise ValueError(
                f"end time '{end_time}' must be greater than start time '{start_time}'!"
            )

        if satellites:
            satellites = self.parse_satellites(satellites)
            d["satellites"] = ",".join(satellites)
        return d

    # Create a query for satellite data as a URL pointing to the location of a CSV file with the data.

    # Parameters
    # ----------
    # area : str
    #     String specifying location of desired data.  The three forms allowed by the API are:
    #         - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
    #         - bbox=115,28,150,52
    #         - lon=10.9&lat=55.9&radius=100000
    #     A few named domains can also be used:
    #         - GS_NorthSea, GS_BalticSea, GS_SouthChinaSea
    # satellites : Union[List[str], str], optional
    #     List of short or long names of satellite to include, an empty string, or the string 'sentinels' to specify
    #         the two sentinel satellites 3a and 3b. Default: '3a'.
    # start_time : str or datetime, optional
    #     First date for which data is wanted, in the format '20100101' or as an empty string. If an empty string is
    #         given, data starting from when it was first available is returned. Default: ''.
    # end_time : str or datetime, optional
    #     Last date for which data is wanted, in the format '20100101' or as an empty string. If an empty string is
    #         given, data until the last time available is returned. Default: "20200101".
    # nan_value : str, optional
    #     Value to use to indicate bad or missing data, or an empty string to use the default (-9999). Default: ''.
    # qual_filters : int, list[int], optional
    #         Accepted qualities 0=god, 1=acceptable, 2=bad, e.g. [0, 1],
    #         by default None meaning no filter (=all data)
    # # numeric : bool, optional
    #     If True, return columns as numeric and return fewer columns in order to comply with the Met-Ocean on Demand
    #         analysis systems. If False, all columns are returned, and string types are preserved as such.
    #         Default: False.

    def _create_query_payload(
        self,
        area="bbox=-11.913345,48.592117,12.411167,63.084148",
        start_time="20200101",
        end_time=None,
        satellites="3a",
        nan_value=None,
        qual_filters=None,
        numeric=False,
    ) -> dict:
        d = self._area_time_sat_payload(area, start_time, end_time, satellites)
        if nan_value:
            d["nodata"] = nan_value
        if qual_filters is not None:
            qual_filters = (
                qual_filters if hasattr(qual_filters, "__len__") else [qual_filters]
            )
            if len(qual_filters) > 0:
                d["qual_filters"] = str(qual_filters).strip("[] ").replace(" ", "")
        if numeric:
            d["numeric"] = numeric
        return d

    def get_altimetry_data_raw(self, payload: dict) -> pd.DataFrame:
        """Request data from altimetry api

        Parameters
        ----------
        payload : dict
            params dict build with _create_query_payload()

        Raises
        ------
        APIAuthenticationFailed: if api key is wrong

        Returns
        -------
        pd.DataFrame
            with altimetry data
        """
        t_start = time.time()
        r = requests.get(
            self.API_URL + "query-csv",
            params=payload,
            headers=self.HEADERS,
        )
        if r.status_code == 400:
            print(r.text)
        if r.status_code == 401:
            raise APIAuthenticationFailed
        r.raise_for_status()
        response_data = r.json()
        if ("download_url" in response_data) and response_data["download_url"]:
            df = pd.read_csv(
                response_data["download_url"],
                parse_dates=True,
                index_col="datetime",
                na_values=self.NA_VALUE,
            )
        else:
            print("No data retrieved!")
            return None

        elapsed = time.time() - t_start
        nrecords = len(df)
        if nrecords > 0:
            print(
                f"Succesfully retrieved {nrecords} records from API in {elapsed:.2f} seconds"
            )
        else:
            print("No data retrieved!")

        return df

    def parse_satellites(self, satellites):
        """
        Parse a list of satellite names into an argument string to pass as part of a URL query.

        Parameters
        ----------
        satellites : Union[List[str], str]
            List of short or long names of satellite to include, an empty string, or the string 'sentinels' to specify
                the two sentinel satellites 3a and 3b.

        Returns
        -------
        str
            String representing argument specifying which satellites to retrieve data from.

        Raises
        --------
        InvalidSatelliteName
            If a string that is not the empty string, 'sentinels', or part of the following lists is passed:
            ['TOPEX', 'Poseidon', 'Jason-1', 'Envisat', 'Jason-2', 'SARAL', 'Jason-3',
                                'Geosat', 'GFO', 'ERS-1', 'ERS-2', 'CryoSat-2', 'Sentinel-3A', 'Sentinel-3B']
            ['tx', 'ps', 'j1', 'n1', 'j2', 'sa', 'j3', 'gs', 'g1', 'e1', 'e2', 'c2', '3a', '3b']
        """
        if not satellites:
            return ""
        if isinstance(satellites, str):
            satellites = [satellites]

        sat_short_names = self.satellites
        sat_long_names = self._sat_long_names

        satellite_dict = dict(zip(sat_long_names, sat_short_names))

        satellite_strings = []
        for sat in satellites:
            if sat in sat_short_names:
                satellite_strings.append(sat)
            elif sat in sat_long_names:
                satellite_strings.append(satellite_dict[sat])
            else:
                raise InvalidSatelliteName("Invalid satellite name: " + sat)
        return satellite_strings


class CMEMSSatObsRepository(_DHISatMixin):
    # To be done:
    # - Merge the four functions that converts CMEMS to df. Especially the .nc reading can be condensed.

    def __init__(
        self,
        *,
        product_id: str | None = None,
        dataset_id: str | None = None,
        start_time=None,
        end_time=None,
        area=None,
    ):
        self._validate_set_product_id(product_id, dataset_id)
        self.start_time = start_time
        self.end_time = end_time
        self.area = area

    def _validate_set_product_id(
        self, product_id: str | None, dataset_id: str | None = None
    ):
        if product_id is None and dataset_id is None:
            raise ValueError("Either product_id or dataset_id must be provided!")

        if product_id is not None:
            print(f"Loading CMEMS product id: {product_id}")
            self.catalogue = copernicusmarine.describe(
                product_id=product_id, disable_progress_bar=True
            )
            print("\tDone!")
        elif dataset_id is not None:
            print(f"Loading CMEMS dataset id: {dataset_id}")
            self.catalogue = copernicusmarine.describe(
                dataset_id=dataset_id, disable_progress_bar=True
            )
            print("\tDone!")

    @staticmethod
    def validate_login(
        username: Optional[str] = None,
        password: Optional[str] = None,
        credentials_file: Optional[Path] = None,
    ):
        try:
            _, _ = (
                copernicusmarine.core_functions.credentials_utils.get_and_check_username_password(
                    username, password, credentials_file
                )
            )
            return True
        except copernicusmarine.InvalidUsernameOrPassword:
            return False

    def get_satobs_data(self, start_time=None, end_time=None, area=None, filter=None):
        """Main function that retrieves data from CMEMS through api requests"""

        tmpdir = tempfile.TemporaryDirectory(prefix=".cmems_", dir=os.getcwd())

        def download_2_df(dataset_id, start_time, end_time, area, filter, tmpdir):
            if filter is None:
                # Validate start/end time
                start_time_ = self._parse_datetime(start_time, utc=True)
                end_time_ = self._parse_datetime(end_time, utc=True)

                dataset_meta = self.datasets.loc[dataset_id]
                if (
                    dataset_meta.min_date is not None
                    and dataset_meta.max_date is not None
                ):
                    if end_time_ < dataset_meta.min_date:
                        print(
                            "Warning: end_time is before the dataset's minimum date; no data."
                        )
                        return None
                    if start_time_ > dataset_meta.max_date:
                        print(
                            "Warning: start_time is after the dataset's maximum date; no data."
                        )
                        return None
                    if not start_time_ >= dataset_meta.min_date:
                        print(
                            "Warning: start_time is before the dataset's minimum date."
                        )
                        start_time = dataset_meta.min_date.tz_convert(None)
                    if not end_time_ <= dataset_meta.max_date:
                        print("Warning: end_time is after the dataset's maximum date.")
                        end_time = dataset_meta.max_date.tz_convert(None)
                else:
                    print(
                        "Warning: Dataset has no min/max date info; proceeding with given time range."
                    )
                # Sub-function to download and convert to dataframe
                temp_dir, file_path = self._download_copernicus_data(
                    dataset_id, start_time, end_time, area, filter, tmpdir
                )
            else:
                # Sub-function to download and convert to dataframe
                temp_dir, file_path = self._download_copernicus_data(
                    dataset_id, None, None, None, filter, tmpdir
                )

            # If download is successful, convert to dataframe... else emtpy
            if temp_dir is not None and file_path is not None:
                df = self._cmems_format_raw_data(temp_dir, file_path)
                return df.dropna(how="all")

            else:
                return None

        try:
            # For multiple datasets at once
            df_lst = []

            dataset_ids = list(self.dataset_ids)

            for i, dataset_id in enumerate(dataset_ids):
                print(f"Downloading dataset {i+1} of {len(dataset_ids)}: {dataset_id}")
                ## Download data
                df = download_2_df(
                    dataset_id, start_time, end_time, area, filter, tmpdir
                )
                if df is None:
                    continue
                ## Add meta data
                dataset_meta = self.datasets.loc[dataset_id]
                meta_satellite = dataset_meta.short_name
                if dataset_meta.asc_desc is not None:  # Waves do not incl asc/desc
                    meta_satellite += "-" + dataset_meta.asc_desc
                df["satellite"] = meta_satellite
                df_lst.append(df)

            df = pd.concat(df_lst)
            df = df.dropna(how="all")

            tmpdir.cleanup()

            # Finally, convert to AltimetryData object
            return AltimetryData(df, area=area)
        except Exception as e:
            print(f"Error obtaining data! {e}")
            tmpdir.cleanup()
            return None

    def _download_copernicus_data(
        self,
        dataset_id=None,
        start_time=None,
        end_time=None,
        area=None,
        filter=None,
        tmpdir=None,
    ):
        """Main function that retrieves data from CMEMS through api requests

        Parameters
        ----------
        dataset_id : str
            String specifying the dataset to be downloaded, e.g. 'cmems_obs-wind_glo_phy_nrt_l3-hy2b-hscat-asc-0.25deg_P1D-i'
        area : str
            String specifying location of desired data.  The three forms allowed by the API are:
                - polygon=6.811,54.993,8.009,54.993,8.009,57.154,6.811,57.154,6.811,54.993
                - bbox=115,28,150,52
                - lon=10.9&lat=55.9&radius=100
        start_time : str, datetime, optional
            Start of data to be retrieved, by default '20200101'
        end_time : str, datetime, optional
            End of data to be retrieved, by default datetime.now()
        satellites : str, list of str, optional
            Satellites to be downloaded, e.g. '', '3a', 'j3, by default ''
        qual_filters : int, list[int], optional
            Accepted qualities 0=god, 1=acceptable, 2=bad, e.g. [0, 1],
            by default None meaning no filter (=all data)
        filter : str, optional
            Filter string to specify subset of data, by default None
        Examples
        --------
        >>> repo = DHIAltimetryRepository(api_key="...")
        >>> data = repo.get_altimetry_data("lon=10.9&lat=55.9&radius=10.0", start_time="2021")
        Succesfully retrieved 133 records from API in 0.69 seconds

        Returns
        -------
        DataFrame
            With columns 'longitude', 'latitude', 'water_level', ...
        """

        if area is not None:
            d = self._validate_area(area)
        else:
            d = {"start_date": None, "end_date": None}

        if start_time is not None:
            start_time = self._parse_datetime(start_time)

        if end_time:
            end_time = self._parse_datetime(end_time)
        else:
            end_time = datetime.now()

        if start_time is not None and end_time is not None:
            if start_time > end_time:
                raise ValueError(
                    f"end time '{end_time}' must be greater than start time '{start_time}'!"
                )

        # Handle area parsing
        global_flag = True

        if area is not None:
            global_flag = False
            parsed = False

            dd = self._area_str_to_dict(area)
            ar_split = dd["bbox"].split(",")

            # Validate bbox coordinates
            try:
                bbox_coords = [float(x) for x in ar_split]
                if len(bbox_coords) != 4:
                    raise ValueError(f"Expected 4 coordinates, got {len(bbox_coords)}")
            except ValueError as e:
                raise ValueError(f"Invalid bbox coordinates: {e}")

        # Create a temporary directory
        temp_dir = Path(tmpdir.name)
        before = set(os.listdir(temp_dir))

        def cmems_get_filter(dataset_id, temp_dir, filter, download_str):
            try:
                copernicusmarine.get(
                    dataset_id=dataset_id,
                    output_directory=str(temp_dir),
                    # filter=f"*/{year}/{month:02d}/*",
                    filter=filter,
                )
            except Exception as download_error:
                logger.error(f"Failed to download {download_str}: {download_error}")
                raise

        # Download logic
        if not global_flag:
            try:
                print(f"-- Downloading subset of {dataset_id}")
                copernicusmarine.subset(
                    dataset_id=dataset_id,
                    start_datetime=start_time.strftime("%Y-%m-%d"),
                    end_datetime=end_time.strftime("%Y-%m-%d"),
                    minimum_longitude=bbox_coords[0],
                    maximum_longitude=bbox_coords[2],
                    minimum_latitude=bbox_coords[1],
                    maximum_latitude=bbox_coords[3],
                    output_directory=str(temp_dir),
                )
                print("-- Download successful.")
            except Exception as e:
                logger.warning(
                    f"Subset download failed: {e}. Falling back to full dataset download."
                )
                # Generate proper date ranges instead of month numbers
                date_range = pd.date_range(start=start_time, end=end_time, freq="MS")
                for period in date_range:
                    year, month = period.year, period.month
                    month_name = calendar.month_name[month]
                    print(f"-- Downloading {month_name} {year}")
                    filter = f"*/{year}/{month:02d}/*"
                    cmems_get_filter(
                        dataset_id, temp_dir, filter, f"{month_name} {year}"
                    )

                print("-- Download successful.")
        else:
            if filter is None:
                # Generate proper date ranges
                date_range = pd.date_range(start=start_time, end=end_time, freq="MS")
                for period in date_range:
                    year, month = period.year, period.month
                    month_name = calendar.month_name[month]
                    print(f"- Downloading {month_name} {year}")
                    filter = f"*/{year}/{month:02d}/*"
                    cmems_get_filter(
                        dataset_id, temp_dir, filter, f"{month_name} {year}"
                    )
            else:
                cmems_get_filter(dataset_id, temp_dir, filter, filter)

        after = set(os.listdir(temp_dir))
        downloaded = after - before
        if len(downloaded) == 1:
            file_path = temp_dir / list(downloaded)[0]
        elif len(downloaded) > 1:
            logger.warning("Multiple files downloaded, using directory path")
            file_path = temp_dir
        elif len(downloaded) == 0:
            logger.warning("Empty download, no files found")
            return None, None

        return temp_dir, file_path

    def get_var_float64(self, f, varname):
        data = f[varname].astype(np.float64)
        scale_factor = f[varname].attrs.get("scale_factor", 1.0)
        add_offset = f[varname].attrs.get("add_offset", 0.0)
        missing_value = f[varname].attrs.get("missing_value", None)
        if missing_value is not None:
            data = data.where(data != missing_value, np.nan)
        data = data * scale_factor + add_offset
        data = data.values.astype(np.float64).squeeze()
        return data

    def cmems_subset_wind_nc_to_df(self, file):
        try:
            f = xarray.open_dataset(file, decode_cf=False)
        except FileNotFoundError:
            print(f"File not found: {file}")
            df = pd.DataFrame()
            return df
        except OSError as e:
            print(f"Error opening file: {e}")
            df = pd.DataFrame()
            return df
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            df = pd.DataFrame()
            return df
        #
        try:
            lon = f.longitude.values
            lat = f.latitude.values
        except (AttributeError, KeyError):
            # Fall back to alternative coordinate names
            lon = f.lon.values
            lat = f.lat.values
        xlon, xlat = np.meshgrid(lon, lat)
        time = self.get_var_float64(f, "measurement_time")
        ws = self.get_var_float64(f, "wind_speed")
        wd = self.get_var_float64(f, "wind_to_dir")
        uwnd = self.get_var_float64(f, "eastward_wind")
        vwnd = self.get_var_float64(f, "northward_wind")
        bs_date_unit = f.measurement_time.attrs["units"]
        if not bs_date_unit.endswith("00:00:00"):
            bs_date_unit = bs_date_unit + " 00:00:00"
        base_date = pd.to_datetime(
            bs_date_unit, format="seconds since %Y-%m-%d %H:%M:%S"
        )

        lat_idx, lon_idx = np.meshgrid(range(len(lat)), range(len(lon)), indexing="ij")
        lat_idx = np.tile(lat_idx.flatten(), len(time))
        lon_idx = np.tile(lon_idx.flatten(), len(time))

        # Difference in how longitude and latitude is structure in global and subset (local) files, hence this piece of code
        try:
            # Build DataFrame
            df = pd.DataFrame(
                {
                    "time": time.flatten(),
                    "longitude": lon[lon_idx],
                    "latitude": lat[lat_idx],
                    "WS": ws.flatten(),
                    "WD": wd.flatten(),
                    "U10": uwnd.flatten(),
                    "V10": vwnd.flatten(),
                }
            )
        except (IndexError, ValueError) as e:
            # Use meshgrid coordinates for global files
            logger.debug(f"Using meshgrid coordinates: {e}")
            df = pd.DataFrame(
                {
                    "time": time.flatten(),
                    "longitude": xlon.flatten(),
                    "latitude": xlat.flatten(),
                    "WS": ws.flatten(),
                    "WD": wd.flatten(),
                    "U10": uwnd.flatten(),
                    "V10": vwnd.flatten(),
                }
            )

        time_deltas = pd.to_timedelta(df["time"], unit="s")
        df["time"] = base_date + time_deltas  # base_date+time_deltas
        df = df.set_index("time")
        return df

    def cmems_glo_wind_nc_to_df(self, file):
        try:
            f = xarray.open_dataset(file, decode_cf=False)
        except FileNotFoundError:
            print(f"File not found: {file}")
            df = pd.DataFrame()
            return df
        except OSError as e:
            print(f"Error opening file: {e}")
            df = pd.DataFrame()
            return df
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            df = pd.DataFrame()
            return df
        lon = f.lon.values
        lat = f.lat.values
        xlon, xlat = np.meshgrid(lon, lat)
        time = self.get_var_float64(f, "measurement_time")
        ws = self.get_var_float64(f, "wind_speed")
        wd = self.get_var_float64(f, "wind_to_dir")
        uwnd = self.get_var_float64(f, "eastward_wind")
        vwnd = self.get_var_float64(f, "northward_wind")
        df = pd.DataFrame(
            {
                "time": time.flatten(),
                "longitude": xlon.flatten(),
                "latitude": xlat.flatten(),
                "WS": ws.flatten(),
                "WD": wd.flatten(),
                "U10": uwnd.flatten(),
                "V10": vwnd.flatten(),
            }
        )
        df = df[df["WS"] >= 0.0]
        df = df.reset_index(drop=True)
        base_date = pd.to_datetime(
            f.measurement_time.attrs["units"], format="seconds since %Y-%m-%d %H:%M:%S"
        )
        time_deltas = pd.to_timedelta(df["time"], unit="s")
        df["time"] = base_date + time_deltas
        df = df.set_index("time")
        return df

    def cmems_wave_csv_to_df(self, file):
        try:
            with open(file, "rb") as f:
                print(f)
                data = pd.read_csv(f)
        except FileNotFoundError:
            print(f"File not found: {file}")
            df = pd.DataFrame()
            return df
        except OSError as e:
            print(f"Error opening file: {e}")
            df = pd.DataFrame()
            return df
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            df = pd.DataFrame()
            return df
        # Converting CMEMS WAVE .csv to rightly formatted .csv
        df = data[data.variable == "VAVH"]

        df.set_index("time", inplace=True)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.strftime("%Y-%m-%d %H:%M:%S")
        df = df.rename(columns={"value": "SWH"})
        df = df.drop(
            columns=[
                "is_depth_from_producer",
                "variable",
                "platform_id",
                "platform_type",
                "doi",
                "product_doi",
                "pressure",
                "depth",
                "institution",
            ]
        )
        return df

    def cmems_glo_wave_to_df(self, file):
        try:
            f = xarray.open_dataset(file, decode_times=False)
        except FileNotFoundError:
            print(f"File not found: {file}")
            df = pd.DataFrame()
            return df
        except OSError as e:
            print(f"Error opening file: {e}")
            df = pd.DataFrame()
            return df
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            df = pd.DataFrame()
            return df
        #
        df = f.to_pandas()
        df = df.reset_index()

        base_date = pd.to_datetime(f.first_meas_time[:19], format="%Y-%m-%d %H:%M:%S")

        time_deltas = pd.to_timedelta(df.index - df.index[0], unit="s")
        df["time"] = base_date + time_deltas
        df = df.set_index("time")
        df = df.rename(columns={"VAVH": "SWH", "VAVH_UNFILTERED": "SWH_UNFILTERED"})
        return df

    def cmems_glo_specwave_to_df(self, file):
        try:
            f = xarray.open_dataset(file, decode_times=False)
        except FileNotFoundError:
            print(f"File not found: {file}")
            df = pd.DataFrame()
            return df
        except OSError as e:
            print(f"Error opening file: {e}")
            df = pd.DataFrame()
            return df
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            df = pd.DataFrame()
            return df

        # Multi-dim to single dums
        data_dict = {}
        var_mapping = {"VAVH": "SWH", "VTPK": "TP", "VPED": "PWD"}

        # Get time axis
        idx = f["time"].mean(dim="side_looking").values
        idx = pd.to_timedelta(idx, unit="d").round("1s") + pd.to_datetime(
            "1950-01-01T00:00:00"
        )
        df_concat = []
        for side in f["side_looking"].values:
            data_side = {}
            lon = f["longitude"].sel(side_looking=side).to_series()
            lat = f["latitude"].sel(side_looking=side).to_series()
            data_side["longitude"] = lon
            data_side["latitude"] = lat
            for var, var_r in var_mapping.items():
                data_side[f"{var_r}"] = f[var].sel(side_looking=side).to_series()

            df_side = pd.concat(data_side, axis=1)
            df_side.index = idx
            df_concat.append(df_side)
        df = pd.concat(df_concat)

        # To dataframe
        df = df.dropna(how="all", subset=["SWH", "TP", "PWD"])

        return df

    def _cmems_format_raw_data(self, temp_dir, file_path):
        print("Formatting data to DataFrame...")
        temp_dir = Path(temp_dir)
        file_path = Path(file_path)

        if file_path.is_file():
            if file_path.suffix == ".csv":
                df = self.cmems_wave_csv_to_df(file_path)

            elif (
                file_path.suffix == ".nc" in str(file_path)
                and "wind" in str(file_path).lower()
            ):  # Maybe this needs to change, right now only wind files are .nc
                df = self.cmems_subset_wind_nc_to_df(file_path)

            elif (
                file_path.suffix == ".nc" in str(file_path)
                and "swh" in str(file_path).lower()
            ):  # Maybe this needs to change, right now only wind files are .nc
                df = self.cmems_glo_wave_to_df(file_path)

            else:
                raise ValueError("Product unknown")
            df.to_csv(os.path.join(file_path))

        else:
            for ds in os.listdir(file_path):
                ds_path = file_path / ds
                years = os.listdir(ds_path)
                df = pd.DataFrame()
                for year in years:
                    print(f"Merging files for {year}")
                    year_path = ds_path / year
                    files = list(year_path.glob("**/*.nc"))
                    cfo = pd.DataFrame()

                    for file in files:
                        file_path_name = file_path.name.upper()

                        if cfo.empty:
                            if "WIND" in file_path_name:
                                cfo = self.cmems_glo_wind_nc_to_df(file)
                            elif (
                                "WAVE" in file_path_name and "SPC" not in file_path_name
                            ):
                                cfo = self.cmems_glo_wave_to_df(file)
                            elif "WAVE" in file_path_name and "SPC" in file_path_name:
                                cfo = self.cmems_glo_specwave_to_df(file)
                            else:
                                logger.error(f"Unknown product type: {file_path_name}")
                                raise ValueError(
                                    f"Unknown product type: {file_path_name}"
                                )
                        else:
                            if "WIND" in file_path_name:
                                sing_df = self.cmems_glo_wind_nc_to_df(file)
                            elif (
                                "WAVE" in file_path_name and "SPC" not in file_path_name
                            ):
                                sing_df = self.cmems_glo_wave_to_df(file)
                            elif "WAVE" in file_path_name and "SPC" in file_path_name:
                                sing_df = self.cmems_glo_specwave_to_df(file)
                            else:
                                logger.error(f"Unknown product type: {file_path_name}")
                                raise ValueError(
                                    f"Unknown product type: {file_path_name}"
                                )

                            if sing_df.empty:
                                logger.warning(f"Empty data from {file.name}, skipping")
                                continue
                            else:
                                cfo = pd.concat([cfo, sing_df], axis=0)
                                logger.debug(f"Reading {file.name}")
                    df = pd.concat([df, cfo], axis=0)

        print("\tDone!")
        return df

    def get_observation_stats(self):
        df = (
            self.datasets.groupby("short_name").first().loc[:, ["min_date", "max_date"]]
        )
        return df

    @property
    def datasets(self):
        df = get_catalogue_info(self.catalogue)

        if "asc_desc" in df.columns:
            keys = ["short_name", "asc_desc"]
        else:
            keys = ["short_name"]

        # For now, just get highest resolution
        if "spatial_resolution" not in df.columns:
            return df
        else:
            if len(df["spatial_resolution"].dropna()) == 0:
                return df
            df_tmp = df.copy()
            df_tmp["spatial_resolution"] = [
                float(res.split("deg")[0]) for res in df["spatial_resolution"]
            ]
            idx = df_tmp.groupby(keys)["spatial_resolution"].idxmin()
            df = df.loc[idx]
            return df

    @property
    def dataset_ids(self):
        return list(self.datasets.index.values)
