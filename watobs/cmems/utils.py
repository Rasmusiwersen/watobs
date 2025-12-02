"""Copied from https://help.marine.copernicus.eu/en/articles/8286798-copernicus-marine-toolbox-api-explore-the-catalogue-and-metadata"""

from datetime import datetime, timezone
from typing import Any, Optional, Tuple
import re
import pandas as pd
import copernicusmarine


def convert_to_unix_seconds(value: Any) -> float:
    """
    Convert various time representations to UNIX timestamp in seconds (UTC).
    Supports: numeric (s/ms/ns), ISO-8601 strings, datetime objects.
    """
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1e16:  # nanoseconds since epoch
            return timestamp / 1e9
        if timestamp > 1e11:  # milliseconds since epoch
            return timestamp / 1e3
        return timestamp  # seconds since epoch

    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()

    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).timestamp()

    raise TypeError(f"Unsupported datetime value type: {type(value)!r}")


def select_version(dataset, version_label: Optional[str]):
    """Return the matching dataset version, or the default one if none is specified."""
    if version_label is None:
        return dataset.versions[0]

    for version in dataset.versions:
        if version.label == version_label:
            return version

    available = [v.label for v in dataset.versions]
    raise ValueError(f"Version {version_label!r} not found. Available: {available}")


def select_part(version, part_name: Optional[str]):
    """Return the matching dataset part, or the default one if none is specified."""
    if part_name is None:
        return version.parts[0]

    for part in version.parts:
        if part.name == part_name:
            return part

    available = [p.name for p in version.parts]
    raise ValueError(
        f"Part {part_name!r} not found in version {version.label!r}. Available: {available}"
    )


def find_time_coordinate(part) -> Any:
    """
    Return the coordinate object representing time from a dataset part.
    Raises RuntimeError if no such coordinate is found.
    """
    coordinates = part.get_coordinates()

    for coord_id, (coord_obj, _var_ids, _svc_names) in coordinates.items():
        is_time_axis = (
            getattr(coord_obj, "axis", None) == "t"
            or str(coord_id).lower() == "time"
            or str(getattr(coord_obj, "standard_name", "")).lower() == "time"
        )
        if is_time_axis:
            return coord_obj

    raise RuntimeError(
        f"No time coordinate found in part {part.name!r} of version {getattr(part, 'version_label', '?')}."
    )


def get_time_coverage(
    dataset_identifier: str,
    dataset_version: Optional[str] = None,
    dataset_part: Optional[str] = None,
) -> Tuple[datetime, datetime]:
    """
    Return the (start_time, end_time) for a dataset version/part as timezone-aware UTC datetimes.
    """
    catalog_entry = copernicusmarine.describe(
        dataset_id=dataset_identifier,
        disable_progress_bar=True,
    )

    dataset = catalog_entry.products[0].datasets[0]
    version = select_version(dataset, dataset_version)
    part = select_part(version, dataset_part)
    time_coord = find_time_coordinate(part)

    start_sec = convert_to_unix_seconds(time_coord.minimum_value)
    end_sec = convert_to_unix_seconds(time_coord.maximum_value)

    start_dt = datetime.fromtimestamp(start_sec, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(end_sec, tz=timezone.utc)

    return start_dt, end_dt


def print_time_coverage(
    dataset_identifier: str,
    dataset_version: Optional[str] = None,
    dataset_part: Optional[str] = None,
):
    """Helper to print start and end time coverage for a dataset."""
    start_dt, end_dt = get_time_coverage(
        dataset_identifier, dataset_version, dataset_part
    )
    print("Start time:", start_dt.isoformat())
    print("End time:  ", end_dt.isoformat())


# ----------------------------
# DHI Extended methods
# ----------------------------


def parse_dataset_id(dataset):
    name_list = re.split("_|-", dataset.dataset_id)

    source = name_list[0]
    source_type = name_list[1]
    obs_type = name_list[2]
    coverage = name_list[3]
    ocean_category = name_list[4]
    if obs_type == "wave":
        param = name_list[5]
        multiyear_or_nearrealtime = name_list[6]
        short_name = name_list[7]
        processing_level = name_list[8]
        asc_desc = None
        spatial_resolution = None
        temporal_resolution = name_list[9]
        if len(name_list) > 10:  # some datasets have temporal type
            temporal_type = name_list[10]
        else:
            temporal_type = None
    elif obs_type == "wind":
        param = None
        multiyear_or_nearrealtime = name_list[5]
        processing_level = name_list[6]
        short_name = name_list[7]
        # orbit_type = name_list[8]
        asc_desc = name_list[9]
        spatial_resolution = name_list[10]
        temporal_resolution = name_list[11]
        temporal_type = name_list[12]

    name_dict = {
        "source": source,
        "source_type": source_type,
        "obs_type": obs_type,
        "coverage": coverage,
        "ocean_category": ocean_category,
        "param": param,
        "multiyear_or_nearrealtime": multiyear_or_nearrealtime,
        "short_name": short_name,
        "processing_level": processing_level,
        "asc_desc": asc_desc,
        "spatial_resolution": spatial_resolution,
        "temporal_resolution": temporal_resolution,
        "temporal_type": temporal_type,
    }

    return name_dict


def parse_datetime(time_coord_value):
    time_sec = convert_to_unix_seconds(time_coord_value)
    return datetime.fromtimestamp(time_sec, tz=timezone.utc)


def get_catalogue_info(catalogue) -> dict:
    """Get CMEMS catalogue stats for altimetry and winds datasets."""
    dct_coverage = {}

    for product in catalogue.products:
        for dataset in product.datasets:
            for version in dataset.versions:
                for part in version.parts:
                    time_coord = find_time_coordinate(part)
                    name_dict = parse_dataset_id(dataset)

                    # if name_dict["obs_type"] not in ["wave", "wind"]:
                    #     continue

                    # Not interested in level other than L3
                    # if name_dict["processing_level"] != "l3":
                    #     continue

                    # key = dataset.dataset_id

                    dct_coverage[dataset.dataset_id] = name_dict
                    dct_coverage[dataset.dataset_id]["min_date"] = parse_datetime(
                        time_coord.minimum_value
                    )
                    dct_coverage[dataset.dataset_id]["max_date"] = parse_datetime(
                        time_coord.maximum_value
                    )

    df_coverage = pd.DataFrame.from_dict(dct_coverage, orient="index")
    df_coverage.index.name = "dataset_id"

    return df_coverage
