from tsfresh.feature_extraction.settings import from_columns
from typing import List, Tuple
from itertools import combinations
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import List
from md2pdf.core import md2pdf
from collections import defaultdict
import sys
from tsfresh.feature_extraction import feature_calculators
from tsfresh.utilities.string_manipulation import get_config_from_string




def generate_feature_name_encoding(feature_names):
    """
    returns a dictionary which maps encoded feature
    names to their long full feature names 
    """
    pass

def clean_feature_timeseries_name(feature_timeseries_name:str, window_length: int) -> str:
    """
    Logic to clean up the feature time series name after the first round of extraction

    NB: This might not be sufficient but use this for now
    """

    return feature_timeseries_name.replace("__", "||", 1).replace("__", "|") + f"@window_{window_length}"


def parse_feature_dynamic_name(feature_dynamic_full_name):
    window_length_token = "@"
    ts_kind_token = "||"
    param_token = "|"
    print(feature_dynamic_full_name)


    cleaned_feature_dynamic_name = feature_dynamic_full_name.replace(ts_kind_token, "__").replace(param_token, "__").replace(window_length_token, "__")
    parts = cleaned_feature_dynamic_name.split("__")
    print(parts)

    window_length = feature_dynamic_full_name.split(window_length_token)[1].split("__")[0]
    ts_kind = feature_dynamic_full_name.split(ts_kind_token)[0]
    feature_timeseries_calculator_name = feature_dynamic_full_name.split(ts_kind_token)[1].split(window_length_token)[0]
    feature_timeseries_calculator_params = 0
    feature_dynamic_calculator_name = 0
    feature_dynamic_calculator_params = 0  


    return {
        "feature_dynamic_full_name" : feature_dynamic_full_name,
        "window_length" : window_length,
        "ts_kind" : ts_kind,
        "feature_timeseries_calculator_name": feature_timeseries_calculator_name,
        "feature_timeseries_calculator_params": feature_timeseries_calculator_params,
        "feature_dynamic_calculator_name": feature_dynamic_calculator_name,
        "feature_dynamic_calculator_params": feature_dynamic_calculator_params,
    }


def add_to_feature_dictionary_with_windows(feature_dictionary, window_length, feature_parts):
    """
    Assume parts is kind, feature_name, feature_params
    Adds entries to a feature calculator dictionary
    """

    # Data validation stuff
    # <insert validaiton here>

    # Splitting into the right stuff
    ts_kind = feature_parts[0]
    feature_name = feature_parts[1]

    if window_length not in feature_dictionary:
        feature_dictionary[window_length] = {}

    if ts_kind not in feature_dictionary[window_length]:
        feature_dictionary[window_length][ts_kind] = {} 

    if not hasattr(feature_calculators, feature_name):
        raise ValueError("Unknown feature name {}".format(feature_name))

    config = get_config_from_string(feature_parts)
    if config:
        if feature_name in feature_dictionary[window_length][ts_kind]:
            if config not in feature_dictionary[window_length][ts_kind][feature_name]:
                feature_dictionary[window_length][ts_kind][feature_name].append(config)
        else:
            feature_dictionary[window_length][ts_kind][feature_name] = [config]
    else:
        feature_dictionary[window_length][ts_kind][feature_name] = None

    return feature_dictionary
    


def derive_features_dictionaries(feature_names: List[str]) -> Tuple[dict, dict]:
    """
    Derives and writes out two feature dictionaries which can be used with the feature dynamics framework.

        params:
            feature_names (list of str): the relevant feature names in the form of <ts_kind>||<feature_time_series>__<feature_dynamic>

        returns:
            feature_timeseries_mapping (dict): The feature calculators used to compute the feature time-series on the input time-series
            feature_dynamics_mapping (dict): The feature calculators used to compute the feature dynamics on the feature time-series

    """

    # TODO: NEEDS A HUGE CLEANUP!!!

    assert bool(feature_names) and all(
        isinstance(feature_name, str) for feature_name in feature_names
    )

    # Tokens for ts_kind separator 
    # and the param separator 
    # and the feature timeseries window_length separator 
    ts_kind_token = "||"
    param_token = "|"
    window_length_token = "@"

    # TODO: Map window length to ts_kind_dictionary
    # TODO: Map ts_kind to feature timeseries calculators
    # TODO: Map feature_timeseries's to feature dynamics calculators
    fts_mapping = {}  # window_length ---> ts_kind ---> feature calcs
    fd_mapping = {} # window_length ---> feature_timeseries name with window_length --> feature dynamics calcs
    skee = {}
    # Then function does for window_length, {ts_kind--->fts calculator}: extract_features(...)
    print(feature_names)
    for full_feature_name in feature_names:

        if not isinstance(full_feature_name, str):
            raise TypeError("Column name {} should be a string or unicode".format(full_feature_name))

        # Split according to our separator into <col_name>, <feature_name>, <feature_params> <window_length>
        fts_parts = full_feature_name.split("__")[0].replace(ts_kind_token, "__").replace(param_token, "__").replace(window_length_token, "__").split("__")
        n_fts_parts = len(fts_parts)

        if n_fts_parts == 1:
            raise ValueError(
                "Splitting of columnname {} resulted in only one part.".format(full_feature_name)
            )

        # split up into parts
        fd_parts = full_feature_name.split("__")
        n_fd_parts = len(fd_parts)


        if n_fd_parts == 1:
            raise ValueError(
                "Splitting of columnname {} resulted in only one part.".format(full_feature_name)
            )


        # Bunch of ugly stuff here
        feature_timeseries_name = fd_parts[0]
        feature_dynamics_name = fd_parts[1]
        kind = fts_parts[0]
        feature_ts_name = fts_parts[1]
        window_length_full_name = fts_parts[-1]
        window_length = int(fts_parts[-1].replace("window_", ""))
        fts_parts.remove(window_length_full_name)
        
        # fd mapping
        if window_length not in fd_mapping:
            fd_mapping[window_length] = {}

        if feature_timeseries_name not in fd_mapping:
            fd_mapping[window_length][feature_timeseries_name] = {}

        if not hasattr(feature_calculators, feature_dynamics_name):
            raise ValueError("Unknown feature name {}".format(feature_dynamics_name))

        fd_config = get_config_from_string(fd_parts)
        if fd_config:
            if feature_dynamics_name in fd_mapping[window_length][feature_timeseries_name]:
                fd_mapping[window_length][feature_timeseries_name][feature_dynamics_name].append(fd_config)
            else:
                fd_mapping[window_length][feature_timeseries_name][feature_dynamics_name] = [fd_config]
        else:
            fd_mapping[window_length][feature_timeseries_name][feature_dynamics_name] = None



        # Fts mapping
        if window_length not in fts_mapping:
            fts_mapping[window_length] = {}

        if kind not in fts_mapping[window_length]:
            fts_mapping[window_length][kind] = {}       

        if not hasattr(feature_calculators, feature_ts_name):
            raise ValueError("Unknown feature name {}".format(feature_ts_name))

        config = get_config_from_string(fts_parts)
        if config:
            if feature_ts_name in fts_mapping[window_length][kind]:
                if config not in fts_mapping[window_length][kind][feature_ts_name]:
                    fts_mapping[window_length][kind][feature_ts_name].append(config)
            else:
                fts_mapping[window_length][kind][feature_ts_name] = [config]
        else:
            fts_mapping[window_length][kind][feature_ts_name] = None

        print("THE MAIN EVENT")
        skee = add_to_feature_dictionary_with_windows(skee, window_length, fts_parts)
        print("THIS IS WHAT skee looks like")
        print(skee)
    sys.exit()
    return fts_mapping, fd_mapping

def engineer_input_timeseries(
    timeseries: pd.DataFrame,
    column_id: str = None,
    column_sort: str = None,
    compute_differences_within_series: bool = True,
    compute_differences_between_series: bool = False,
) -> pd.DataFrame:
    """
    Time series differencing with 1 order of differencing and phase difference operations to add new engineered time series to the input time series

    params:
         ts (pd.DataFrame): The pandas.DataFrame with the time series to compute the features for.
         compute_differences_within_series (bool): Temporal differences
         compute_differences_between_series (bool): Differences between two timeseries
         column_id (str): The name of the id column to group by. Please see :ref:`data-formats-label`.
         column_sort (str): The name of the sort column. Please see :ref:`data-formats-label`.
    """

    def series_differencing(ts: pd.DataFrame, ts_kinds: List[str]) -> pd.DataFrame:
        for ts_kind in ts_kinds:
            ts["dt_" + ts_kind] = ts[ts_kind].diff()
            ts.loc[
                0, ["dt_" + ts_kind]
            ] = 0  # adjust for the NaN value at first index.
        return ts

    def diff_between_series(ts: pd.DataFrame, ts_kinds: str) -> pd.DataFrame:
        assert (
            len(ts_kinds) > 1
        ), "Can only difference `ts` if there is more than one series"

        combs = combinations(ts_kinds, r=2)
        for first_ts_kind, second_ts_kind in combs:
            ts["D_" + first_ts_kind + second_ts_kind] = (
                ts[first_ts_kind] - ts[second_ts_kind]
            )
        return ts

    assert isinstance(timeseries, pd.DataFrame), "`ts` expected to be a pd.DataFrame"

    ts = timeseries.copy()

    ts_meta = ts[[column for column in [column_id, column_sort] if column is not None]]
    ts = ts.drop(
        [column for column in [column_id, column_sort] if column is not None], axis=1
    )

    assert all(
        is_numeric_dtype(ts[col]) for col in ts.columns.tolist()
    ), "All columns except `column_id` and `column_sort` in `ts` must be float or int"

    ts_kinds = ts.columns
    if compute_differences_within_series:
        ts = series_differencing(ts, ts_kinds)
    if compute_differences_between_series:
        ts = diff_between_series(ts, ts_kinds)

    return ts.join(ts_meta)
    


def interpret_feature_dynamic(feature_dynamic: str) -> dict:
    """
    
    """
    assert isinstance(feature_dynamic, str)

    # Generate the dictionaries that describe the information
    feature_timeseries_mapping, feature_dynamics_mapping = derive_features_dictionaries(
        feature_names=[feature_dynamic]
    )

    # Derive the key information that parameterises the feature name
    input_timeseries = next(iter(feature_timeseries_mapping.keys()))
    feature_dynamic_calculator = next(iter(feature_dynamics_mapping.values()))
    feature_timeseries_calculator, window_length = list(feature_timeseries_mapping.values())[0]

    # Return the key information as a dictionary
    return {
        "Full Feature Dynamic Name": feature_dynamic,
        "Input Timeseries": input_timeseries,
        "Feature Timeseries Calculator": feature_timeseries_calculator,
        "Window Length": window_length,
        "Feature Dynamic Calculator": feature_dynamic_calculator,
    }


def dictionary_to_string(dictionary: dict) -> str:
    formatted_output = ""
    for key, value in dictionary.items():
        formatted_output += f"**{key}** : ```{value}```<br>"
    return formatted_output


def gen_pdf_for_feature_dynamics(
    feature_dynamics_names: List[str],
    window_length: int,
    output_path: str = "feature_dynamics_interpretation",
) -> None:
    """ """
    feature_dynamics_summary = "<br/><br/><br/>".join(
        [
            dictionary_to_string(
                interpret_feature_dynamic(
                    feature_dynamic=feature_dynamics_name,
                    window_length=window_length,
                )
            )
            for feature_dynamics_name in feature_dynamics_names
        ]
    )

    title = "# Feature Dynamics Summary"
    linebreak = "---"
    context = "**Read more at:**"
    link1 = "* [How to interpret feature dynamics](https://github.com/blue-yonder/tsfresh/tree/main/notebooks/examples)"
    link2 = "* [List of feature calculators](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)"

    with open("feature_dynamics_interpretation.md", "w") as f:
        f.write(
            f"{title}\n\n{linebreak}\n\n{context}\n\n{link1}\n\n{link2}\n\n{linebreak}\n\n{feature_dynamics_summary}"
        )

    md2pdf(
        pdf_file_path=f"{output_path}.pdf",
        md_file_path=f"{output_path}.md",
    )
