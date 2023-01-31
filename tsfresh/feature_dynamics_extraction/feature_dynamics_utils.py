from tsfresh.feature_extraction.settings import from_columns
from typing import List, Tuple
from itertools import combinations
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import List
from md2pdf.core import md2pdf
from tsfresh.feature_extraction import feature_calculators
from tsfresh.utilities.string_manipulation import get_config_from_string
from tsfresh.feature_extraction.data import (
    to_tsdata,
    Timeseries,
    WideTsFrameAdapter,
    LongTsFrameAdapter,
    TsDictAdapter,
)


def clean_feature_timeseries_name(
    feature_timeseries_name: str, window_length: int
) -> str:
    """
    Logic to clean up the feature time series name after the first round of extraction
    including adding the window length information into the feature timeseries name
    """
    ts_kind_token = "||"
    return (
        feature_timeseries_name.replace("__", ts_kind_token)
        + f"@window_{window_length}"
    )


def update_feature_dictionary(feature_dictionary, window_length, feature_parts):
    """
    Adds a single entry into a feature calculator dictionary.

    This dictionary includes the window length information i.e. the dictionary
    maps:
    window_length ----> ts_kind ---> name_of_feature_calc ---> feature_calc_params`

    This function assumes that `feature_parts` is organised like:
    kind, feature_name, *feature_params
    """

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


def parse_feature_timeseries_parts(full_feature_name: str) -> dict:
    """
    Generates the feature timeseries (fts) parts from a full feature dynamic (fd) name

    i.e. given a ful feature dynamic name like:
    <ts_kind>||<fts_name>||<fts_params>@<window_length>__<fd_name>__<fd_params>

    Then this function will return a dictionary with two keys:
    1) window_length, an integer
    2) fts_parts, a list containing:
    [
        <ts_kind>,
        <fts_name>,
        <fts_params>*
    ]
    (where multiple fts_params are allowed and are each a separate element)
    """

    ts_kind_token = "||"
    window_length_token = "@"

    # Split according to our separator into <col_name>, <feature_name>, <feature_params> <window_length>
    fts_parts = (
        full_feature_name.split("__")[0]
        .replace(ts_kind_token, "__")
        .replace(window_length_token, "__")
        .split("__")
    )

    if "window_" not in fts_parts[-1]:
        raise ValueError(
            "Window length information not found in feature time series name. Did you pass in a window length of zero?"
        )

    else:
        # remove window length information from ts_parts and store separately
        window_length = int(fts_parts[-1].replace("window_", ""))
        fts_parts = fts_parts[:-1]

    n_fts_parts = len(fts_parts)
    if n_fts_parts == 1:
        raise ValueError(
            "Splitting of columnname {} resulted in only one part.".format(
                full_feature_name
            )
        )

    return {"window_length": window_length, "fts_parts": fts_parts}


def parse_feature_dynamics_parts(full_feature_name):
    """
    Generates the feature dynamics (fd) parts from a full feature dynamic (fd) name

    i.e. given a full feature dynamic name like:
    <ts_kind>||<fts_name>||<fts_params>@<window_length>__<fd_name>__<fd_params>

    Then this function will return a dictionary with one key:
    1) fd_parts, a list containing:
    [
        <ts_kind>||<fts_name>||<fts_params>@<window_length>,
        <fd_name>,
        <fd_params>*
    ]
    (where multiple fd_params are allowed and are each a separate element)
    """
    # Split according to our separator into <col_name>, <feature_name>, <feature_params>
    fd_parts = full_feature_name.split("__")
    n_fd_parts = len(fd_parts)
    if n_fd_parts == 1:
        raise ValueError(
            "Splitting of columnname {} resulted in only one part.".format(
                full_feature_name
            )
        )
    return {"fd_parts": fd_parts}


def derive_features_dictionaries(feature_names: List[str]) -> Tuple[dict, dict]:
    """
    Derives and writes out two feature dictionaries which can be used with the feature dynamics framework.

        params:
            feature_names (list of str): the relevant feature names in the form of:
            <ts_kind>||<feature_time_series>||<feature_times_series_params>@<window_length>__<feature_dynamic_name>__<feature_dynamics_params>

        returns:
            fts_mapping (dict): The mapping used to compute the feature time-series on the ts kinds.
            fd_mapping (dict): The mapping used to compute the feature dynamics on the feature time series.

            The fts and fd mappings map window_lengths to a dictionary of column names (either
            ts_kinds names or fts names) which map to a dictionary of feature calculators.
            i.e. window_length ---> column_name ---> feature_calculator_name ---> [*feature_calculator_params]

    """
    fts_mapping = {}
    fd_mapping = {}

    for full_feature_name in feature_names:
        if not isinstance(full_feature_name, str):
            raise TypeError(
                "Column name {} should be a string or unicode".format(full_feature_name)
            )

        fts_information = parse_feature_timeseries_parts(full_feature_name)
        fts_mapping = update_feature_dictionary(
            fts_mapping,
            window_length=fts_information.get("window_length"),
            feature_parts=fts_information.get("fts_parts"),
        )

        fd_information = parse_feature_dynamics_parts(full_feature_name)
        fd_mapping = update_feature_dictionary(
            fd_mapping,
            window_length=fts_information.get("window_length"),
            feature_parts=fd_information.get("fd_parts"),
        )

    return fts_mapping, fd_mapping


def diff_within_series(
    timeseries_container,
    column_id: str = None,
    column_sort: str = None,
    column_kind: str = None,
    column_value: str = None,
):
    """
    Time series differencing with 1 order of differencing new engineered time series to the input time series
    params:
         ts (pd.DataFrame): The pandas.DataFrame with the time series to compute the features for.
         compute_differences_within_series (bool): Differences within the same timeseries.
         The first value is always set to zero i.e. f([1,5,2,6]) = [0,4,-3,4]
         compute_differences_between_series (bool): Differences between two different timeseries.
         i.e. f([1,2,3,4],[4,3,2,1]) = [3,1,-1,-3]
         column_id (str): The name of the id column to group by. Please see :ref:`data-formats-label`.
         column_sort (str): The name of the sort column. Please see :ref:`data-formats-label`.
    returns:
         Returns the original dataframe with the engineered timeseries added as new columns
    """
    data = to_tsdata(
        timeseries_container, column_id, column_kind, column_value, column_sort
    )

    if not isinstance(data, (WideTsFrameAdapter, LongTsFrameAdapter, TsDictAdapter)):
        raise ValueError(
            "Please use a valid supported data format (Dask and PySpark are not supported)."
        )

    if column_sort is None:
        indexing_columns = [column_id]
        sort_column = []
    else:
        indexing_columns = [column_id, column_sort]
        sort_column = [column_sort]

    # Case 1: Flat dataframe
    if isinstance(data, WideTsFrameAdapter):

        timeseries_container_cp = timeseries_container.copy()

        new_kinds = [
            f"dt_{kind}" for kind in timeseries_container.drop(indexing_columns, axis=1)
        ]

        timeseries_container_cp[new_kinds] = (
            timeseries_container.sort_values(sort_column)
            .drop(sort_column, axis=1)
            .groupby(column_id)
            .diff()
            .fillna(0)
        )

    # Case 2: Stacked dataframe
    elif isinstance(data, LongTsFrameAdapter):

        def stacked_df_within_differencer(timeseries_container):
            """
            First yields the full input dataframe, and then
            every subsequent call yields a new dataframe each time
            with the value column being the difference within
            a single timeseries kind as found in the original
            stacked dataframe
            """

            yield timeseries_container

            for kind, dataframe in timeseries_container.groupby(column_kind):

                new_timeseries = dataframe.copy()

                new_timeseries[column_value] = (
                    new_timeseries.sort_values(sort_column)
                    .groupby(column_id)[column_value]
                    .diff()
                    .fillna(0)
                )

                new_timeseries[column_kind] = f"dt_{kind}"

                yield new_timeseries

        timeseries_container_cp = pd.concat(
            stacked_df_within_differencer(timeseries_container)
        ).reset_index(drop=True)

    # Case 3: Dict of flat dataframes
    elif isinstance(data, TsDictAdapter):

        timeseries_container_cp = timeseries_container.copy()

        for kind, flat_dataframe in timeseries_container.items():

            new_timeseries = flat_dataframe.copy()

            new_timeseries[column_value] = (
                new_timeseries.sort_values(sort_column)
                .groupby(column_id)[column_value]
                .diff()
                .fillna(0)
            )

            timeseries_container_cp[f"dt_{kind}"] = new_timeseries

    return timeseries_container_cp


def diff_between_series(
    timeseries_container, column_id, column_sort, column_kind, column_value
):
    """
    Time series differencing with 1 order of differencing and phase difference operations to add new engineered time series to the input time series

    params:
         ts (pd.DataFrame): The pandas.DataFrame with the time series to compute the features for.
         compute_differences_within_series (bool): Differences within the same timeseries.
         The first value is always set to zero i.e. f([1,5,2,6]) = [0,4,-3,4]
         compute_differences_between_series (bool): Differences between two different timeseries.
         i.e. f([1,2,3,4],[4,3,2,1]) = [3,1,-1,-3]
         column_id (str): The name of the id column to group by. Please see :ref:`data-formats-label`.
         column_sort (str): The name of the sort column. Please see :ref:`data-formats-label`.

    returns:
         Returns the original dataframe with the engineered timeseries added as new columns

    """

    data = to_tsdata(
        timeseries_container, column_id, column_kind, column_value, column_sort
    )

    if not isinstance(data, (WideTsFrameAdapter, LongTsFrameAdapter, TsDictAdapter)):
        raise ValueError(
            "Please use a valid supported data format (Dask and PySpark is not supported)."
        )

    if len(data) <= 1:
        raise ValueError(
            "len(data) needs to be greater than 1. Can only difference `timeseries_container` if there is more than one series"
        )

    if column_sort is None:
        indexing_columns = [column_id]
    else:
        indexing_columns = [column_id, column_sort]

    # Case 1: Flat
    if isinstance(data, WideTsFrameAdapter):

        timeseries_container_cp = timeseries_container.copy()

        for first_kind, second_kind in combinations(
            timeseries_container.drop(indexing_columns, axis=1), r=2
        ):
            new_kind = f"D_{first_kind}{second_kind}"

            timeseries_container_cp[new_kind] = (
                timeseries_container_cp.set_index(indexing_columns)[first_kind]
                .subtract(
                    timeseries_container_cp.set_index(indexing_columns)[second_kind]
                )
                .reset_index(indexing_columns, drop=True)
            )

    # Case 2: Stacked
    elif isinstance(data, LongTsFrameAdapter):

        def stacked_df_between_differencer(timeseries_container):
            """
            First yields the full input dataframe, and then
            every subsequent call yields a new dataframe each time
            with the value column being the difference between
            two timeseries kinds as found in the original
            stacked dataframe
            """

            yield timeseries_container

            for first_timeseries, second_timeseries in combinations(
                timeseries_container.groupby(column_kind), r=2
            ):

                first_kind, first_dataframe = first_timeseries
                second_kind, second_dataframe = second_timeseries

                new_timeseries = (
                    first_dataframe.drop(column_kind, axis=1)
                    .set_index(indexing_columns)
                    .subtract(
                        second_dataframe.drop(column_kind, axis=1).set_index(
                            indexing_columns
                        )
                    )
                    .reset_index()
                )

                new_timeseries[column_kind] = f"D_{first_kind}{second_kind}"

                yield new_timeseries

        timeseries_container_cp = pd.concat(
            stacked_df_between_differencer(timeseries_container)
        ).reset_index(drop=True)

    # Case 3: Dict of flat
    elif isinstance(data, TsDictAdapter):

        timeseries_container_cp = timeseries_container.copy()

        for first_timeseries, second_timeseries in combinations(
            timeseries_container.items(), r=2
        ):
            first_kind, first_dataframe = first_timeseries
            second_kind, second_dataframe = second_timeseries

            new_timeseries = (
                first_dataframe.set_index(indexing_columns)
                .subtract(second_dataframe.set_index(indexing_columns))
                .reset_index()
            )

            timeseries_container_cp[f"D_{first_kind}{second_kind}"] = new_timeseries

    return timeseries_container_cp


def interpret_feature_dynamic(feature_dynamic: str) -> dict:
    """
    Breaks up a full feature dynamic name into its constituent parts.

    It turns a complex full feature dynamic (fd) name:
    <ts_kind>||<fts_name>||<fts_params*>@<window_length>__<fd_name>__<fd_params>

    Into a dictionary containing each part, including the full original name

    {
        "Full Feature Dynamic Name" : <ts_kind>||<fts_name>||<fts_params*>@<window_length>__<fd_name>__<fd_params>,
        "Input Timeseries": <ts_kind>,
        "Feature Timeseries Calculator": {<fts_name> : [*<fts_params>]},
        "Window Length": <window_length>,
        "Feature Dynamic Calculator": {<fd_name> : [*<fd_params>]},
    }

    """
    assert isinstance(feature_dynamic, str)

    # Generate the dictionaries that describe the information
    feature_timeseries_mapping, feature_dynamics_mapping = derive_features_dictionaries(
        feature_names=[feature_dynamic]
    )

    # Derive the key information that parameterises the feature name
    window_length = next(iter(feature_timeseries_mapping.keys()))
    input_timeseries = next(iter(feature_timeseries_mapping[window_length].keys()))
    feature_dynamic_calculator = next(
        iter(feature_dynamics_mapping[window_length].values())
    )
    feature_timeseries_calculator = next(
        iter(feature_timeseries_mapping[window_length].values())
    )

    # Return the key information as a dictionary
    return {
        "Full Feature Dynamic Name": feature_dynamic,
        "Input Timeseries": input_timeseries,
        "Feature Timeseries Calculator": feature_timeseries_calculator,
        "Window Length": window_length,
        "Feature Dynamic Calculator": feature_dynamic_calculator,
    }


def dictionary_to_string(dictionary: dict) -> str:
    """
    Formats a dictionary as a string, where the
    keys are bolded and values are italicised
    """
    formatted_output = ""
    for key, value in dictionary.items():
        formatted_output += f"**{key}** : ```{value}```<br>"
    return formatted_output


def gen_pdf_for_feature_dynamics(
    feature_dynamics_names: List[str],
    output_filename: str = "feature_dynamics_interpretation",
) -> None:
    """
    Given a list of full feature dynamics names, write these
    names and their interpretations to PDF and markdown documents
    """
    feature_dynamics_summary = "<br/><br/><br/>".join(
        [
            dictionary_to_string(
                interpret_feature_dynamic(feature_dynamic=feature_dynamics_name)
            )
            for feature_dynamics_name in feature_dynamics_names
        ]
    )

    title = "# Feature Dynamics Summary"
    linebreak = "---"
    context = "**Read more at:**"
    link1 = "* [How to interpret feature dynamics](https://github.com/blue-yonder/tsfresh/tree/main/notebooks/examples)"
    link2 = "* [List of feature calculators](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)"

    with open(f"{output_filename}.md", "w") as f:
        f.write(
            f"{title}\n\n{linebreak}\n\n{context}\n\n{link1}\n\n{link2}\n\n{linebreak}\n\n{feature_dynamics_summary}"
        )

    md2pdf(
        pdf_file_path=f"{output_filename}.pdf",
        md_file_path=f"{output_filename}.md",
    )
