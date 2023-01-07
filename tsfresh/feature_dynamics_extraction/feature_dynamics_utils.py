from tsfresh.feature_extraction.settings import from_columns
from typing import List, Tuple
import itertools
import pandas as pd
from pandas.api.types import is_numeric_dtype
from typing import List
from md2pdf.core import md2pdf


def generate_feature_name_encoding(feature_names):
    """
    returns a dictionary which maps encoded feature
    names to their long full feature names 
    """
    pass

def clean_feature_timeseries_name(feature_timeseries_name:str, window_length: int) -> str:
    """
    Logic to clean up the feature time series name after the first round of extraction

    NB: This might not be sufficient
    """

    return feature_timeseries_name.replace("__", f"||window_{window_length}@",1)

def derive_features_dictionaries(feature_names: List[str]) -> Tuple[dict, dict]:
    """
    Derives and writes out two feature dictionaries which can be used with the feature dynamics framework.

        params:
            feature_names (list of str): the relevant feature names in the form of <ts_kind>||<feature_time_series>__<feature_dynamic>

        returns:
            feature_timeseries_mapping (dict): The feature calculators used to compute the feature time-series on the input time-series
            feature_dynamics_mapping (dict): The feature calculators used to compute the feature dynamics on the feature time-series

    """

    assert bool(feature_names) and all(
        isinstance(feature_name, str) for feature_name in feature_names
    )

    replacement_token = "||"
    feature_dynamics_mapping = from_columns(feature_names)
    feature_timeseries_mapping = from_columns(
        [str(x).replace(replacement_token, "__") for x in [*feature_dynamics_mapping]]
    )
    return feature_timeseries_mapping, feature_dynamics_mapping

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

        combs = itertools.combinations(ts_kinds, r=2)
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
    


def interpret_feature_dynamic(feature_dynamic: str, window_length: int) -> dict:
    """
    TODO: Fix for new feature naming convention
    """
    assert isinstance(feature_dynamic, str)

    feature_timeseries_mapping, feature_dynamics_mapping = derive_features_dictionaries(
        feature_names=[feature_dynamic]
    )

    return {
        "Full Feature Dynamic Name": feature_dynamic,
        "Input Timeseries": list(feature_timeseries_mapping.keys())[0],
        "Feature Timeseries Calculator": list(feature_timeseries_mapping.values())[0],
        "Window Length": window_length,
        "Feature Dynamic Calculator": list(feature_dynamics_mapping.values())[0],
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
