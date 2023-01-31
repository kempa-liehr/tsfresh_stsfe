# -*- coding: utf-8 -*-

"""
This module contains the main function to interact with tsfresh: extract feature dynamics
"""

from collections.abc import Iterable
from dask import dataframe as dd
import pandas as pd

from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.feature_dynamics_extraction.feature_dynamics_data import (
    IterableSplitTsData,
    ApplyableSplitTsData,
)
from tsfresh.feature_extraction.data import to_tsdata
from tsfresh.feature_dynamics_extraction.feature_dynamics_utils import (
    clean_feature_timeseries_name,
)


def extract_feature_dynamics(
    timeseries_container,
    feature_timeseries_fc_parameters=None,
    feature_timeseries_kind_to_fc_parameters=None,
    feature_dynamics_fc_parameters=None,
    feature_dynamics_kind_to_fc_parameters=None,
    column_id=None,
    column_sort=None,
    column_kind=None,
    column_value=None,
    **kwargs,
):
    """ """

    relevant_fts_dicts = [
        fts_dict_calc
        for fts_dict_calc in [
            feature_timeseries_fc_parameters,
            feature_timeseries_kind_to_fc_parameters,
        ]
        if fts_dict_calc is not None
    ]
    relevant_fd_dicts = [
        fd_dict_calc
        for fd_dict_calc in [
            feature_dynamics_fc_parameters,
            feature_dynamics_kind_to_fc_parameters,
        ]
        if fd_dict_calc is not None
    ]
    if len(relevant_fts_dicts) != 1 or len(relevant_fd_dicts) != 1:
        error_mssg = "Must supply exactly one feature timeseries feature calculator dictionary and exactly one feature timeseries feature calculator dictionary"
        more_info = "Check that one of `feature_timeseries_fc_parameters` or `feature_timeseries_kind_to_fc_parameters` is supplied, \
            and that one of `feature_dynamics_fc_parameters` or `feature_dynamics_kind_to_fc_parameters` is supplied."
        raise ValueError("\n".join([error_mssg, more_info]))

    (relevant_fts_dict,) = relevant_fts_dicts
    (relevant_fd_dict,) = relevant_fd_dicts

    if relevant_fts_dict == {} or relevant_fts_dict == {}:
        raise ValueError("Dictionaries are empty")

    if set(relevant_fd_dict) != set(relevant_fts_dict):
        raise ValueError(
            "Window lengths of the feature timeseries feature calculators mismatched with the window lengths of the feature dynamics feature calculators"
        )

    Xs = []

    for window_length in relevant_fts_dict.copy().keys():

        X = do_feature_dynamics_extraction(
            timeseries_container,
            window_length=window_length,
            feature_timeseries_fc_parameters=feature_timeseries_fc_parameters[
                window_length
            ]
            if feature_timeseries_fc_parameters
            else None,
            feature_timeseries_kind_to_fc_parameters=feature_timeseries_kind_to_fc_parameters[
                window_length
            ]
            if feature_timeseries_kind_to_fc_parameters
            else None,
            feature_dynamics_fc_parameters=feature_dynamics_fc_parameters[window_length]
            if feature_dynamics_fc_parameters
            else None,
            feature_dynamics_kind_to_fc_parameters=feature_dynamics_kind_to_fc_parameters[
                window_length
            ]
            if feature_dynamics_kind_to_fc_parameters
            else None,
            column_id=column_id,
            column_sort=column_sort,
            column_kind=column_kind,
            column_value=column_value,
            **kwargs,
        )

        Xs.append(X)

    if isinstance(X, dd.DataFrame):
        return dd.multi.concat(Xs, axis=1)
    else:
        return pd.concat(Xs, axis=1)


def do_feature_dynamics_extraction(
    timeseries_container,
    window_length=None,
    feature_timeseries_fc_parameters=None,
    feature_timeseries_kind_to_fc_parameters=None,
    feature_dynamics_fc_parameters=None,
    feature_dynamics_kind_to_fc_parameters=None,
    column_id=None,
    column_sort=None,
    column_kind=None,
    column_value=None,
    **kwargs,
):
    """
    Extract feature dynamics from a time series.
    This process involves calling `extract_features` on the original time series which produces a new set of columns - feature time series.
    Then for each of these features generated, they are again windowed and `extract_features` is called again producing a set
    of feature-dynamics or sub-features.

    The details of this function are explained in the following paper "Data Mining on Extremely Long Time Series"
    see at https://ieeexplore.ieee.org/document/9679945.

    :param timeseries_container: The pandas.DataFrame with the time series to compute the features for.
    :type timeseries_container: pandas.DataFrame or dask.DataFrame.

    :param window_length: The size of the window from which the first set of features is extracted.
    :type sub_feature_split: int

    :param feature_timeseries_fc_parameters: mapping from feature calculator names to parameters.
           These are applied to the first set of features generated.
    :type sub_default_fc_parameters: dict

    :param feature_timeseries_kind_to_fc_parameters: apping from kind names to objects of the same type as the ones for
            default_fc_parameters.These are applied to the first set of features generated.
    :type sub_kind_to_fc_parameters: dict

    :param feature_dynamics_fc_parameters: mapping from feature calculator names to parameters. Only those names
           which are keys in this dict will be calculated. See the class:`ComprehensiveFCParameters` for
           more information.
    :type default_fc_parameters: dict

    :param feature_dynamics_kind_to_fc_parameters: mapping from kind names to objects of the same type as the ones for
            default_fc_parameters. If you put a kind as a key here, the fc_parameters
            object (which is the value), will be used instead of the default_fc_parameters. This means that kinds, for
            which kind_of_fc_parameters doe not have any entries, will be ignored by the feature selection.
    :type kind_to_fc_parameters: dict

    :param column_id: The name of the id column to group by. Please see :ref:`data-formats-label`.
    :type column_id: str

    :param column_sort: The name of the sort column. Please see :ref:`data-formats-label`.
    :type column_sort: str

    :param column_kind: The name of the column keeping record on the kind of the value.
            Please see :ref:`data-formats-label`.
    :type column_kind: str

    :param column_value: The name for the column keeping the value itself. Please see :ref:`data-formats-label`.
    :type column_value: str

    :param kwargs: optional keyword arguments passed to `extract_features`.
    :type kwargs: dict

    :return: DataFrame of the extracted feature dynamics.
    :rtype: pandas.DataFrame or dask.DataFrame

    Examples
    ========

    >>> import pandas as pd
    >>> from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
    >>> from tsfresh.feature_extraction import extract_features, extract_features_on_sub_features

    >>> download_robot_execution_failures()
    >>> timeseries, y = load_robot_execution_failures()

    >>> extracted_sub_features = extract_features_on_sub_features(timeseries_container=timeseries,
                                                                  sub_feature_split=11,
                                                                  column_id="id",
                                                                  column_sort="time"))

    """

    if not isinstance(window_length, int):
        raise TypeError(
            f"Window length needs to be of type int, but was found to be type {type(window_length)}. Currently only integer window lengths are accepted for extracting feature dynamics"
        )

    # TODO: Implement a check which makes sure that the window length is not too large... otherwise there will be weird bugs
    # IMPORTANT TODO ^^^ right now there are no checks on illegal window lengths...

    ts_data = to_tsdata(
        timeseries_container,
        column_id=column_id,
        column_sort=column_sort,
        column_kind=column_kind,
        column_value=column_value,
    )

    if isinstance(ts_data, Iterable):
        split_ts_data = IterableSplitTsData(ts_data, window_length)
    else:
        split_ts_data = ApplyableSplitTsData(ts_data, window_length)

    feature_timeseries = extract_features(
        split_ts_data,
        default_fc_parameters=feature_timeseries_fc_parameters,
        kind_to_fc_parameters=feature_timeseries_kind_to_fc_parameters,
        **kwargs,
        pivot=False,
    )

    column_kind = column_kind or "variable"
    column_id = column_id or "id"
    column_sort = column_sort or "sort"
    column_value = column_value or "value"

    # The feature names include many "_", which will confuse tsfresh where the sub feature name ends
    # and where the real feature name starts.
    # Also, we split up the index into the id and the sort
    # We need to do this separately for dask dataframes,
    # as the return type is not a list, but already a dataframe.
    # We also need to drop values that contain NaNs.
    if isinstance(feature_timeseries, dd.DataFrame):

        feature_timeseries = feature_timeseries.reset_index(drop=True)

        feature_timeseries[column_kind] = feature_timeseries[column_kind].apply(
            lambda col: clean_feature_timeseries_name(col, window_length),
            meta=(column_kind, object),
        )

        feature_timeseries[column_sort] = feature_timeseries[column_id].apply(
            lambda x: x[1], meta=(column_id, "int64")
        )
        feature_timeseries[column_id] = feature_timeseries[column_id].apply(
            lambda x: x[0], meta=(column_id, ts_data.df_id_type)
        )

        # Need to drop features for all windows which contain at least one NaN
        null_columns = (
            feature_timeseries[feature_timeseries[column_value].isnull()][column_kind]
            .unique()
            .compute()
        )
        feature_timeseries = feature_timeseries[
            ~feature_timeseries[column_kind].isin(null_columns)
        ]

    else:
        feature_timeseries = pd.DataFrame(
            feature_timeseries, columns=[column_id, column_kind, column_value]
        )

        feature_timeseries[column_kind] = feature_timeseries[column_kind].apply(
            lambda col: clean_feature_timeseries_name(col, window_length)
        )

        feature_timeseries[column_sort] = feature_timeseries[column_id].apply(
            lambda x: x[1]
        )

        feature_timeseries[column_id] = feature_timeseries[column_id].apply(
            lambda x: x[0]
        )

        # Need to drop features for all windows which contain at least one NaN
        null_columns = feature_timeseries[feature_timeseries[column_value].isnull()][
            column_kind
        ].unique()

        feature_timeseries = feature_timeseries[
            ~feature_timeseries[column_kind].isin(null_columns)
        ]

    # Coerce time series values, which include boolean values, integers, and floats
    # (currently all stored as dtype: "object") into floats.
    feature_timeseries[column_value] = pd.to_numeric(feature_timeseries[column_value])

    X = extract_features(
        feature_timeseries,
        column_id=column_id,
        column_sort=column_sort,
        column_kind=column_kind,
        column_value=column_value,
        default_fc_parameters=feature_dynamics_fc_parameters,
        kind_to_fc_parameters=feature_dynamics_kind_to_fc_parameters,
        **kwargs,
    )

    # Drop all feature dynamics that are associated with at least one NaN.
    if isinstance(feature_timeseries, dd.DataFrame):
        cols_to_keep = X.isna().compute().any() == False
        X = X.loc[:, cols_to_keep.tolist()]
    elif isinstance(feature_timeseries, pd.DataFrame):
        X = X.dropna(axis="columns", how="any")
    else:
        raise ValueError

    return X
