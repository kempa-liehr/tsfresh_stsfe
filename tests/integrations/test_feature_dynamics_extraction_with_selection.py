import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
from unittest import TestCase
import pytest

from tsfresh.feature_dynamics_extraction.feature_dynamics_extraction import (
    extract_feature_dynamics,
)
from tsfresh.feature_extraction.settings import (
    MinimalFCParameters,
    EfficientFCParameters,
)
from tsfresh.feature_selection import select_features

from tsfresh.feature_dynamics_extraction.feature_dynamics_utils import (
    derive_features_dictionaries,
    gen_pdf_for_feature_dynamics,
    diff_within_series,
    diff_between_series,
)

from typing import List, Dict


class FixturesForFeatureDynamicsIntegrationTests(TestCase):
    def column_params_picker(self, data_format):
        """
        Picks out the correct column params depending on
        the particular data format type
        """

        if data_format not in ["wide", "long", "dict"]:
            raise ValueError

        if data_format == "wide":
            return {
                "column_sort": "t",
                "column_kind": None,
                "column_id": "measurement_id",
                "column_value": None,
            }

        elif data_format == "long":
            return {
                "column_sort": "t",
                "column_kind": "kind",
                "column_id": "measurement_id",
                "column_value": "value",
            }

        elif data_format == "dict":
            return {
                "column_sort": "t",
                "column_kind": None,
                "column_id": "measurement_id",
                "column_value": "value",
            }

    def gen_feature_calculators_for_e2e_tests(self, feature_complexity="minimal"):

        if feature_complexity not in ["minimal", "not-minimal"]:
            raise ValueError(
                'feature_complexity needs to be one of ["minimal", "not-minimal"]'
            )

        if feature_complexity == "minimal":
            return MinimalFCParameters()
        elif feature_complexity == "not-minimal":
            # Get a reasonably sized subset of somewhat comeplex features
            non_simple_features_fc_parameters = {
                "fft_coefficient": [{"coeff": 1, "attr": "real"}],
                "number_cwt_peaks": [{"n": 3}],
                "permutation_entropy": [{"tau": 1, "dimension": 2}],
                "quantile": [{"q": 0.2}],
            }

            return non_simple_features_fc_parameters

    def check_correct_ts_are_engineered(
        self,
        expected_ts: List,
        timeseries_container,
        data_format: str,
        column_params_config: Dict,
    ):

        if data_format == "wide":
            self.assertEqual(set(expected_ts), set(timeseries_container.columns))

        elif data_format == "long":
            long_columns = set(timeseries_container.columns)
            self.assertTrue(
                column_params_config["column_kind"] in long_columns
                and column_params_config["column_value"] in long_columns
            )
            long_columns.remove(column_params_config["column_kind"])
            long_columns.remove(column_params_config["column_value"])
            ts_kinds = set(timeseries_container[column_params_config["column_kind"]])
            self.assertEqual(
                set(expected_ts),
                set.union(set(timeseries_container[list(long_columns)]), ts_kinds),
            )

        elif data_format == "dict":
            dict_columns = set()
            for df in timeseries_container.values():
                dict_columns.update(df.columns.tolist())

            self.assertTrue(column_params_config["column_value"] in dict_columns)
            dict_columns.remove(column_params_config["column_value"])

            self.assertEqual(
                set(expected_ts),
                set.union(set(timeseries_container.keys()), dict_columns),
            )

        else:
            raise ValueError

    def gen_example_timeseries_data_for_e2e_tests(self, container_type, data_format):

        """
        TODO: This data should be refactored, but keeping it around for now
        TODO: This should absolutely be replaced or deleted very soon before being merged into anything...
        """

        if container_type not in ["pandas", "dask"]:
            raise ValueError
        if data_format not in ["wide", "long", "dict"]:
            raise ValueError

        y1 = [
            "0",
            "0",
            "0",
            "345346",
            "1356",
            "135",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "32425436",
            "0",
            "0",
            "345346",
            "0",
            "44444444444",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "32425436",
            "0",
            "0",
            "345346",
            "0",
            "44444444444",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "32425436",
            "0",
            "0",
            "345346",
            "0",
            "44444444444",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "32425436",
            "0",
            "0",
            "345346",
            "0",
            "44444444444",
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
        ]

        y2 = [
            "457",
            "352",
            "3524",
            "124532",
            "24",
            "24",
            "214",
            "21",
            "46",
            "42521",
            "532",
            "634",
            "32",
            "64375",
            "235",
            "325",
            "563323",
            "6",
            "32",
            "532",
            "52",
            "57",
            "324",
            "643",
            "32",
            "436",
            "34",
            "57",
            "34",
            "65",
            "643",
            "34",
            "346",
            "43",
            "54",
            "8",
            "4",
            "43",
            "537",
            "543",
            "43",
            "56",
            "32",
            "34",
            "32",
            "5",
            "65",
            "43",
            "435",
            "54",
            "7654",
            "5",
            "67",
            "54",
            "345",
            "43",
            "32",
            "32",
            "65",
            "76",
        ]

        y3 = [
            "3454",
            "13452",
            "23534",
            "12432",
            "412432",
            "324",
            "43",
            "5",
            "64",
            "356",
            "3245235",
            "32",
            "325",
            "5467",
            "657",
            "235",
            "234",
            "34",
            "2344234",
            "56",
            "21435",
            "214",
            "1324",
            "4567",
            "34232",
            "132214",
            "42",
            "34",
            "343",
            "3443",
            "124",
            "5477",
            "36478",
            "879",
            "414",
            "45",
            "7899",
            "786",
            "657",
            "677",
            "45645",
            "3534",
            "424",
            "354545",
            "36645",
            "67867",
            "56867",
            "78876",
            "5646",
            "3523",
            "2434",
            "324423",
            "68",
            "89",
            "456",
            "435",
            "3455",
            "35443",
            "24332",
            "12313",
        ]

        measurement_id = [
            "1",
            "1",
            "1",
            "1",
            "1",
            "1",
            "2",
            "2",
            "2",
            "2",
            "2",
            "2",
            "3",
            "3",
            "3",
            "3",
            "3",
            "3",
            "4",
            "4",
            "4",
            "4",
            "4",
            "4",
            "5",
            "5",
            "5",
            "5",
            "5",
            "5",
            "6",
            "6",
            "6",
            "6",
            "6",
            "6",
            "7",
            "7",
            "7",
            "7",
            "7",
            "7",
            "8",
            "8",
            "8",
            "8",
            "8",
            "8",
            "9",
            "9",
            "9",
            "9",
            "9",
            "9",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
        ]

        df = pd.DataFrame(
            {
                "t": np.repeat([1, 2, 3, 4, 5, 6], 10),
                "y1": np.asarray(y1, dtype=float),
                "y2": np.asarray(y2, dtype=float),
                "y3": np.asarray(y3, dtype=float),
                "measurement_id": np.asarray(measurement_id, dtype=int),
            }
        )

        if data_format == "wide":

            ts = df
            if container_type == "dask":
                ts = dd.from_pandas(ts, npartitions=3)

        elif data_format == "long":

            ts = pd.melt(
                df,
                id_vars=["measurement_id", "t"],
                value_vars=["y1", "y2", "y3"],
                value_name="value",
                var_name="kind",
            ).reset_index(drop=True)

            if container_type == "dask":
                ts = dd.from_pandas(ts, npartitions=3)

        elif data_format == "dict":

            ts = {kind: pd.DataFrame() for kind in ["y1", "y2", "y3"]}
            for kind in ts.keys():
                ts[kind] = (
                    df[["t"] + [kind] + ["measurement_id"]]
                    .rename(columns={kind: "value"})
                    .reset_index(drop=True)
                )

            if container_type == "dask":
                for kind in ts.keys():
                    ts[kind] = dd.from_pandas(ts[kind], npartitions=3)

        response = (
            pd.DataFrame(
                {
                    "response": np.asarray([0, 1] * 5),
                    "measurement_id": np.asarray(np.arange(1, 11, dtype=int)),
                }
            )
            .set_index("measurement_id")
            .squeeze()
        )

        return ts, response


class EngineerMoreTsTestCase(FixturesForFeatureDynamicsIntegrationTests):
    """
    Tests for engineering more timeseries,
    and then extracting feature dynamics based
    on these.
    """

    # Engineer on pandas (3 input formats, large and small dicts)
    # Engineer on pandas then convert to dask check it still works (3 formats, large and small dicts)

    def test_engineer_more_ts_and_then_extraction_on_pandas_wide(self):

        data_format = "wide"

        ts, response = self.gen_example_timeseries_data_for_e2e_tests(
            container_type="pandas", data_format=data_format
        )
        column_params_config = self.column_params_picker(data_format=data_format)

        # Engineer some more timeseries from input timeseries [D_y1y2, D_y1y3, D_y2y3]
        ts_with_extra_timeseries_between = diff_between_series(
            ts,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts = [
            "t",
            "measurement_id",
            "y1",
            "y2",
            "y3",
            "D_y1y2",
            "D_y1y3",
            "D_y2y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between,
            data_format,
            column_params_config,
        )

        # add an even extra layer of ts differencing
        ts_with_extra_timeseries_between_and_within = diff_within_series(
            ts_with_extra_timeseries_between,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts += [
            "dt_y1",
            "dt_y2",
            "dt_y3",
            "dt_D_y1y2",
            "dt_D_y1y3",
            "dt_D_y2y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between_and_within,
            data_format,
            column_params_config,
        )

        # Extract feature dynamics
        fts_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        fd_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        window_length_1 = 4
        window_length_2 = 5
        fts_fcs_with_window_lengths = {
            window_length_1: fts_fcs,
            window_length_2: fts_fcs,
        }
        fts_fds_with_window_lengths = {window_length_1: fd_fcs, window_length_2: fd_fcs}
        X = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            feature_timeseries_fc_parameters=fts_fcs_with_window_lengths,
            feature_dynamics_fc_parameters=fts_fds_with_window_lengths,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )
        some_expected_feature_dynamics_names = (
            'dt_y2||quantile||q_0.2@window_5__fft_coefficient__attr_"real"__coeff_1',
            "y2||number_cwt_peaks||n_3@window_5__number_cwt_peaks__n_3",
            "D_y2y3||permutation_entropy||dimension_2||tau_1@window_4__quantile__q_0.2",
            'dt_D_y1y2||fft_coefficient||attr_"real"||coeff_1@window_4__number_cwt_peaks__n_3',
        )

        self.assertTrue(
            set(some_expected_feature_dynamics_names).issubset(X.columns.tolist())
        )
        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(len(X) == 10)
        # We cant make strong claims about the number of features produced because some feature timeseries have NaNs and are dropped

    def test_engineer_more_ts_and_then_extraction_on_pandas_long(self):

        data_format = "long"

        ts, response = self.gen_example_timeseries_data_for_e2e_tests(
            container_type="pandas", data_format=data_format
        )
        column_params_config = self.column_params_picker(data_format=data_format)

        # a) Engineer some more timeseries from input timeseries [D_y1y2, D_y1y3, D_y2y3]
        ts_with_extra_timeseries_between = diff_between_series(
            ts,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts = [
            "t",
            "measurement_id",
            "y1",
            "y2",
            "y3",
            "D_y1y2",
            "D_y1y3",
            "D_y2y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between,
            data_format,
            column_params_config,
        )

        # add an even extra layer of ts differencing
        ts_with_extra_timeseries_between_and_within = diff_within_series(
            ts_with_extra_timeseries_between,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts += [
            "dt_y1",
            "dt_y2",
            "dt_y3",
            "dt_D_y1y2",
            "dt_D_y1y3",
            "dt_D_y2y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between_and_within,
            data_format,
            column_params_config,
        )

        # Extract feature dynamics
        fts_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        fd_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        window_length_1 = 4
        window_length_2 = 5
        fts_fcs_with_window_lengths = {
            window_length_1: fts_fcs,
            window_length_2: fts_fcs,
        }
        fts_fds_with_window_lengths = {window_length_1: fd_fcs, window_length_2: fd_fcs}
        X = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            feature_timeseries_fc_parameters=fts_fcs_with_window_lengths,
            feature_dynamics_fc_parameters=fts_fds_with_window_lengths,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )
        some_expected_feature_dynamics_names = (
            'dt_y2||quantile||q_0.2@window_5__fft_coefficient__attr_"real"__coeff_1',
            "y2||number_cwt_peaks||n_3@window_5__number_cwt_peaks__n_3",
            "D_y2y3||permutation_entropy||dimension_2||tau_1@window_4__quantile__q_0.2",
            'dt_D_y1y2||fft_coefficient||attr_"real"||coeff_1@window_4__number_cwt_peaks__n_3',
        )

        self.assertTrue(
            set(some_expected_feature_dynamics_names).issubset(X.columns.tolist())
        )
        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(len(X) == 10)
        # We cant make strong claims about the number of features produced because some feature timeseries have NaNs and are dropped

    def test_engineer_more_ts_and_then_extraction_on_pandas_dict(self):

        data_format = "dict"

        ts, response = self.gen_example_timeseries_data_for_e2e_tests(
            container_type="pandas", data_format=data_format
        )
        column_params_config = self.column_params_picker(data_format=data_format)
        # a) Engineer some more timeseries from input timeseries [D_y1y2, D_y1y3, D_y2y3]
        ts_with_extra_timeseries_between = diff_between_series(
            ts,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts = [
            "t",
            "measurement_id",
            "y1",
            "y2",
            "y3",
            "D_y1y2",
            "D_y1y3",
            "D_y2y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between,
            data_format,
            column_params_config,
        )

        # add an even extra layer of ts differencing
        ts_with_extra_timeseries_between_and_within = diff_within_series(
            ts_with_extra_timeseries_between,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        # TODO: Ideally this test should be agnostic to the order i.e. if D_y1y2 is expected, then D_y2y1 should also pass this test too
        expected_ts += [
            "dt_y1",
            "dt_y2",
            "dt_y3",
            "dt_D_y1y2",
            "dt_D_y1y3",
            "dt_D_y2y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between_and_within,
            data_format,
            column_params_config,
        )

        # Extract feature dynamics
        fts_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        fd_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        window_length_1 = 4
        window_length_2 = 5
        fts_fcs_with_window_lengths = {
            window_length_1: fts_fcs,
            window_length_2: fts_fcs,
        }
        fts_fds_with_window_lengths = {window_length_1: fd_fcs, window_length_2: fd_fcs}
        X = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            feature_timeseries_fc_parameters=fts_fcs_with_window_lengths,
            feature_dynamics_fc_parameters=fts_fds_with_window_lengths,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )
        some_expected_feature_dynamics_names = (
            'dt_y2||quantile||q_0.2@window_5__fft_coefficient__attr_"real"__coeff_1',
            "y2||number_cwt_peaks||n_3@window_5__number_cwt_peaks__n_3",
            "D_y2y3||permutation_entropy||dimension_2||tau_1@window_4__quantile__q_0.2",
            'dt_D_y1y2||fft_coefficient||attr_"real"||coeff_1@window_4__number_cwt_peaks__n_3',
        )

        self.assertTrue(
            set(some_expected_feature_dynamics_names).issubset(X.columns.tolist())
        )
        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(len(X) == 10)
        # We cant make strong claims about the number of features produced because some feature timeseries have NaNs and are dropped


class FullFeatureDynamicsWorkflowTestCase(FixturesForFeatureDynamicsIntegrationTests):
    """
    Test the integrations related to the following workflow:

    a) Engineer more timeseries then
    b) Extract features then
    c) Select featues then
    d) Interpret the top relevant feature dynamics via PDF file then
    e) Extract relevant features on more timeseries data
    """

    # IMPORTANT TODO: Test that all 3 input formats will give the same extracted feature dynamics output, irrespective of input format.
    def test_full_feature_dynamics_workflow_pandas_wide(self):

        # Test the end to end process of engineer,extract, select, interpret, extract on selected
        # for pandas for the wide input format
        data_format = "wide"
        ts, response = self.gen_example_timeseries_data_for_e2e_tests(
            container_type="pandas", data_format=data_format
        )
        fts_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        fd_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        window_length_1 = 4
        window_length_2 = 5
        fts_fcs_with_window_lengths = {
            window_length_1: fts_fcs,
            window_length_2: fts_fcs,
        }
        fts_fds_with_window_lengths = {window_length_1: fd_fcs, window_length_2: fd_fcs}
        column_params_config = self.column_params_picker(data_format=data_format)

        # a) Engineer some more timeseries from input timeseries
        ts_with_extra_timeseries_within = diff_within_series(
            ts,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts = [
            "t",
            "measurement_id",
            "y1",
            "y2",
            "y3",
            "dt_y1",
            "dt_y2",
            "dt_y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_within,
            data_format,
            column_params_config,
        )

        # add an extra layer of ts differencing
        ts_with_extra_timeseries_between_and_within = diff_between_series(
            ts_with_extra_timeseries_within,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        # TODO: Ideally this test should be agnostic to the order i.e. if D_y1y2 is expected, then D_y2y1 should also pass this test too
        expected_ts += [
            "D_y1y2",
            "D_y1y3",
            "D_y2y3",
            "D_dt_y1dt_y2",
            "D_dt_y1dt_y3",
            "D_dt_y2dt_y3",
            "D_y2dt_y1",
            "D_y3dt_y1",
            "D_y3dt_y2",
            "D_y1dt_y1",
            "D_y2dt_y1",
            "D_y3dt_y1",
            "D_y1dt_y2",
            "D_y2dt_y2",
            "D_y3dt_y2",
            "D_y1dt_y3",
            "D_y2dt_y3",
            "D_y3dt_y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between_and_within,
            data_format,
            column_params_config,
        )

        # b) Extract
        X = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            feature_timeseries_fc_parameters=fts_fcs_with_window_lengths,
            feature_dynamics_fc_parameters=fts_fds_with_window_lengths,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )

        some_expected_feature_dynamics_names = (
            'D_dt_y1dt_y2||quantile||q_0.2@window_5__fft_coefficient__attr_"real"__coeff_1',
            "D_y3dt_y1||number_cwt_peaks||n_3@window_5__number_cwt_peaks__n_3",
            "y2||permutation_entropy||dimension_2||tau_1@window_4__quantile__q_0.2",
            'D_y2y3||fft_coefficient||attr_"real"||coeff_1@window_4__number_cwt_peaks__n_3',
        )

        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(len(X) == 10)
        self.assertTrue(
            set(some_expected_feature_dynamics_names).issubset(X.columns.tolist())
        )
        # We cant make strong claims about the number of different features produced because some feature timeseries have NaNs and are dropped

        # c) Select
        X_relevant = select_features(X, response, fdr_level=0.95)

        # d) Gen relevant features dictionaries
        rel_feature_names = list(X_relevant.columns)
        (
            rel_feature_time_series_dict,
            rel_feature_dynamics_dict,
        ) = derive_features_dictionaries(rel_feature_names)

        # e) Relevant features interpretation
        output_filename_prefix = "feature_dynamics_interpretation_test"
        gen_pdf_for_feature_dynamics(
            rel_feature_names, output_filename=output_filename_prefix
        )

        pdf_exists = os.path.exists(f"{output_filename_prefix}.pdf")
        markdown_exists = os.path.exists(f"{output_filename_prefix}.md")

        if pdf_exists:
            os.remove(f"{output_filename_prefix}.pdf")
        if markdown_exists:
            os.remove(f"{output_filename_prefix}.md")

        # f) extract on selected features
        X_more = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            n_jobs=0,
            feature_timeseries_kind_to_fc_parameters=rel_feature_time_series_dict,
            feature_dynamics_kind_to_fc_parameters=rel_feature_dynamics_dict,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )

        self.assertIsInstance(X_more, pd.DataFrame)
        self.assertTrue(len(X_more) == 10)
        # Check that feature vectors are the same no matter how they are extracted
        for feature_name in X_more.columns:
            pd.testing.assert_series_equal(
                X_more[feature_name], X[feature_name]
            )  # checking idempotency

    def test_full_feature_dynamics_workflow_pandas_long(self):
        # Test the end to end process of engineer,extract, select, interpret, extract on selected
        # for pandas for the long input format
        data_format = "long"
        ts, response = self.gen_example_timeseries_data_for_e2e_tests(
            container_type="pandas", data_format=data_format
        )
        fts_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        fd_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        window_length_1 = 4
        window_length_2 = 5
        fts_fcs_with_window_lengths = {
            window_length_1: fts_fcs,
            window_length_2: fts_fcs,
        }
        fts_fds_with_window_lengths = {window_length_1: fd_fcs, window_length_2: fd_fcs}
        column_params_config = self.column_params_picker(data_format=data_format)

        # a) Engineer some more timeseries from input timeseries
        ts_with_extra_timeseries_within = diff_within_series(
            ts,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts = [
            "t",
            "measurement_id",
            "y1",
            "y2",
            "y3",
            "dt_y1",
            "dt_y2",
            "dt_y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_within,
            data_format,
            column_params_config,
        )

        # add an extra layer of ts differencing
        # TODO: Ideally this test should be agnostic to the order i.e. if D_y1y2 is expected, then D_y2y1 should also pass this test too
        ts_with_extra_timeseries_between_and_within = diff_between_series(
            ts_with_extra_timeseries_within,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts += [
            "D_y1y2",
            "D_y1y3",
            "D_y2y3",
            "D_dt_y1dt_y2",
            "D_dt_y1dt_y3",
            "D_dt_y2dt_y3",
            "D_dt_y1y2",
            "D_dt_y1y3",
            "D_dt_y2y3",
            "D_dt_y1y1",
            "D_dt_y1y2",
            "D_dt_y1y3",
            "D_dt_y2y1",
            "D_dt_y2y2",
            "D_dt_y2y3",
            "D_dt_y3y1",
            "D_dt_y3y2",
            "D_dt_y3y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between_and_within,
            data_format,
            column_params_config,
        )

        # b) Extract
        X = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            feature_timeseries_fc_parameters=fts_fcs_with_window_lengths,
            feature_dynamics_fc_parameters=fts_fds_with_window_lengths,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )

        some_expected_feature_dynamics_names = (
            'D_dt_y1dt_y2||quantile||q_0.2@window_5__fft_coefficient__attr_"real"__coeff_1',
            "D_dt_y1y3||number_cwt_peaks||n_3@window_5__number_cwt_peaks__n_3",
            "y2||permutation_entropy||dimension_2||tau_1@window_4__quantile__q_0.2",
            'D_y2y3||fft_coefficient||attr_"real"||coeff_1@window_4__number_cwt_peaks__n_3',
        )

        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(len(X) == 10)
        self.assertTrue(
            set(some_expected_feature_dynamics_names).issubset(X.columns.tolist())
        )
        # We cant make strong claims about the number of different features produced because some feature timeseries have NaNs and are dropped

        # c) Select
        X_relevant = select_features(X, response, fdr_level=0.95)

        # d) Gen relevant features dictionaries
        rel_feature_names = list(X_relevant.columns)
        (
            rel_feature_time_series_dict,
            rel_feature_dynamics_dict,
        ) = derive_features_dictionaries(rel_feature_names)

        # e) Relevant features interpretation
        output_filename_prefix = "feature_dynamics_interpretation_test"
        gen_pdf_for_feature_dynamics(
            rel_feature_names, output_filename=output_filename_prefix
        )

        pdf_exists = os.path.exists(f"{output_filename_prefix}.pdf")
        markdown_exists = os.path.exists(f"{output_filename_prefix}.md")

        if pdf_exists:
            os.remove(f"{output_filename_prefix}.pdf")
        if markdown_exists:
            os.remove(f"{output_filename_prefix}.md")

        # f) extract on selected features
        X_more = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            n_jobs=0,
            feature_timeseries_kind_to_fc_parameters=rel_feature_time_series_dict,
            feature_dynamics_kind_to_fc_parameters=rel_feature_dynamics_dict,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )

        self.assertIsInstance(X_more, pd.DataFrame)
        self.assertTrue(len(X_more) == 10)
        # Check that feature vectors are the same no matter how they are extracted
        for feature_name in X_more.columns:
            pd.testing.assert_series_equal(
                X_more[feature_name], X[feature_name]
            )  # checking idempotency

    def test_full_feature_dynamics_workflow_pandas_dict(self):

        # Test the end to end process of engineer,extract, select, interpret, extract on selected
        # for pandas for the dict input format
        data_format = "dict"
        ts, response = self.gen_example_timeseries_data_for_e2e_tests(
            container_type="pandas", data_format=data_format
        )
        fts_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        fd_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        window_length_1 = 4
        window_length_2 = 5
        fts_fcs_with_window_lengths = {
            window_length_1: fts_fcs,
            window_length_2: fts_fcs,
        }
        fts_fds_with_window_lengths = {window_length_1: fd_fcs, window_length_2: fd_fcs}
        column_params_config = self.column_params_picker(data_format=data_format)

        # a) Engineer some more timeseries from input timeseries
        ts_with_extra_timeseries_within = diff_within_series(
            ts,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts = [
            "t",
            "measurement_id",
            "y1",
            "y2",
            "y3",
            "dt_y1",
            "dt_y2",
            "dt_y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_within,
            data_format,
            column_params_config,
        )

        # add an extra layer of ts differencing
        ts_with_extra_timeseries_between_and_within = diff_between_series(
            ts_with_extra_timeseries_within,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_value=column_params_config["column_value"],
            column_kind=column_params_config["column_kind"],
        )
        expected_ts += [
            "D_y1y2",
            "D_y1y3",
            "D_y2y3",
            "D_dt_y1dt_y2",
            "D_dt_y1dt_y3",
            "D_dt_y2dt_y3",
            "D_y2dt_y1",
            "D_y3dt_y1",
            "D_y3dt_y2",
            "D_y1dt_y1",
            "D_y2dt_y1",
            "D_y3dt_y1",
            "D_y1dt_y2",
            "D_y2dt_y2",
            "D_y3dt_y2",
            "D_y1dt_y3",
            "D_y2dt_y3",
            "D_y3dt_y3",
        ]
        self.check_correct_ts_are_engineered(
            expected_ts,
            ts_with_extra_timeseries_between_and_within,
            data_format,
            column_params_config,
        )

        # b) Extract
        X = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            feature_timeseries_fc_parameters=fts_fcs_with_window_lengths,
            feature_dynamics_fc_parameters=fts_fds_with_window_lengths,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )

        some_expected_feature_dynamics_names = (
            'D_dt_y1dt_y2||quantile||q_0.2@window_5__fft_coefficient__attr_"real"__coeff_1',
            "D_y3dt_y1||number_cwt_peaks||n_3@window_5__number_cwt_peaks__n_3",
            "y2||permutation_entropy||dimension_2||tau_1@window_4__quantile__q_0.2",
            'D_y2y3||fft_coefficient||attr_"real"||coeff_1@window_4__number_cwt_peaks__n_3',
        )

        self.assertIsInstance(X, pd.DataFrame)
        self.assertTrue(len(X) == 10)
        self.assertTrue(
            set(some_expected_feature_dynamics_names).issubset(X.columns.tolist())
        )
        # We cant make strong claims about the number of different features produced because some feature timeseries have NaNs and are dropped

        # c) Select
        X_relevant = select_features(X, response, fdr_level=0.95)

        # d) Generate relevant features dictionaries and interpret
        rel_feature_names = list(X_relevant.columns)
        (
            rel_feature_time_series_dict,
            rel_feature_dynamics_dict,
        ) = derive_features_dictionaries(rel_feature_names)

        # e) Relevant features interpretation
        output_filename_prefix = "feature_dynamics_interpretation_test"
        gen_pdf_for_feature_dynamics(
            rel_feature_names, output_filename=output_filename_prefix
        )

        pdf_exists = os.path.exists(f"{output_filename_prefix}.pdf")
        markdown_exists = os.path.exists(f"{output_filename_prefix}.md")

        if pdf_exists:
            os.remove(f"{output_filename_prefix}.pdf")
        if markdown_exists:
            os.remove(f"{output_filename_prefix}.md")

        # f) extract on selected features
        X_more = extract_feature_dynamics(
            timeseries_container=ts_with_extra_timeseries_between_and_within,
            n_jobs=0,
            feature_timeseries_kind_to_fc_parameters=rel_feature_time_series_dict,
            feature_dynamics_kind_to_fc_parameters=rel_feature_dynamics_dict,
            column_id=column_params_config["column_id"],
            column_sort=column_params_config["column_sort"],
            column_kind=column_params_config["column_kind"],
            column_value=column_params_config["column_value"],
        )

        self.assertIsInstance(X_more, pd.DataFrame)
        self.assertTrue(len(X_more) == 10)
        # Check that feature vectors are the same no matter how they are extracted
        for feature_name in X_more.columns:
            pd.testing.assert_series_equal(
                X_more[feature_name], X[feature_name]
            )  # checking idempotency

    def test_full_feature_dynamics_workflow_dask_long(self):
        # TODO: In progress
        pass

        ts, response = self.gen_example_timeseries_data_for_e2e_tests(
            container_type="dask", data_format="wide"
        )
        fts_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )
        fd_fcs = self.gen_feature_calculators_for_e2e_tests(
            feature_complexity="not-minimal"
        )

        window_length_1 = 4
        window_length_2 = 5
        fts_fcs_with_window_lengths = {
            window_length_1: fts_fcs,
            window_length_2: fts_fcs,
        }
        fts_fds_with_window_lengths = {window_length_1: fd_fcs, window_length_2: fd_fcs}

        # a) Extract
        X = extract_feature_dynamics(
            timeseries_container=ts,
            feature_timeseries_fc_parameters=fts_fcs_with_window_lengths,
            feature_dynamics_fc_parameters=fts_fds_with_window_lengths,
            column_id="measurement_id",
            column_sort="t",
            column_kind=None,
            column_value=None,
        )

        # b) Select
        X_pandas = X.compute()

        X_relevant = select_features(X_pandas, response, fdr_level=0.95)

        # c) Interpret
        rel_feature_names = list(X_relevant.columns)

        (
            rel_feature_time_series_dict,
            rel_feature_dynamics_dict,
        ) = derive_features_dictionaries(rel_feature_names)

        # d) extract on selected features
        X_more = extract_feature_dynamics(
            timeseries_container=ts,
            n_jobs=0,
            feature_timeseries_kind_to_fc_parameters=rel_feature_time_series_dict,
            feature_dynamics_kind_to_fc_parameters=rel_feature_dynamics_dict,
            column_id="measurement_id",
            column_sort="t",
            column_kind=None,
            column_value=None,
        )
