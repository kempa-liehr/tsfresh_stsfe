from unittest import TestCase
import os
from tests.fixtures import DataTestCase
import pandas as pd

from tsfresh.feature_dynamics_extraction.feature_dynamics_utils import (
    clean_feature_timeseries_name,
    update_feature_dictionary,
    parse_feature_timeseries_parts,
    parse_feature_dynamics_parts,
    derive_features_dictionaries,
    engineer_input_timeseries,
    interpret_feature_dynamic,
    dictionary_to_string,
    gen_pdf_for_feature_dynamics,
)


class FeatureDynamicsStringManipulationTestCase(TestCase):
    """"""

    def test_clean_feature_timeseries_name(self):
        window_length = 15
        fts_name_inputs = (
            "y__energy_ratio_by_chunks__num_segments_10__segment_focus_0",
            "y__number_crossing_m__m_0",
            'y__fft_coefficient__attr_"angle"__coeff_89',
            'y__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0',
            "y__permutation_entropy__dimension_5__tau_1",
        )

        expected_cleaned_fts_names = (
            f"y||energy_ratio_by_chunks|num_segments_10|segment_focus_0@window_{window_length}",
            f"y||number_crossing_m|m_0@window_{window_length}",
            f'y||fft_coefficient|attr_"angle"|coeff_89@window_{window_length}',
            f'y||change_quantiles|f_agg_"var"|isabs_False|qh_0.2|ql_0.0@window_{window_length}',
            f"y||permutation_entropy|dimension_5|tau_1@window_{window_length}",
        )

        feature_timeseries_cleaned_name_outputs = tuple(
            clean_feature_timeseries_name(input_fts_name, window_length)
            for input_fts_name in fts_name_inputs
        )
        self.assertTrue(
            feature_timeseries_cleaned_name_outputs == expected_cleaned_fts_names
        )

    def test_update_feature_dictionary(self):

        # test empty dictionary
        dictionary_1 = {}
        window_length_1 = 3
        feature_parts_1 = [
            "y",
            "change_quantiles",
            'f_agg_"var"',
            "isabs_False",
            "qh_0.2",
            "ql_0.0",
        ]
        updated_dictionary_1 = update_feature_dictionary(
            dictionary_1, window_length_1, feature_parts_1
        )
        expected_updated_dictionary_1 = {
            window_length_1: {
                feature_parts_1[0]: {
                    feature_parts_1[1]: [
                        {"f_agg": "var", "isabs": False, "qh": 0.2, "ql": 0.0}
                    ]
                }
            }
        }
        self.assertTrue(updated_dictionary_1 == expected_updated_dictionary_1)

        # test dictionary with entries already, with a new param
        dictionary_2 = {
            4: {"y": {"change_quantiles": [{"f_agg": "var"}]}},
            10: {"y": {"change_quantiles": [{"isabs": False}]}},
        }
        window_length_2 = 10
        feature_parts_2 = ["y", "change_quantiles", "qh_0.2"]
        updated_dictionary_2 = update_feature_dictionary(
            dictionary_2, window_length_2, feature_parts_2
        )
        expected_updated_dictionary_2 = {
            4: {"y": {"change_quantiles": [{"f_agg": "var"}]}},
            10: {"y": {"change_quantiles": [{"isabs": False}, {"qh": 0.2}]}},
        }
        self.assertTrue(updated_dictionary_2 == expected_updated_dictionary_2)

        # add a dictionary with a new window to dictionary 2
        window_length_3 = 100
        updated_dictionary_3 = update_feature_dictionary(
            expected_updated_dictionary_2, window_length_3, feature_parts_2
        )
        expected_updated_dictionary_3 = {
            4: {"y": {"change_quantiles": [{"f_agg": "var"}]}},
            10: {"y": {"change_quantiles": [{"isabs": False}, {"qh": 0.2}]}},
            100: {"y": {"change_quantiles": [{"qh": 0.2}]}},
        }
        self.assertTrue(updated_dictionary_3 == expected_updated_dictionary_3)

        # Add a new feature name to dictionary 3
        feature_parts_3 = ["y", "permutation_entropy", "dimension_5", "tau_1"]
        updated_dictionary_4 = update_feature_dictionary(
            expected_updated_dictionary_3, window_length_3, feature_parts_3
        )
        expected_updated_dictionary_4 = {
            4: {"y": {"change_quantiles": [{"f_agg": "var"}]}},
            10: {"y": {"change_quantiles": [{"isabs": False}, {"qh": 0.2}]}},
            100: {
                "y": {
                    "change_quantiles": [{"qh": 0.2}],
                    "permutation_entropy": [{"dimension": 5, "tau": 1}],
                }
            },
        }
        self.assertTrue(updated_dictionary_4 == expected_updated_dictionary_4)

        # Add a new ts kind to dictionary 4
        feature_parts_4 = ["z", "permutation_entropy", "dimension_5", "tau_1"]
        updated_dictionary_5 = update_feature_dictionary(
            expected_updated_dictionary_4,
            window_length=4,
            feature_parts=feature_parts_4,
        )
        expected_updated_dictionary_5 = {
            4: {
                "z": {"permutation_entropy": [{"dimension": 5, "tau": 1}]},
                "y": {"change_quantiles": [{"f_agg": "var"}]},
            },
            10: {"y": {"change_quantiles": [{"isabs": False}, {"qh": 0.2}]}},
            100: {
                "y": {
                    "change_quantiles": [{"qh": 0.2}],
                    "permutation_entropy": [{"dimension": 5, "tau": 1}],
                }
            },
        }
        self.assertTrue(updated_dictionary_5 == expected_updated_dictionary_5)

        # Check similar and identical params are handled correctly
        dictionary_to_repeatedly_mutate = {}
        window_length_4 = 1
        feature_parts_5 = ["y", "change_quantiles", "dimension_5", "tau_1"]
        feature_parts_6 = ["y", "change_quantiles", "dimension_5", "tau_1", "qh_0.2"]
        for feature_parts_n in [feature_parts_5, feature_parts_6]:
            dictionary_to_repeatedly_mutate = update_feature_dictionary(
                dictionary_to_repeatedly_mutate, window_length_4, feature_parts_n
            )
        expected_dictionary_to_repeatedly_mutate = {
            1: {
                "y": {
                    "change_quantiles": [
                        {"dimension": 5, "tau": 1},
                        {"dimension": 5, "tau": 1, "qh": 0.2},
                    ]
                }
            }
        }
        self.assertTrue(
            dictionary_to_repeatedly_mutate == expected_dictionary_to_repeatedly_mutate
        )

        # TODO: Test giving it an illegal feature name...
        # TODO: Test giving it weird window length
        # TODO: Test giving it a duplicate feature

    def test_parse_feature_timeseries_parts(self):
        full_feature_names_inputs = (
            "x||ratio_beyond_r_sigma|r_2@window_10__energy_ratio_by_chunks__num_segments_10__segment_focus_3",
            "y||variance_larger_than_standard_deviation@window_200__lempel_ziv_complexity__bins_5",
            "z||permutation_entropy|dimension_5|tau_1@window_800__symmetry_looking__r_0.35000000000000003",
            'x||value_count|value_1@window_10__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
            'y||time_reversal_asymmetry_statistic|lag_3@window_20__fft_coefficient__attr_"imag"__coeff_1',
            "x||ratio_value_number_to_time_series_length@window_5__range_count__max_1__min_-1",
        )
        expected_fts_parts_outputs = (
            {"window_length": 10, "fts_parts": ["x", "ratio_beyond_r_sigma", "r_2"]},
            {
                "window_length": 200,
                "fts_parts": ["y", "variance_larger_than_standard_deviation"],
            },
            {
                "window_length": 800,
                "fts_parts": ["z", "permutation_entropy", "dimension_5", "tau_1"],
            },
            {"window_length": 10, "fts_parts": ["x", "value_count", "value_1"]},
            {
                "window_length": 20,
                "fts_parts": ["y", "time_reversal_asymmetry_statistic", "lag_3"],
            },
            {
                "window_length": 5,
                "fts_parts": ["x", "ratio_value_number_to_time_series_length"],
            },
        )
        actual_fts_parts_outputs = tuple(
            parse_feature_timeseries_parts(full_feature_name)
            for full_feature_name in full_feature_names_inputs
        )
        self.assertTrue(expected_fts_parts_outputs == actual_fts_parts_outputs)

        # TODO: Test some stuff that should fail

    def test_parse_feature_dynamics_parts(self):

        full_feature_names_inputs = (
            "x||ratio_beyond_r_sigma|r_2@window_10__energy_ratio_by_chunks__num_segments_10__segment_focus_3",
            "y||variance_larger_than_standard_deviation@window_200__lempel_ziv_complexity__bins_5",
            "z||permutation_entropy|dimension_5|tau_1@window_800__symmetry_looking__r_0.35000000000000003",
            'x||value_count|value_1@window_10__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
            'y||time_reversal_asymmetry_statistic|lag_3@window_20__fft_coefficient__attr_"imag"__coeff_1',
            "x||ratio_value_number_to_time_series_length@window_5__range_count__max_1__min_-1",
        )
        expected_fd_parts_outputs = (
            {
                "fd_parts": [
                    "x||ratio_beyond_r_sigma|r_2@window_10",
                    "energy_ratio_by_chunks",
                    "num_segments_10",
                    "segment_focus_3",
                ]
            },
            {
                "fd_parts": [
                    "y||variance_larger_than_standard_deviation@window_200",
                    "lempel_ziv_complexity",
                    "bins_5",
                ]
            },
            {
                "fd_parts": [
                    "z||permutation_entropy|dimension_5|tau_1@window_800",
                    "symmetry_looking",
                    "r_0.35000000000000003",
                ]
            },
            {
                "fd_parts": [
                    "x||value_count|value_1@window_10",
                    "change_quantiles",
                    'f_agg_"var"',
                    "isabs_False",
                    "qh_0.8",
                    "ql_0.4",
                ]
            },
            {
                "fd_parts": [
                    "y||time_reversal_asymmetry_statistic|lag_3@window_20",
                    "fft_coefficient",
                    'attr_"imag"',
                    "coeff_1",
                ]
            },
            {
                "fd_parts": [
                    "x||ratio_value_number_to_time_series_length@window_5",
                    "range_count",
                    "max_1",
                    "min_-1",
                ]
            },
        )
        full_feature_names_outputs = tuple(
            parse_feature_dynamics_parts(full_feature_name_input)
            for full_feature_name_input in full_feature_names_inputs
        )
        self.assertTrue(full_feature_names_outputs == expected_fd_parts_outputs)

        # TODO: Potentially add some more tests to validate

    def test_derive_features_dictionaries(self):
        full_feature_names_inputs = (
            "x||ratio_beyond_r_sigma|r_2@window_10__energy_ratio_by_chunks__num_segments_10__segment_focus_3",
            "y||variance_larger_than_standard_deviation@window_200__lempel_ziv_complexity__bins_5",
            "z||permutation_entropy|dimension_5|tau_1@window_800__symmetry_looking__r_0.35000000000000003",
            'x||value_count|value_1@window_10__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
            'y||time_reversal_asymmetry_statistic|lag_3@window_20__fft_coefficient__attr_"imag"__coeff_1',
            "x||ratio_value_number_to_time_series_length@window_5__range_count__max_1__min_-1",
        )

        fts_dict, fd_dict = derive_features_dictionaries(full_feature_names_inputs)

        expected_fts_dict = {
            10: {
                "x": {"ratio_beyond_r_sigma": [{"r": 2}], "value_count": [{"value": 1}]}
            },
            200: {"y": {"variance_larger_than_standard_deviation": None}},
            800: {"z": {"permutation_entropy": [{"dimension": 5, "tau": 1}]}},
            20: {"y": {"time_reversal_asymmetry_statistic": [{"lag": 3}]}},
            5: {"x": {"ratio_value_number_to_time_series_length": None}},
        }
        expected_fd_dict = {
            10: {
                "x||ratio_beyond_r_sigma|r_2@window_10": {
                    "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 3}]
                },
                "x||value_count|value_1@window_10": {
                    "change_quantiles": [
                        {"f_agg": "var", "isabs": False, "qh": 0.8, "ql": 0.4}
                    ]
                },
            },
            200: {
                "y||variance_larger_than_standard_deviation@window_200": {
                    "lempel_ziv_complexity": [{"bins": 5}]
                }
            },
            800: {
                "z||permutation_entropy|dimension_5|tau_1@window_800": {
                    "symmetry_looking": [{"r": 0.35000000000000003}]
                }
            },
            20: {
                "y||time_reversal_asymmetry_statistic|lag_3@window_20": {
                    "fft_coefficient": [{"attr": "imag", "coeff": 1}]
                }
            },
            5: {
                "x||ratio_value_number_to_time_series_length@window_5": {
                    "range_count": [{"max": 1, "min": -1}]
                }
            },
        }

        self.assertTrue(fts_dict == expected_fts_dict)
        self.assertTrue(fd_dict == expected_fd_dict)

    def test_interpret_feature_dynamic(self):

        # Test input
        # TODO: Could factor out the test input as it is used in multiple different testing functions
        full_feature_names_inputs = (
            "x||ratio_beyond_r_sigma|r_2@window_10__energy_ratio_by_chunks__num_segments_10__segment_focus_3",
            "y||variance_larger_than_standard_deviation@window_200__lempel_ziv_complexity__bins_5",
            "z||permutation_entropy|dimension_5|tau_1@window_800__symmetry_looking__r_0.35000000000000003",
            'x||value_count|value_1@window_10__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
            'y||time_reversal_asymmetry_statistic|lag_3@window_20__fft_coefficient__attr_"imag"__coeff_1',
            "x||ratio_value_number_to_time_series_length@window_5__range_count__max_1__min_-1",
        )

        # Expected values
        expected_multiple_input_timeseries = ("x", "y", "z", "x", "y", "x")
        expected_feature_timeseries_calculators = (
            {"ratio_beyond_r_sigma": [{"r": 2}]},
            {"variance_larger_than_standard_deviation": None},
            {"permutation_entropy": [{"dimension": 5, "tau": 1}]},
            {"value_count": [{"value": 1}]},
            {"time_reversal_asymmetry_statistic": [{"lag": 3}]},
            {"ratio_value_number_to_time_series_length": None},
        )
        expected_window_lengths = (10, 200, 800, 10, 20, 5)
        expected_feature_dynamic_calculators = (
            {"energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": 3}]},
            {"lempel_ziv_complexity": [{"bins": 5}]},
            {"symmetry_looking": [{"r": 0.35000000000000003}]},
            {
                "change_quantiles": [
                    {"f_agg": "var", "isabs": False, "qh": 0.8, "ql": 0.4}
                ]
            },
            {"fft_coefficient": [{"attr": "imag", "coeff": 1}]},
            {"range_count": [{"max": 1, "min": -1}]},
        )

        # Combine expected values into the expected intepretation
        expected_intepreted_feature_dynamics = tuple(
            {
                "Full Feature Dynamic Name": full_feature_dynamic_name,
                "Input Timeseries": expected_single_input_timeseries,
                "Feature Timeseries Calculator": expected_feature_timeseries_calculator,
                "Window Length": expected_window_length,
                "Feature Dynamic Calculator": expected_feature_dynamic_calculator,
            }
            for (
                full_feature_dynamic_name,
                expected_single_input_timeseries,
                expected_feature_timeseries_calculator,
                expected_window_length,
                expected_feature_dynamic_calculator,
            ) in zip(
                full_feature_names_inputs,
                expected_multiple_input_timeseries,
                expected_feature_timeseries_calculators,
                expected_window_lengths,
                expected_feature_dynamic_calculators,
            )
        )
        # Get the actual interpretation
        actual_interpreted_feature_dynamics = tuple(
            interpret_feature_dynamic(full_feature_name_input)
            for full_feature_name_input in full_feature_names_inputs
        )
        self.assertTrue(
            actual_interpreted_feature_dynamics == expected_intepreted_feature_dynamics
        )

    def test_dictionary_to_string(self):
        dictionary_input = {
            "This should be bold": "This should be italicised",
            "This should also be bold": "This should also be italicised",
            500: 600,
            "500": "600",
            float(500.00): float(600.00),
            True: False,
        }

        expected_string_output = "**This should be bold** : ```This should be italicised```<br>**This should also be bold** : ```This should also be italicised```<br>**500** : ```600.0```<br>**500** : ```600```<br>**True** : ```False```<br>"
        actual_string_output = dictionary_to_string(dictionary_input)
        self.assertTrue(actual_string_output == expected_string_output)

        # TODO: Test to make it break if a strange type is given i.e. not float int str or bool
        # TODO: Test to break when there are unhashable types

    def test_gen_pdf_for_feature_dynamics(self):
        # This unit test is just going to call the
        # function. If there is no error, then the test passes!
        # NOTE: Not sure if this is the correct way to unit test
        # something that writes to a file but this is my best guess.

        full_feature_names_inputs = (  # Could/should factor this out into a fixture as used multiple times
            "x||ratio_beyond_r_sigma|r_2@window_10__energy_ratio_by_chunks__num_segments_10__segment_focus_3",
            "y||variance_larger_than_standard_deviation@window_200__lempel_ziv_complexity__bins_5",
            "z||permutation_entropy|dimension_5|tau_1@window_800__symmetry_looking__r_0.35000000000000003",
            'x||value_count|value_1@window_10__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
            'y||time_reversal_asymmetry_statistic|lag_3@window_20__fft_coefficient__attr_"imag"__coeff_1',
            "x||ratio_value_number_to_time_series_length@window_5__range_count__max_1__min_-1",
        )

        output_filename_prefix = "feature_dynamics_interpretation_test"
        gen_pdf_for_feature_dynamics(
            full_feature_names_inputs, output_filename=output_filename_prefix
        )

        pdf_exists = os.path.exists(f"{output_filename_prefix}.pdf")
        markdown_exists = os.path.exists(f"{output_filename_prefix}.md")

        if pdf_exists:
            os.remove(f"{output_filename_prefix}.pdf")
        if markdown_exists:
            os.remove(f"{output_filename_prefix}.md")

        self.assertTrue(pdf_exists and markdown_exists)


class EngineerTimeSeriesTestCase(DataTestCase):
    """"""

    # Test all three input formats

    def test_engineer_input_timeseries_flat_dataframe(self):
        # Input format 1: A Flat dataframe (if column_value and column_kind are both set to None)
        id = [1, 1, 1, 2, 2, 2]
        sort = [1, 2, 3, 1, 2, 3]
        y1 = [1, 3, 27, 18, 12, -34]
        y2 = [-10, 0, 1, 3, 14, 12]
        y3 = [6, 5, 4, 3, 2, 1]
        flat_timeseries_container = pd.DataFrame(
            {"id": id, "sort": sort, "y1": y1, "y2": y2, "y3": y3}
        )

        print("flat_dataframe goes to WideTsFrameAdapter")

        engineered_ts = engineer_input_timeseries(
            timeseries_container=flat_timeseries_container,
            column_sort="sort",
            column_id="id",
            column_value=None,
            column_kind=None,
            compute_differences_within_series=True,
            compute_differences_between_series=False,
        )

        engineered_ts = engineer_input_timeseries(
            timeseries_container=flat_timeseries_container,
            column_sort="sort",
            column_id="id",
            column_value=None,
            column_kind=None,
            compute_differences_within_series=False,
            compute_differences_between_series=True,
        )

        engineered_ts = engineer_input_timeseries(
            timeseries_container=flat_timeseries_container,
            column_sort="sort",
            column_id="id",
            column_value=None,
            column_kind=None,
            compute_differences_within_series=True,
            compute_differences_between_series=False,
        )

        expected_engineered_ts = 0

        self.assertTrue(engineered_ts.equal(expected_engineered_ts))

    def test_engineer_input_timeseries_stacked_dataframe(self):
        # Input format 2: A stacked dataframe (if column_value and column_kind are set)
        stacked_timeseries_container = self.create_test_data_sample()
        print(stacked_timeseries_container)

        engineered_ts = engineer_input_timeseries(
            timeseries_container=stacked_timeseries_container,
            column_sort="sort",
            column_id="id",
            column_kind="kind",
            column_value="val",
            compute_differences_within_series=True,
            compute_differences_between_series=False,
        )

        engineered_ts = engineer_input_timeseries(
            timeseries_container=stacked_timeseries_container,
            column_sort="sort",
            column_id="id",
            column_kind="kind",
            column_value="val",
            compute_differences_within_series=False,
            compute_differences_between_series=True,
        )

        engineered_ts = engineer_input_timeseries(
            timeseries_container=stacked_timeseries_container,
            column_sort="sort",
            column_id="id",
            column_kind="kind",
            column_value="val",
            compute_differences_within_series=True,
            compute_differences_between_series=True,
        )

        expected_engineered_ts = 0

        self.assertTrue(engineered_ts.equal(expected_engineered_ts))

    def test_engineer_input_timeseries_flat_dictionary(self):
        # Input format 3: A dictionary of flat dataframes (if only column_kind is set to none, and a dictionary is passed)
        # Maybe TODO: Make the test out of syncrony for each timeseries or not?

        class testable_dictionary_of_dataframes(dict[pd.DataFrame]):
            def __eq__(self, other_dictionary_of_dataframes):
                for key in self.keys():
                    if key not in other_dictionary_of_dataframes:
                        return False
                    elif self[key].equals(other_dictionary_of_dataframes[key]) is False:
                        return False
                return True

        # Set up input time series
        id = [1, 1, 1, 2, 2, 2]
        sort = [1, 2, 3, 1, 2, 3]
        y1 = [1, 3, 27, 18, 12, -34]
        y2 = [-10, 0, 1, 3, 14, 12]
        y3 = [6, 5, 4, 3, 2, 1]
        ys = {"y1": y1, "y2": y2, "y3": y3}
        dictionary_timeseries_container = {
            y_name: pd.DataFrame({"id": id, "sort": sort, "value": y_values})
            for (y_name, y_values) in ys.items()
        }

        # Test differences within
        engineered_ts_within = engineer_input_timeseries(
            timeseries_container=dictionary_timeseries_container,
            differences_type="within",
            column_sort="sort",
            column_id="id",
            column_value="value",
            column_kind=None,
        )

        expected_dt_y1 = [0.0, 2.0, 24.0, -9.0, -6.0, -46.0]
        expected_dt_y2 = [0.0, 10.0, 1.0, 2.0, 11.0, -2.0]
        expected_dt_y3 = [0.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        expected_ys = {
            "y1": y1,
            "y2": y2,
            "y3": y3,
            "dt_y1": expected_dt_y1,
            "dt_y2": expected_dt_y2,
            "dt_y3": expected_dt_y3,
        }
        expected_engineered_ts_within = {
            y_name: pd.DataFrame({"id": id, "sort": sort, "value": y_values})
            for (y_name, y_values) in expected_ys.items()
        }
        self.assertTrue(
            testable_dictionary_of_dataframes(engineered_ts_within)
            == testable_dictionary_of_dataframes(expected_engineered_ts_within)
        )

        # Test differences between
        engineered_ts_between = engineer_input_timeseries(
            timeseries_container=dictionary_timeseries_container,
            differences_type="between",
            column_sort="sort",
            column_id="id",
            column_value="value",
            column_kind=None,
        )

        expected_D_y1y2 = [11, 3, 26, 15, -2, -46]
        expected_D_y1y3 = [-5, -2, 23, 15, 10, -35]
        expected_D_y2y3 = [-16, -5, -3, 0, 12, 11]
        expected_ys = {
            "y1": y1,
            "y2": y2,
            "y3": y3,
            "D_y1y2": expected_D_y1y2,
            "D_y1y3": expected_D_y1y3,
            "D_y2y3": expected_D_y2y3,
        }
        expected_engineered_ts_between = {
            y_name: pd.DataFrame({"id": id, "sort": sort, "value": y_values})
            for (y_name, y_values) in expected_ys.items()
        }

        self.assertTrue(
            testable_dictionary_of_dataframes(engineered_ts_between)
            == testable_dictionary_of_dataframes(expected_engineered_ts_between)
        )

    # def test_engineer_input_timeseries(self):
    #     # NOTE: Final version of this unit test should do the following:
    #     # 1. Handle any of the 3 time series data input formats
    #     # 2. Handle irregular time series elegantly
    #     # 3. Handle NaNs in the input timeseries elegantly (maybe - might be better as the function caller's responsibility to clean)

    #     # VERY simple example (TODO: Add more examples that cover the cases especially using self.create_test_data_sample())
    #     id = [1, 1, 1, 2, 2, 2]
    #     sort = [1, 2, 3, 1, 2, 3]
    #     y1 = [1, 3, 27, 18, 12, -34]
    #     y2 = [-10, 0, 1, 3, 14, 12]
    #     y3 = [6, 5, 4, 3, 2, 1]
    #     ts = pd.DataFrame({"id": id, "sort": sort, "y1": y1, "y2": y2, "y3": y3})
    #     # Expected outputs:
    #     dt_y1 = [0.0, 2.0, 24.0, -9.0, -6.0, -46.0]
    #     dt_y2 = [0.0, 10.0, 1.0, 2.0, 11.0, -2.0]
    #     dt_y3 = [0.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    #     D_y1y2 = [11, 3, 26, 15, -2, -46]
    #     D_y1y3 = [-5, -2, 23, 15, 10, -35]
    #     D_y2y3 = [-16, -5, -3, 0, 12, 11]

    #     ts_expected_1 = pd.DataFrame(
    #         {
    #             "y1": y1,
    #             "y2": y2,
    #             "y3": y3,
    #             "dt_y1": dt_y1,
    #             "dt_y2": dt_y2,
    #             "dt_y3": dt_y3,
    #             "id": id,
    #             "sort": sort,
    #         }
    #     )
    #     ts_actual_1 = engineer_input_timeseries(
    #         timeseries_container=ts,
    #         column_sort="sort",
    #         column_id="id",
    #         compute_differences_within_series=True,
    #         compute_differences_between_series=False,
    #     )
    #     self.assertTrue(ts_actual_1.equals(ts_expected_1))

    #     ts_expected_2 = pd.DataFrame(
    #         {
    #             "y1": y1,
    #             "y2": y2,
    #             "y3": y3,
    #             "D_y1y2": D_y1y2,
    #             "D_y1y3": D_y1y3,
    #             "D_y2y3": D_y2y3,
    #             "id": id,
    #             "sort": sort,
    #         }
    #     )
    #     ts_actual_2 = engineer_input_timeseries(
    #         timeseries_container=ts,
    #         column_sort="sort",
    #         column_id="id",
    #         compute_differences_within_series=False,
    #         compute_differences_between_series=True,
    #     )
    #     self.assertTrue(ts_actual_2.equals(ts_expected_2))

    #     ts_expected_3 = pd.DataFrame(
    #         {
    #             "y1": y1,
    #             "y2": y2,
    #             "y3": y3,
    #             "dt_y1": dt_y1,
    #             "dt_y2": dt_y2,
    #             "dt_y3": dt_y3,
    #             "D_y1y2": D_y1y2,
    #             "D_y1y3": D_y1y3,
    #             "D_y2y3": D_y2y3,
    #             "id": id,
    #             "sort": sort,
    #         }
    #     )
    #     ts_actual_3 = engineer_input_timeseries(
    #         timeseries_container=ts,
    #         column_sort="sort",
    #         column_id="id",
    #         compute_differences_within_series=True,
    #         compute_differences_between_series=True,
    #     )
    #     self.assertTrue(ts_actual_3.equals(ts_expected_3))

    #     # TODO: More tests are absolutely neccessary, and should be able to take in ANY timeseries container,
    #     # and have suitable input validation checks
