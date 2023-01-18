from unittest import TestCase
import os
from tests.fixtures import DataTestCase
import pandas as pd
from typing import Dict
import copy

from tsfresh.feature_dynamics_extraction.feature_dynamics_utils import (
    clean_feature_timeseries_name,
    update_feature_dictionary,
    parse_feature_timeseries_parts,
    parse_feature_dynamics_parts,
    derive_features_dictionaries,
    diff_between_series,
    diff_within_series,
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

    def test_diff_within_series_flat(self):
        (
            flat_timeseries_container,
            ((column_id, column_sort, column_kind, column_value)),
            (y1, y2, y3),
        ) = self.create_simple_test_data_sample_wide(randomise_sort_order = False)

        expected_unmodified_data = flat_timeseries_container.copy(deep=True)

        engineered_ts_within = diff_within_series(
            timeseries_container=flat_timeseries_container,
            column_sort=column_sort,
            column_id=column_id,
            column_value=column_value,
            column_kind=column_kind,
        )
        expected_dt_y1 = [0.0, 2.0, 24.0, 0, -6.0, -46.0]
        expected_dt_y2 = [0.0, 10.0, 1.0, 0, 11.0, -2.0]
        expected_dt_y3 = [0.0, -1.0, -1.0, 0, -1.0, -1.0]
        expected_engineered_ts_within = pd.DataFrame(
            {
                column_id: flat_timeseries_container[column_id].tolist(),
                column_sort: flat_timeseries_container[column_sort].tolist(),
                "y1": y1,
                "y2": y2,
                "y3": y3,
                "dt_y1": expected_dt_y1,
                "dt_y2": expected_dt_y2,
                "dt_y3": expected_dt_y3,
            }
        )

        print("actual")
        print(engineered_ts_within)
        print("EXPECTED")
        print(engineered_ts_within)


        pd.testing.assert_frame_equal(engineered_ts_within, expected_engineered_ts_within, check_like=False)

        # Also check that the original input was not inadvertently mutated
        # i.e. check that the modifications return whole new objects instead of mutating
        # the original input df
        pd.testing.assert_frame_equal(flat_timeseries_container, expected_unmodified_data)

    def test_differences_within_stacked_dataframe(self):
        (
            stacked_dataframe_timeseries_container,
            (column_id, column_sort, column_kind, column_value),
        ) = self.create_simple_test_data_sample_stacked(randomise_sort_order = False)

        expected_unmodified_data = stacked_dataframe_timeseries_container.copy(deep=True)

        engineered_ts_within = diff_within_series(
            timeseries_container=stacked_dataframe_timeseries_container,
            column_sort=column_sort,
            column_id=column_id,
            column_kind=column_kind,
            column_value=column_value,
        )

        expected_dt_y1 = [0.0, 2.0, 24.0, 0, -6.0, -46.0]
        expected_dt_y2 = [0.0, 10.0, 1.0, 0, 11.0, -2.0]
        expected_dt_y3 = [0.0, -1.0, -1.0, 0, -1.0, -1.0]

        expected_within_values = expected_dt_y1 + expected_dt_y2 + expected_dt_y3
        expected_within_kinds = 6 * ["dt_y1"] + 6 * ["dt_y2"] + 6 * ["dt_y3"]

        expected_engineered_ts_within = pd.concat(
            [
                stacked_dataframe_timeseries_container,
                pd.DataFrame(
                    {
                        column_id: stacked_dataframe_timeseries_container[
                            column_id
                        ].tolist(),
                        column_sort: stacked_dataframe_timeseries_container[
                            column_sort
                        ].tolist(),
                        column_kind: expected_within_kinds,
                        column_value: expected_within_values,
                    }
                ),
            ]
        ).reset_index(drop=True)

        print("actual")
        print(engineered_ts_within)
        print("EXPECTED")
        print(engineered_ts_within)

        pd.testing.assert_frame_equal(engineered_ts_within, expected_engineered_ts_within, check_like=False)

        # Also check that the original input was not inadvertently mutated
        # i.e. check that the modifications return whole new objects instead of mutating
        # the original input df
        pd.testing.assert_frame_equal(stacked_dataframe_timeseries_container, expected_unmodified_data)

    def test_differences_within_dictionary(self):
        (
            dict_timeseries_container,
            (column_id, column_sort, column_kind, column_value),
            (y1, y2, y3),
            (id_values, sort_values),
        ) = self.create_simple_test_data_sample_dict(randomise_sort_order = False)
        
        expected_unmodified_data = copy.deepcopy(dict_timeseries_container)

        engineered_ts_within = diff_within_series(
            timeseries_container=dict_timeseries_container,
            column_sort=column_sort,
            column_id=column_id,
            column_value=column_value,
            column_kind=column_kind,
        )
        expected_dt_y1 = [0.0, 2.0, 24.0, 0, -6.0, -46.0]
        expected_dt_y2 = [0.0, 10.0, 1.0, 0, 11.0, -2.0]
        expected_dt_y3 = [0.0, -1.0, -1.0, 0, -1.0, -1.0]
        expected_ys = {
            "y1": y1,
            "y2": y2,
            "y3": y3,
            "dt_y1": expected_dt_y1,
            "dt_y2": expected_dt_y2,
            "dt_y3": expected_dt_y3,
        }
        expected_engineered_ts_within = {
            y_name: pd.DataFrame(
                {column_id: id_values, column_sort: sort_values, column_value: y_values}
            )
            for (y_name, y_values) in expected_ys.items()
        }

        print("actual")
        print(engineered_ts_within)

        print("EXPECTED")
        print(engineered_ts_within)

        self.assertTrue(
            testable_dictionary_of_dataframes(engineered_ts_within)
            == testable_dictionary_of_dataframes(expected_engineered_ts_within)
        )

        # Also check that the original input was not inadvertently mutated
        # i.e. check that the modifications return whole new objects instead of mutating
        # the original input df
        self.assertTrue(
            testable_dictionary_of_dataframes(dict_timeseries_container)
            == testable_dictionary_of_dataframes(expected_unmodified_data)
        )

    def test_diff_between_series_flat(self):
        (
            flat_timeseries_container,
            (column_id, column_sort, column_kind, column_value),
            (y1, y2, y3),
        ) = self.create_simple_test_data_sample_wide(randomise_sort_order = False)

        expected_unmodified_data = flat_timeseries_container.copy(deep=True)

        engineered_ts_between = diff_between_series(
            timeseries_container=flat_timeseries_container,
            column_sort=column_sort,
            column_id=column_id,
            column_value=column_value,
            column_kind=column_kind,
        )

        expected_D_y1y2 = [11, 3, 26, 15, -2, -46]
        expected_D_y1y3 = [-5, -2, 23, 15, 10, -35]
        expected_D_y2y3 = [-16, -5, -3, 0, 12, 11]

        expected_engineered_ts_between = pd.DataFrame(
            {
                column_id: flat_timeseries_container[column_id].tolist(),
                column_sort: flat_timeseries_container[column_sort].tolist(),
                "y1": y1,
                "y2": y2,
                "y3": y3,
                "D_y1y2": expected_D_y1y2,
                "D_y1y3": expected_D_y1y3,
                "D_y2y3": expected_D_y2y3,
            }
        )

        print("actual")
        print(engineered_ts_between)

        print("EXPECTED")
        print(expected_engineered_ts_between)

        pd.testing.assert_frame_equal(engineered_ts_between, expected_engineered_ts_between, check_like=False)

        # Also check that the original input was not inadvertently mutated
        # i.e. check that the modifications return whole new objects instead of mutating
        # the original input df
        pd.testing.assert_frame_equal(flat_timeseries_container, expected_unmodified_data)

    def test_diff_between_series_wide_dataframe(self):
        (
            stacked_dataframe_timeseries_container,
            (column_id, column_sort, column_kind, column_value),
        ) = self.create_simple_test_data_sample_stacked(randomise_sort_order = False)

        expected_unmodified_data = stacked_dataframe_timeseries_container.copy(deep=True)

        engineered_ts_between = diff_between_series(
            timeseries_container=stacked_dataframe_timeseries_container,
            column_sort=column_sort,
            column_id=column_id,
            column_kind=column_kind,
            column_value=column_value,
        )

        expected_D_y1y2 = [11, 3, 26, 15, -2, -46]
        expected_D_y1y3 = [-5, -2, 23, 15, 10, -35]
        expected_D_y2y3 = [-16, -5, -3, 0, 12, 11]
        expected_between_values = expected_D_y1y2 + expected_D_y1y3 + expected_D_y2y3
        expected_between_kinds = 6 * ["D_y1y2"] + 6 * ["D_y1y3"] + 6 * ["D_y2y3"]

        expected_engineered_ts_between = pd.concat(
            [
                stacked_dataframe_timeseries_container,
                pd.DataFrame(
                    {
                        column_id: stacked_dataframe_timeseries_container[
                            column_id
                        ].tolist(),
                        column_sort: stacked_dataframe_timeseries_container[
                            column_sort
                        ].tolist(),
                        column_kind: expected_between_kinds,
                        column_value: expected_between_values,
                    }
                ),
            ]
        ).reset_index(drop=True)

        print("actual")
        print(expected_engineered_ts_between)

        print("EXPECTED")
        print(expected_engineered_ts_between)

        pd.testing.assert_frame_equal(engineered_ts_between, expected_engineered_ts_between, check_like=False)

        # Also check that the original input was not inadvertently mutated
        # i.e. check that the modifications return whole new objects instead of mutating
        # the original input df
        pd.testing.assert_frame_equal(stacked_dataframe_timeseries_container, expected_unmodified_data)

    def test_differences_between_dictionary(self):
        (
            dict_timeseries_container,
            (column_id, column_sort, column_kind, column_value),
            (_, _, _),
            (id_values, sort_values),
        ) = self.create_simple_test_data_sample_dict(randomise_sort_order = False)

        expected_unmodified_data = copy.deepcopy(dict_timeseries_container)

        engineered_ts_between = diff_between_series(
            timeseries_container=dict_timeseries_container,
            column_sort=column_sort,
            column_id=column_id,
            column_value=column_value,
            column_kind=column_kind,
        )

        expected_D_y1y2 = [11, 3, 26, 15, -2, -46]
        expected_D_y1y3 = [-5, -2, 23, 15, 10, -35]
        expected_D_y2y3 = [-16, -5, -3, 0, 12, 11]
        expected_id_sort_keys = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        id_values, sort_values = list(zip(*expected_id_sort_keys))
        expected_new_ys = {
            "D_y1y2": expected_D_y1y2,
            "D_y1y3": expected_D_y1y3,
            "D_y2y3": expected_D_y2y3,
        }
        expected_engineered_ts_between = {**dict_timeseries_container, **{
            y_name: pd.DataFrame(
                {column_id: id_values, column_sort: sort_values, column_value: y_values}
            )
            for (y_name, y_values) in expected_new_ys.items()
        }}

        print("actual")
        print(expected_engineered_ts_between)
        print("EXPECTED")
        print(expected_engineered_ts_between)

        # Check expected output
        self.assertTrue(
            testable_dictionary_of_dataframes(engineered_ts_between)
            == testable_dictionary_of_dataframes(expected_engineered_ts_between)
        )

        # Also check that the original input was not inadvertently mutated
        # i.e. check that the modifications return whole new objects instead of mutating
        # the original input df
        self.assertTrue(
            testable_dictionary_of_dataframes(dict_timeseries_container)
            == testable_dictionary_of_dataframes(expected_unmodified_data)
        )

    def test_diff_within_series_no_sort_order_provided(self):
        """
        For louis
        """
        pass

    def test_diff_between_series_no_sort_order_provided(self):
        """
        For louis
        """
        pass


class testable_dictionary_of_dataframes(dict):
    """
    Class to test equality of dictionaries of dataframes
    """

    def __eq__(self, other_dictionary_of_dataframes):

        if any(isinstance(self[key], pd.DataFrame) is False for key in self.keys()):
            raise ValueError("Expects a dictionary of dataframes")

        if self.keys() != other_dictionary_of_dataframes.keys():
            return False

        for key in self.keys():
            if key not in other_dictionary_of_dataframes:
                return False
            elif pd.testing.assert_frame_equal(self[key], other_dictionary_of_dataframes[key], check_like=False) is False:
                return False
        return True
