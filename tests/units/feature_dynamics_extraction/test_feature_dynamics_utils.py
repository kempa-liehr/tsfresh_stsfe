from unittest import TestCase

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
                        {"f_agg": "var",
                        "isabs": False,
                        "qh": 0.2,
                        "ql": 0.0}
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
            4: {'y': {'change_quantiles': [{'f_agg': "var"}]}},
            10: {'y': {'change_quantiles': [{'isabs': False}, {'qh': 0.2}]}},
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
            expected_updated_dictionary_4,  window_length = 4, feature_parts = feature_parts_4
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
        feature_parts_5 = ['y', 'change_quantiles', 'dimension_5', 'tau_1']
        feature_parts_6 = ['y', 'change_quantiles', 'dimension_5', 'tau_1',"qh_0.2"] 
        for feature_parts_n in [feature_parts_5, feature_parts_6]:
            dictionary_to_repeatedly_mutate = update_feature_dictionary(dictionary_to_repeatedly_mutate, window_length_4, feature_parts_n)
        expected_dictionary_to_repeatedly_mutate = {1 : {'y':{'change_quantiles': [{'dimension': 5, 'tau': 1}, {'dimension': 5, 'tau': 1, 'qh': 0.2}]}}}
        self.assertTrue(dictionary_to_repeatedly_mutate == expected_dictionary_to_repeatedly_mutate)

        # Test giving it an illegal feature name...
        # Test giving it weird window length
        # Tets giving it a duplicate feature

    def test_parse_feature_timeseries_parts(self):
        assert True

    def test_parse_feature_dynamics_parts(self):
        assert True

    def test_derive_features_dictionaries(self):
        assert True

    def test_interpret_feature_dynamic(self):
        assert True

    def test_derive_features_dictionaries(self):
        assert True

    def test_dictionary_to_string(self):
        assert True

    def test_gen_pdf_for_feature_dynamics(self):
        assert True


class EngineerTimeSeriesTestCase(TestCase):
    """"""

    def test_engineer_input_timeseries(self):
        assert True

    def test_series_differencing(self):
        assert True

    def test_diff_between_series(self):
        assert True
