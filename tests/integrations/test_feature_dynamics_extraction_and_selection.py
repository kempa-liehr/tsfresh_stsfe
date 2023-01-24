import dask.dataframe as dd
import pandas as pd
import numpy as np
from unittest import TestCase

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
    diff_between_series
)



# WARNING/NOTE:: This file is still in progress! Highly likely that some really big changes will be made to this whole file 
# soon....


class FeatureDynamicsExtractionTestCase(TestCase):

    def test_pandas(self):
        pass

    def test_pandas_no_pivot(self):
        pass

    def test_dask(self):
        pass

    def test_dask_no_pivot(self):
        pass

    def test_extract_more_features_based_on_feature_names(self):
        """
        Suppose there are many features deemed relevant, 
        with different window lengths in a feature store that 
        have been accumulated over time.

        New timeseries data has come in, and we want to extract
        features based on the feature names already in the feature
        store.
        """

        # get_feature_names = get_feature_names_with_varying_window_lengths()

        # extract_feature_dynamics()


class EndToEndTestDataCase(TestCase):


    def column_params_picker(self, data_format):
        """
        Picks out the correct column params depending on 
        the particular data format type
        """

        if data_format not in ["wide", "long", "dict"]:
            raise ValueError

        print("For now we havent implemented long and dict")
        #TODO: Remove once implemented more stuff 
        data_format = "wide"

        if data_format == "wide":
            return {
                "column_sort":None,
                "column_kind":None,
                "column_id": "measurement_id",
                "column_value":None
            }

        elif data_format == "long":
            return {
                "column_sort":"t",
                "column_kind":"kind",
                "column_id": "measurement_id",
                "column_value":"value"
            }

        elif data_format == "dict":
            return {
                "column_sort":"sort",
                "column_kind":None,
                "column_id": "measurement_id",
                "column_value":"value"
            }

    def gen_feature_calculators_for_e2e_tests(self, feature_complexity = "minimal"):

        if feature_complexity not in ["minimal", "not-minimal"]:
            raise ValueError('feature_complexity needs to be one of ["minimal", "not-minimal"]')

        if feature_complexity == "minimal":
            return MinimalFCParameters()
        elif feature_complexity == "not-minimal":
            # Get a reasonably sized subset of somewhat comeplex features
            non_simple_features_fc_parameters = {
                "agg_autocorrelation": [{'f_agg':"mean",'maxlag':40}],
                "augmented_dickey_fuller": [{'attr':"pvalue",'autolag':"BIC"}],
                "mean_second_derivative_central": None,
                "last_location_of_maximum": None,
                "fft_coefficient": [{"coeff":1, "attr":"real"}],
                "fft_aggregated": [{"aggtype":"variance"}],
                "number_peaks": [{"n":3}],
                "number_cwt_peaks": [{"n":3}],
                "linear_trend": [{"attr":"slope"}],
                "permutation_entropy":[{"tau":1, "dimension":2}], 
                "quantile": [{'q':0.2}],
                "benford_correlation": None
            }
            
            return non_simple_features_fc_parameters

    
    def gen_example_timeseries_data_for_e2e_tests(self, container_type, data_format):

        """
        TODO: This data should be refactored, but keeping it around for now
        TODO: This should absolutely be replaced or deleted very soon before being merged into anything...
        """

        if container_type not in ["pandas", "dask"]:
            raise ValueError
        if data_format not in ["wide", "long","dict"]:
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

        ts = pd.DataFrame(
            {
                "t": np.repeat([1, 2, 3, 4, 5, 6], 10),
                "y1": np.asarray(y1, dtype=float),
                "y2": np.asarray(y2, dtype=float),
                "y3": np.asarray(y3, dtype=float),
                "measurement_id": np.asarray(measurement_id, dtype=int),
            }
        )

        if data_format != "wide":
            print("Haven't yet got around to other formats")

        if container_type == "dask":
            ts = dd.from_pandas(ts, npartitions=3)

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



class SelectFeaturesAndExtractTestCase(TestCase):
    """
    Tests for extracting features, 
    selecting relevant features, then
    saving the feature names to a dictionary,
    and then extracting more relevant feature 
    dynamics.
    """

    def test_select_features_from_minimal_extraction_from_engineered_timeseries(self):

        pass

    def test_select_features_from_efficient_extraction_from_engineered_timeseries(self):
        pass


    # Pandas minimal (a,b,c 3 input formats)
    # def pandas minimal input format a
    # def pandas minimal input format b
    # def pandas minimal input format c
    # extract ---> select ---> save relevant --- extract on relevant

    # Pandas efficient (a,b,c 3 input formats) 
    # def pandas efficient input format a
    # def pandas efficient input format b
    # def pandas efficient input format c
    # extract ---> select ---> save relevant --- extract on relevant

    # Dask minimal (3 formats but as dask dds)
    # def Dask minimal input format a
    # def Dask minimal input format b
    # def Dask minimal input format c
    # extract ---> select ---> save relevant --- extract on relevant


    # Dask efficient (3 formats but as dask dds)
    # def Dask efficient input format a
    # def Dask efficient input format b
    # def Dask efficient input format c
    # extract ---> select ---> save relevant --- extract on relevant



class InterpretFeatureDynamicsTestCase(TestCase):
    """
    Tests for taking a very long feature dynamics names and 
    generating a pdf which interprets what these feature names 
    actually mean
    """

    def test_generate_pdf(self):
        pass

class EngineerMoreTsTestCase(TestCase):
    """
    Tests for engineering more timeseries, 
    and then extracting feature dynamics based 
    on these.
    """
    pass

    # Engineer on pandas (3 input formats, large and small dicts)
    # Engineer on pandas then convert to dask check it still works (3 formats, large and small dicts)
    
    def test_engineer_more_ts_and_then_extraction_on_pandas_minimal(self):
        pass

    def test_engineer_more_ts_and_then_extraction_on_pandas_efficient(self):
        pass

    def test_dynamics_extraction_on_pandas_minimal(self):
        pass

    def test_dynamics_extraction_on_pandas_efficient(self):
        pass

    def test_dynamics_extraction_on_dask_minimal(self):
        pass

    def test_making_many_combos_of_differences(self):
        pass
        # ts_with_extra_timeseries = diff_within_series()
        # ts_with_extra_timeseries = diff_between_series()

        # Apply the differences function repeatedly and test it works alright


class FullEndToEndFeatureDynamicsWorkflowTestCase(EndToEndTestDataCase):
    """
    Test for:

    a) Engineer more timeseries then
    b) Extract features then
    c) Select featues then
    d) Interpret the top relevant feature dynamics via PDF file then
    e) Extract relevant features on more timeseries data
    """

    def test_end_to_end_pandas(self):
        
        # Test the end to end process of engineer,extract, select, interpret, extract on selected 
        # for pandas for each of the pandas input formats

        for data_format in ["wide", "long", "dict"]:

            ts, response = self.gen_example_timeseries_data_for_e2e_tests(container_type = "pandas", data_format = data_format)

            # TODO: Have a small amount of complex fd calculators i.e. take 8 features from efficientfcparams

            fts_fcs = self.gen_feature_calculators_for_e2e_tests(feature_complexity = "not-minimal")
            fd_fcs =  self.gen_feature_calculators_for_e2e_tests(feature_complexity = "not-minimal")

            window_length_1 = 4
            window_length_2 = 5
            fts_fcs_with_window_lengths = {window_length_1:fts_fcs, window_length_2:fts_fcs}
            fts_fds_with_window_lengths = {window_length_1:fd_fcs, window_length_2:fd_fcs}

            column_params_config = self.column_params_picker(data_format=data_format)

            # a) Engineer some more timeseries from input timeseries 
            ts_with_extra_timeseries_within = diff_within_series(ts, column_id = column_params_config["column_id"], column_sort = column_params_config["column_sort"], column_value = column_params_config["column_value"], column_kind = column_params_config["column_kind"])
            # add an even extra layer of ts differencing 
            ts_with_extra_timeseries_between_and_within = diff_between_series(ts_with_extra_timeseries_within, column_id = column_params_config["column_id"], column_sort = column_params_config["column_sort"], column_value = column_params_config["column_value"], column_kind = column_params_config["column_kind"])

            # TODO: Assert stuff here
            
            # b) Extract
            X = extract_feature_dynamics(
                timeseries_container = ts_with_extra_timeseries_between_and_within,
                feature_timeseries_fc_parameters = fts_fcs_with_window_lengths,
                feature_dynamics_fc_parameters = fts_fds_with_window_lengths,
                column_id = column_params_config["column_id"],
                column_sort = column_params_config["column_sort"],
                column_kind = column_params_config["column_kind"],
                column_value = column_params_config["column_value"] 
            )

            # TODO: Assert stuff here

            # c) Select
            X_relevant = select_features(X, response, fdr_level = 0.95)

            # TODO: Assert stuff here

            # d) Interpret
            rel_feature_names = list(X_relevant.columns)

            rel_feature_time_series_dict, rel_feature_dynamics_dict = derive_features_dictionaries(
                rel_feature_names
            )

            gen_pdf_for_feature_dynamics(
                    feature_dynamics_names=rel_feature_names,
                )

            # TODO: Assert stuff here

            # e) extract on selected features
             
            # TODO: Could extract from a new bunch of timeseries to make it clearer what the benefit of this is
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

            # TODO: Assert stuff here

            self.assertTrue(True)





    def end_to_end_dask_long_format(self):
        # NOTE: I think long format is the only format that works with Dask (could be wrong here)
        ts, response = self.gen_example_timeseries_data_for_e2e_tests(container_type="dask", data_format = "wide")
        fts_fcs = self.gen_feature_calculators_for_e2e_tests(feature_complexity = "efficient")
        fd_fcs =  self.gen_feature_calculators_for_e2e_tests(feature_complexity = "efficient")

        window_length_1 = 4
        window_length_2 = 5
        fts_fcs_with_window_lengths = {window_length_1:fts_fcs, window_length_2:fts_fcs}
        fts_fds_with_window_lengths = {window_length_1:fd_fcs, window_length_2:fd_fcs}

        # a) Extract
        X = extract_feature_dynamics(
            timeseries_container = ts,
            feature_timeseries_fc_parameters = fts_fcs_with_window_lengths,
            feature_dynamics_fc_parameters = fts_fds_with_window_lengths,
            column_id = "measurement_id",
            column_sort = "t",
            column_kind = None,
            column_value = None 
        )

        # b) Select
        X_pandas = X.compute()

        X_relevant = select_features(X_pandas, response, fdr_level = 0.95)

        # c) Interpret
        rel_feature_names = list(X_relevant.columns)

        rel_feature_time_series_dict, rel_feature_dynamics_dict = derive_features_dictionaries(
            rel_feature_names
        )

        gen_pdf_for_feature_dynamics(
                feature_dynamics_names=rel_feature_names,
            )

        # d) extract on selected features 
        # TODO: Could extract from a new bunch of timeseries to make it clearer what the benefit of this is
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
    