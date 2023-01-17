from tsfresh.feature_extraction.data import to_tsdata, LongTsFrameAdapter,WideTsFrameAdapter,TsDictAdapter
from tsfresh.feature_dynamics_extraction.feature_dynamics_data import IterableSplitTsData
import pandas as pd
from tests.units.feature_extraction.test_data import DataAdapterTestCase

class IterableSplitTsDataTestCase(DataAdapterTestCase):
    """"""

    def test_iter_on_long_data(self):
        df_stacked = self.create_test_data_sample()
        data_stacked = LongTsFrameAdapter(df_stacked, "id", "kind", "val", "sort")
        expected_windowed_tuples, window_length = self.create_split_up_test_data_expected_tuples()
        split_ts_data = IterableSplitTsData(data_stacked, split_size = window_length)

        # Test equality of object's main members
        self.assertTrue(split_ts_data._split_size == window_length and split_ts_data.df_id_type == object) 
        underlying_data_converted_to_tsdata = to_tsdata(split_ts_data._root_ts_data)
        expected_non_windowed_tuples = self.create_test_data_expected_tuples()
        self.assert_tsdata(underlying_data_converted_to_tsdata, expected_non_windowed_tuples)

        # Test equality of each chunk...
        self.assert_tsdata(split_ts_data,expected_windowed_tuples)

    def test_iter_on_long_data_no_value_column(self):
        df_stacked = self.create_test_data_sample()
        data_stacked_no_val = LongTsFrameAdapter(df_stacked, "id", "kind", None, "sort")
        expected_windowed_tuples, window_length = self.create_split_up_test_data_expected_tuples()
        split_ts_data = IterableSplitTsData(data_stacked_no_val, split_size = window_length)

        # Test equality of object's main members
        self.assertTrue(split_ts_data._split_size == window_length and split_ts_data.df_id_type == object) 
        underlying_data_converted_to_tsdata = to_tsdata(split_ts_data._root_ts_data)
        expected_non_windowed_tuples = self.create_test_data_expected_tuples()
        self.assert_tsdata(underlying_data_converted_to_tsdata, expected_non_windowed_tuples)

        # Test equality of each chunk...
        self.assert_tsdata(split_ts_data,expected_windowed_tuples)

    def test_iter_on_wide_data(self):
        df_wide = self.create_test_data_sample_wide()
        data_wide = WideTsFrameAdapter(df_wide, "id", "sort")
        expected_windowed_tuples, window_length = self.create_split_up_test_data_expected_tuples_wide()
        split_ts_data = IterableSplitTsData(data_wide, split_size = window_length)

        # Test equality of object's main members
        self.assertTrue(split_ts_data._split_size == window_length and split_ts_data.df_id_type == object) 
        underlying_data_converted_to_tsdata = to_tsdata(split_ts_data._root_ts_data)
        expected_non_windowed_tuples = self.create_test_data_expected_tuples_wide()
        self.assert_tsdata(underlying_data_converted_to_tsdata, expected_non_windowed_tuples)

        # Test equality of each chunk...
        self.assert_tsdata(split_ts_data,expected_windowed_tuples)

    def test_iter_on_wide_data_no_sort_column(self):
        df_wide = self.create_test_data_sample_wide()
        del df_wide["sort"]
        data_wide_no_sort = WideTsFrameAdapter(df_wide, "id", None)

        expected_windowed_tuples, window_length = self.create_split_up_test_data_expected_tuples_wide()
        split_ts_data = IterableSplitTsData(data_wide_no_sort, split_size = window_length)

        # Test equality of object's main members
        self.assertTrue(split_ts_data._split_size == window_length and split_ts_data.df_id_type == object) 
        underlying_data_converted_to_tsdata = to_tsdata(split_ts_data._root_ts_data)
        expected_non_windowed_tuples = self.create_test_data_expected_tuples_wide()
        self.assert_tsdata(underlying_data_converted_to_tsdata, expected_non_windowed_tuples)

        # Test equality of each chunk...
        self.assert_tsdata(split_ts_data,expected_windowed_tuples)

    def test_iter_on_dict(self):
        df_dict = {key: df for key, df in self.create_test_data_sample().groupby(["kind"])}
        data_dict = TsDictAdapter(df_dict, "id", "val", "sort")

        expected_windowed_tuples, window_length = self.create_split_up_test_data_expected_tuples()
        split_ts_data = IterableSplitTsData(data_dict, split_size = window_length)

        # Test equality of object's main members i.e each timeseries
        self.assertTrue(split_ts_data._split_size == window_length and split_ts_data.df_id_type == object) 
        underlying_data_converted_to_tsdata = to_tsdata(split_ts_data._root_ts_data)
        expected_non_windowed_tuples = self.create_test_data_expected_tuples()
        self.assert_tsdata(underlying_data_converted_to_tsdata, expected_non_windowed_tuples)

        # Test equality of each chunk... i.e. each subwindow
        self.assert_tsdata(split_ts_data, expected_windowed_tuples)


    def test_too_large_split_size(self):

        window_length = 100000 # this length is too large given the size of the input data

        test_data = to_tsdata(self.create_test_data_sample(), 'id','kind','val','sort')

        self.assertRaises(ValueError, IterableSplitTsData, test_data, window_length)

    def test_negative_split_size(self):

        window_length = -20

        test_data = to_tsdata(self.create_test_data_sample(), 'id','kind','val','sort')

        self.assertRaises(ValueError, IterableSplitTsData, test_data, window_length)

    def test_zero_split_size(self):

        window_length = 0 # this length is too large given the size of the input data

        test_data = to_tsdata(self.create_test_data_sample(), 'id','kind','val','sort')

        self.assertRaises(ValueError, IterableSplitTsData, test_data, window_length)

    def test_fractional_split_size(self):

        window_length = 1.50 # does not make physical sense to have non integer window lengths

        test_data = to_tsdata(self.create_test_data_sample(), 'id','kind','val','sort')

        self.assertRaises(ValueError, IterableSplitTsData, test_data, window_length)

    def test_pivot(self):
        # Test pivot
        self.assertTrue(True)

    ### Insert other tests related to functionality, edge cases etc.


class ApplyableSplitTsDataTestCase(DataAdapterTestCase):
    """ """

    def test_init(self):
        assert True

    def test_apply(self):
        assert True

    def test_wrapped_f(self):
        assert True

    def test_pivot(self):
        assert True
