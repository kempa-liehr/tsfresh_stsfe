# -*- coding: utf-8 -*-
# This file as well as the whole tsfresh package are licenced under the MIT licence (see the LICENCE.txt)
# Maximilian Christ (maximilianchrist.com), Blue Yonder Gmbh, 2016

import warnings
from contextlib import contextmanager
from unittest import TestCase

import numpy as np
import pandas as pd


@contextmanager
def warning_free():
    """Small helper to surpress all warnings"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        yield


# todo: add test cases for float data
# todo: add test cases for nans, -infs, infs
# todo: add test cases with time series of length one


class DataTestCase(TestCase):
    def create_test_data_sample(self):
        cid = np.repeat([10, 500], 40)
        ckind = np.repeat(["a", "b", "a", "b"], 20)
        csort = [
            30,
            53,
            26,
            35,
            42,
            25,
            17,
            67,
            20,
            68,
            46,
            12,
            0,
            74,
            66,
            31,
            32,
            2,
            55,
            59,
            56,
            60,
            34,
            69,
            47,
            15,
            49,
            8,
            50,
            73,
            23,
            62,
            24,
            33,
            22,
            70,
            3,
            38,
            28,
            75,
            39,
            36,
            64,
            13,
            72,
            52,
            40,
            16,
            58,
            29,
            63,
            79,
            61,
            78,
            1,
            10,
            4,
            6,
            65,
            44,
            54,
            48,
            11,
            14,
            19,
            43,
            76,
            7,
            51,
            9,
            27,
            21,
            5,
            71,
            57,
            77,
            41,
            18,
            45,
            37,
        ]
        cval = [
            11,
            9,
            67,
            45,
            30,
            58,
            62,
            19,
            56,
            29,
            0,
            27,
            36,
            43,
            33,
            2,
            24,
            71,
            41,
            28,
            50,
            40,
            39,
            7,
            53,
            23,
            16,
            37,
            66,
            38,
            6,
            47,
            3,
            61,
            44,
            42,
            78,
            31,
            21,
            55,
            15,
            35,
            25,
            32,
            69,
            65,
            70,
            64,
            51,
            46,
            5,
            77,
            26,
            73,
            76,
            75,
            72,
            74,
            10,
            57,
            4,
            14,
            68,
            22,
            18,
            52,
            54,
            60,
            79,
            12,
            49,
            63,
            8,
            59,
            1,
            13,
            20,
            17,
            48,
            34,
        ]
        df = pd.DataFrame({"id": cid, "kind": ckind, "sort": csort, "val": cval})
        df = df.set_index("id", drop=False)
        df.index.name = None
        return df

    def create_test_data_sample_wide(self):
        rec = np.rec.array(
            [
                (0, 10, 0, 11, 50),
                (1, 10, 1, 9, 40),
                (2, 10, 2, 67, 39),
                (3, 10, 3, 45, 7),
                (4, 10, 4, 30, 53),
                (5, 10, 5, 58, 23),
                (6, 10, 6, 62, 16),
                (7, 10, 7, 19, 37),
                (8, 10, 8, 56, 66),
                (9, 10, 9, 29, 38),
                (10, 10, 10, 0, 6),
                (11, 10, 11, 27, 47),
                (12, 10, 12, 36, 3),
                (13, 10, 13, 43, 61),
                (14, 10, 14, 33, 44),
                (15, 10, 15, 2, 42),
                (16, 10, 16, 24, 78),
                (17, 10, 17, 71, 31),
                (18, 10, 18, 41, 21),
                (19, 10, 19, 28, 55),
                (20, 500, 0, 15, 4),
                (21, 500, 1, 35, 14),
                (22, 500, 2, 25, 68),
                (23, 500, 3, 32, 22),
                (24, 500, 4, 69, 18),
                (25, 500, 5, 65, 52),
                (26, 500, 6, 70, 54),
                (27, 500, 7, 64, 60),
                (28, 500, 8, 51, 79),
                (29, 500, 9, 46, 12),
                (30, 500, 10, 5, 49),
                (31, 500, 11, 77, 63),
                (32, 500, 12, 26, 8),
                (33, 500, 13, 73, 59),
                (34, 500, 14, 76, 1),
                (35, 500, 15, 75, 13),
                (36, 500, 16, 72, 20),
                (37, 500, 17, 74, 17),
                (38, 500, 18, 10, 48),
                (39, 500, 19, 57, 34),
            ],
            dtype=[
                ("index", "<i8"),
                ("id", "<i8"),
                ("sort", "<i8"),
                ("a", "<i8"),
                ("b", "<i8"),
            ],
        )
        df = pd.DataFrame.from_records(rec)
        df = df.set_index("index", drop=True)
        return df

    def create_test_data_sample_with_time_index(self):
        cid = np.repeat([10, 500], 40)
        ckind = np.repeat(["a", "b", "a", "b"], 20)
        csort = [
            30,
            53,
            26,
            35,
            42,
            25,
            17,
            67,
            20,
            68,
            46,
            12,
            0,
            74,
            66,
            31,
            32,
            2,
            55,
            59,
            56,
            60,
            34,
            69,
            47,
            15,
            49,
            8,
            50,
            73,
            23,
            62,
            24,
            33,
            22,
            70,
            3,
            38,
            28,
            75,
            39,
            36,
            64,
            13,
            72,
            52,
            40,
            16,
            58,
            29,
            63,
            79,
            61,
            78,
            1,
            10,
            4,
            6,
            65,
            44,
            54,
            48,
            11,
            14,
            19,
            43,
            76,
            7,
            51,
            9,
            27,
            21,
            5,
            71,
            57,
            77,
            41,
            18,
            45,
            37,
        ]
        cval = [
            11,
            9,
            67,
            45,
            30,
            58,
            62,
            19,
            56,
            29,
            0,
            27,
            36,
            43,
            33,
            2,
            24,
            71,
            41,
            28,
            50,
            40,
            39,
            7,
            53,
            23,
            16,
            37,
            66,
            38,
            6,
            47,
            3,
            61,
            44,
            42,
            78,
            31,
            21,
            55,
            15,
            35,
            25,
            32,
            69,
            65,
            70,
            64,
            51,
            46,
            5,
            77,
            26,
            73,
            76,
            75,
            72,
            74,
            10,
            57,
            4,
            14,
            68,
            22,
            18,
            52,
            54,
            60,
            79,
            12,
            49,
            63,
            8,
            59,
            1,
            13,
            20,
            17,
            48,
            34,
        ]
        time = pd.date_range(start="2018-01-01", freq="1D", periods=80)
        df = pd.DataFrame(
            {"time": time, "id": cid, "kind": ckind, "sort": csort, "val": cval}
        )
        df = df.set_index("time", drop=False)
        df.index.name = None
        return df

    def create_test_data_nearly_numerical_indices(self):
        cid = "99999_9999_" + pd.Series(np.repeat([10, 500], 40)).astype(str)
        ckind = np.repeat(["a", "b", "a", "b"], 20)
        csort = [
            30,
            53,
            26,
            35,
            42,
            25,
            17,
            67,
            20,
            68,
            46,
            12,
            0,
            74,
            66,
            31,
            32,
            2,
            55,
            59,
            56,
            60,
            34,
            69,
            47,
            15,
            49,
            8,
            50,
            73,
            23,
            62,
            24,
            33,
            22,
            70,
            3,
            38,
            28,
            75,
            39,
            36,
            64,
            13,
            72,
            52,
            40,
            16,
            58,
            29,
            63,
            79,
            61,
            78,
            1,
            10,
            4,
            6,
            65,
            44,
            54,
            48,
            11,
            14,
            19,
            43,
            76,
            7,
            51,
            9,
            27,
            21,
            5,
            71,
            57,
            77,
            41,
            18,
            45,
            37,
        ]
        cval = [
            11,
            9,
            67,
            45,
            30,
            58,
            62,
            19,
            56,
            29,
            0,
            27,
            36,
            43,
            33,
            2,
            24,
            71,
            41,
            28,
            50,
            40,
            39,
            7,
            53,
            23,
            16,
            37,
            66,
            38,
            6,
            47,
            3,
            61,
            44,
            42,
            78,
            31,
            21,
            55,
            15,
            35,
            25,
            32,
            69,
            65,
            70,
            64,
            51,
            46,
            5,
            77,
            26,
            73,
            76,
            75,
            72,
            74,
            10,
            57,
            4,
            14,
            68,
            22,
            18,
            52,
            54,
            60,
            79,
            12,
            49,
            63,
            8,
            59,
            1,
            13,
            20,
            17,
            48,
            34,
        ]
        df = pd.DataFrame({"id": cid, "kind": ckind, "sort": csort, "val": cval})
        df = df.set_index("id", drop=False)
        df.index.name = None
        return df

    def create_one_valued_time_series(self):
        cid = [1, 2, 2]
        ckind = ["a", "a", "a"]
        csort = [1, 1, 2]
        cval = [1.0, 5.0, 6.0]
        df = pd.DataFrame({"id": cid, "kind": ckind, "sort": csort, "val": cval})
        return df

    def create_test_data_sample_with_target(self):
        """
        Small test data set with target.
        :return: timeseries df
        :return: target y which is the mean of each sample's timeseries
        """
        cid = np.repeat(range(50), 3)
        csort = list(range(3)) * 50
        cval = [1, 2, 3] * 30 + [4, 5, 6] * 20
        df = pd.DataFrame({"id": cid, "kind": "a", "sort": csort, "val": cval})
        y = pd.Series([2] * 30 + [5] * 20)
        return df, y

    def create_test_data_sample_with_multiclass_target(self):
        """
        Small test data set with target.
        :return: timeseries df
        :return: target y which is the mean of each sample's timeseries
        """
        cid = np.repeat(range(75), 3)
        csort = list(range(3)) * 75
        cval = [1, 2, 3] * 30 + [4, 5, 6] * 20 + [7, 8, 9] * 25
        df = pd.DataFrame({"id": cid, "kind": "a", "sort": csort, "val": cval})
        y = pd.Series([2] * 30 + [5] * 20 + [8] * 25)
        return df, y


    def create_simple_test_data_sample_wide(self):
        """
        Small test data set in wide format
        :return: timeseries df in wide format 
        :return: column params corresponding to the format
        :return: values of the column kinds

        TODO: Add a flag to make column sort none
        """
        column_sort = "sort"
        column_id = "id"
        column_value = None
        column_kind = None

        # Set up input timeseries
        id = [1, 1, 1, 2, 2, 2]
        sort = [1, 2, 3, 1, 2, 3]
        y1 = [1, 3, 27, 18, 12, -34]
        y2 = [-10, 0, 1, 3, 14, 12]
        y3 = [6, 5, 4, 3, 2, 1]
        flat_timeseries_container = pd.DataFrame(
            {column_id: id, column_sort: sort, "y1": y1, "y2": y2, "y3": y3}
        )
        column_params = (column_id,column_sort,column_kind,column_value)
        kinds = (y1,y2,y3)
  
        return flat_timeseries_container, column_params, kinds


    def create_simple_test_data_sample_stacked(self):
        """
        Small test data set in long format
        :return: timeseries df in long format (stacked)
        :return: column params corresponding to the format

        TODO: Add a flag to make column sort none
        """
        column_sort = "sort"
        column_id = "id"
        column_kind = "kind"
        column_value = "val"

        # Set up input time series
        id = 3 * [1, 1, 1, 2, 2, 2]
        sort = 3 * [1, 2, 3, 1, 2, 3]
        val = [1, 3, 27, 18, 12, -34] + [-10, 0, 1, 3, 14, 12] + [6, 5, 4, 3, 2, 1]
        kind = 6 * ["y1"] + 6 * ["y2"] + 6 * ["y3"]
        stacked_dataframe_timeseries_container = pd.DataFrame(
            {column_id: id, column_sort: sort, column_kind: kind, column_value: val}
        )
        column_params = (column_id, column_sort, column_kind, column_value)

        return stacked_dataframe_timeseries_container, column_params
        

    def create_simple_test_data_sample_dict(self):
        """
        Small test data set of dictionaries of dfs in wide format
        :return: timeseries df in dict[pd.dataframe] format (wide)
        :return: column params corresponding to the format
        :return: values of the column kinds
        :return: id and sort values which is assumed to be the same for each flat df

        TODO: Add a flag to make column sort none
        """
        column_sort = "sort"
        column_id = "id"
        column_value = "value"
        column_kind = None

        # Set up input time series
        id = [1, 1, 1, 2, 2, 2]
        sort = [1, 2, 3, 1, 2, 3]
        y1 = [1, 3, 27, 18, 12, -34]
        y2 = [-10, 0, 1, 3, 14, 12]
        y3 = [6, 5, 4, 3, 2, 1]
        ys = {"y1": y1, "y2": y2, "y3": y3}
        dictionary_timeseries_container = {
            y_name: pd.DataFrame(
                {column_id: id, column_sort: sort, column_value: y_values}
            )
            for (y_name, y_values) in ys.items()
        }
        column_params = (column_id, column_sort, column_kind, column_value)
        kinds = (y1,y2,y3)
        homogenous_values = (id, sort)

        return dictionary_timeseries_container, column_params, kinds, homogenous_values

    def create_test_data_expected_tuples(self):
        test_data_expected_tuples = [
        (
            10,
            "a",
            pd.Series(
                [
                    36,
                    71,
                    27,
                    62,
                    56,
                    58,
                    67,
                    11,
                    2,
                    24,
                    45,
                    30,
                    0,
                    9,
                    41,
                    28,
                    33,
                    19,
                    29,
                    43,
                ],
                index=list(range(20)),
            ),
        ),
        (
            10,
            "b",
            pd.Series(
                [
                    78,
                    37,
                    23,
                    44,
                    6,
                    3,
                    21,
                    61,
                    39,
                    31,
                    53,
                    16,
                    66,
                    50,
                    40,
                    47,
                    7,
                    42,
                    38,
                    55,
                ],
                index=list(range(20)),
            ),
        ),
        (
            500,
            "a",
            pd.Series(
                [
                    76,
                    72,
                    74,
                    75,
                    32,
                    64,
                    46,
                    35,
                    15,
                    70,
                    57,
                    65,
                    51,
                    26,
                    5,
                    25,
                    10,
                    69,
                    73,
                    77,
                ],
                index=list(range(20, 40)),
            ),
        ),
        (
            500,
            "b",
            pd.Series(
                [
                    8,
                    60,
                    12,
                    68,
                    22,
                    17,
                    18,
                    63,
                    49,
                    34,
                    20,
                    52,
                    48,
                    14,
                    79,
                    4,
                    1,
                    59,
                    54,
                    13,
                ],
                index=list(range(20, 40)),
            ),
        ),
    ]
        return test_data_expected_tuples
    

    def create_test_data_expected_tuples_wide(self):
        wide_test_data_expected_tuples = [
            (
            10,
            "a",
            pd.Series(
                [
                    11,
                    9,
                    67,
                    45,
                    30,
                    58,
                    62,
                    19,
                    56,
                    29,
                    0,
                    27,
                    36,
                    43,
                    33,
                    2,
                    24,
                    71,
                    41,
                    28,
                ],
                index=list(range(20)),
            ),
        ),
        (
            10,
            "b",
            pd.Series(
                [
                    50,
                    40,
                    39,
                    7,
                    53,
                    23,
                    16,
                    37,
                    66,
                    38,
                    6,
                    47,
                    3,
                    61,
                    44,
                    42,
                    78,
                    31,
                    21,
                    55,
                ],
                index=list(range(20)),
            ),
        ),
        (
            500,
            "a",
            pd.Series(
                [
                    15,
                    35,
                    25,
                    32,
                    69,
                    65,
                    70,
                    64,
                    51,
                    46,
                    5,
                    77,
                    26,
                    73,
                    76,
                    75,
                    72,
                    74,
                    10,
                    57,
                ],
                index=list(range(20, 40)),
            ),
        ),
        (
            500,
            "b",
            pd.Series(
                [
                    4,
                    14,
                    68,
                    22,
                    18,
                    52,
                    54,
                    60,
                    79,
                    12,
                    49,
                    63,
                    8,
                    59,
                    1,
                    13,
                    20,
                    17,
                    48,
                    34,
                ],
                index=list(range(20, 40)),
            ),
        ),
    ]   
        return wide_test_data_expected_tuples

    def create_split_up_test_data_expected_tuples(self):
        window_length = 6
        # Window size of 6 should give the following chunks
        test_data_expected_chunked_up_tuples = [
        # 4 chunks with id: 10, kind: 'a'
        ((10,0),
        'a',
        pd.Series(
            [
               36,
               71,
               27,
               62,
               56,
               58
            ],
            index=[10] * 6
        )),
        ((10,1),
        'a',
        pd.Series(
            [
               67,
               11,
               2,
               24,
               45,
               30 
            ],
            index=[10] * 6
        )),
        ((10,2),
        'a',
        pd.Series(
            [
               0,
               9,
               4,
               28,
               33,
               19
            ],
            index=[10] * 6
        )),
        ((10,3),
        'a',
        pd.Series(
            [
               29,
               43 
            ],
            index=[10] * 2
        )),
        # 4 chunks with id: 10, kind: 'b'
        ((10,0),
        'b',
        pd.Series(
            [
               78,
               37,
               23,
               44,
               6,
               3
            ],
            index=[10] * 6
        )),
        ((10,0),
        'b',
        pd.Series(
            [
               21,61,39,31,52,16
            ],
            index=[10] * 6
        )),
        ((10,0),
        'b',
        pd.Series(
            [
              66,50,40,47,7,42
            ],
            index=[10] * 6
        )),
        ((10,0),
        'b',
        pd.Series(
            [
               38,55
            ],
            index=[10] * 2
        )),
        # 4 chunks with id: 500, kind: 'a'
        ((500,0),
        'a',
        pd.Series(
            [
               76,
               72,
               74,
               75,
               32,
               64
            ],
            index=[500] * 6
        )),
        ((500,1),
        'a',
        pd.Series(
            [
               46,
               35,
               15,
               70,
               57,
               65
            ],
            index=[500] * 6
        )),
        ((500,2),
        'a',
        pd.Series(
            [
               51,
               26,
               5,
               25,
               10,
               69
            ],
            index=[500] * 6
        )),
        ((500,3),
        'a',
        pd.Series(
            [
               73,
               77 
            ],
            index=[500] * 2
        )),
        # 4 chunks with id: 500, kind: 'b'
        ((500,0),
        'b',
        pd.Series(
            [
               8,
               60,
               12,
               68,
               22,
               17     
            ],
            index=[500] * 6
        )),
        ((500,1),
        'b',
        pd.Series(
            [
               18,
               63,
               49,
               34,
               20,
               52 
            ],
            index=[500] * 6
        )),
        ((500,2),
        'b',
        pd.Series(
            [
               48,
               14,
               79,
               4,
               1,
               59 
            ],
            index=[500] * 6
        )),
        ((500,3),
        'b',
        pd.Series(
            [
               54,
               13 
            ],
            index=[500] * 2
        )),
        ]
        
        return (test_data_expected_chunked_up_tuples, window_length)

    def create_split_up_test_data_expected_tuples_wide(self):
        window_length = 6
        # Window size of 6 should give the following chunks
        wide_test_data_expected_chunked_up_tuples = [
        # 4 chunks with id: 10, kind: 'a'
        ((10,0),
        'a',
        pd.Series(
            [
               36,
               71,
               27,
               62,
               56,
               58
            ],
            index=[10] * 6
        )),
        ((10,1),
        'a',
        pd.Series(
            [
               67,
               11,
               2,
               24,
               45,
               30 
            ],
            index=[10] * 6
        )),
        ((10,2),
        'a',
        pd.Series(
            [
               0,
               9,
               41,
               28,
               33,
               19
            ],
            index=[10] * 6
        )),
        ((10,3),
        'a',
        pd.Series(
            [
               29,
               43 
            ],
            index=[10] * 2
        )),
        # 4 chunks with id: 10, kind: 'b'
        ((10,0),
        'b',
        pd.Series(
            [
               78,
               37,
               23,
               44,
               6,
               3
            ],
            index=[10] * 6
        )),
        ((10,0),
        'b',
        pd.Series(
            [
               21,61,39,31,52,16
            ],
            index=[10] * 6
        )),
        ((10,0),
        'b',
        pd.Series(
            [
              66,50,40,47,7,42
            ],
            index=[10] * 6
        )),
        ((10,0),
        'b',
        pd.Series(
            [
               38,55
            ],
            index=[10] * 2
        )),
        # 4 chunks with id: 500, kind: 'a'
        ((500,0),
        'a',
        pd.Series(
            [
               76,
               72,
               74,
               75,
               32,
               64
            ],
            index=[500] * 6
        )),
        ((500,1),
        'a',
        pd.Series(
            [
               46,
               35,
               15,
               70,
               57,
               65
            ],
            index=[500] * 6
        )),
        ((500,2),
        'a',
        pd.Series(
            [
               51,
               26,
               5,
               25,
               10,
               69
            ],
            index=[500] * 6
        )),
        ((500,3),
        'a',
        pd.Series(
            [
               73,
               77 
            ],
            index=[500] * 2
        )),
        # 4 chunks with id: 500, kind: 'b'
        ((500,0),
        'b',
        pd.Series(
            [
               8,
               60,
               12,
               68,
               22,
               17     
            ],
            index=[500] * 6
        )),
        ((500,1),
        'b',
        pd.Series(
            [
               18,
               63,
               49,
               34,
               20,
               52 
            ],
            index=[500] * 6
        )),
        ((500,2),
        'b',
        pd.Series(
            [
               48,
               14,
               79,
               4,
               1,
               59 
            ],
            index=[500] * 6
        )),
        ((500,3),
        'b',
        pd.Series(
            [
               54,
               13 
            ],
            index=[500] * 2
        )),
        ]
        return (wide_test_data_expected_chunked_up_tuples, window_length)
