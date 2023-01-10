"""
Contains two subclasses required for the extract_feature_dynamics function to work.

These subclasses could/should eventually belong in tsfresh.feature_extraction.data.py 
instead of this module.
"""

from tsfresh.feature_extraction.data import (
    PartitionedTsData,
    ApplyableTsData,
    Timeseries,
)
import math


class IterableSplitTsData(PartitionedTsData):
    """
    Wrapper around another iterable ts data object, which splits the root ts data
    object into smaller pieces, each of size split_size.

    This means if you iterate over this object, the root ts object
    will also be iterated, but the time series will additionally be split into smaller
    chunks.
    """

    def __init__(self, root_ts_data, split_size):
        """Initialize with the root ts data object and the size to split"""
        self._root_ts_data = root_ts_data
        self._split_size = split_size

        # The resulting type will be a tuple (id, chunk number),
        # so it is an object
        self.df_id_type = object

    def __iter__(self):
        """Iterate over the root ts data object and only return small chunks of the data"""
        tsdata = iter(self._root_ts_data)

        for chunk_id, chunk_kind, chunk_data in tsdata:
            max_chunks = math.ceil(len(chunk_data) / self._split_size)
            for chunk_number in range(max_chunks):
                yield Timeseries(
                    (chunk_id, chunk_number),
                    chunk_kind,
                    chunk_data.iloc[
                        chunk_number
                        * self._split_size : (chunk_number + 1)
                        * self._split_size
                    ],
                )

    def __len__(self):
        """The len needs to be re-calculated"""
        summed_length = 0

        tsdata = iter(self._root_ts_data)

        for _, _, chunk_data in tsdata:
            summed_length += math.ceil(len(chunk_data) / self._split_size)

        return summed_length

    def pivot(self, results):
        """Pivoting can be copied from the root ts object"""
        return self._root_ts_data.pivot(results)


class ApplyableSplitTsData(ApplyableTsData):
    """
    Wrapper around another iterable ts data object, which splits the root ts data
    object into smaller pieces, each of size split_size.

    This means if you apply a function on this object, a temporary
    wrapper will be applied to the root ts data object, which splits
    the result into smaller chunks on which the actual function is called.
    """

    def __init__(self, root_ts_data, split_size):
        """Initialize with the root ts data object and the size to split"""
        self._root_ts_data = root_ts_data
        self._split_size = split_size

        self.column_id = self._root_ts_data.column_id

        # The resulting type will be a tuple (id, chunk number),
        # so it is an object
        self.df_id_type = object

    def apply(self, f, **kwargs):
        """Call f on the chunks of the root ts data object"""

        def wrapped_f(time_series, **kwargs):
            chunk_id, chunk_kind, chunk_data = time_series
            max_chunks = math.ceil(len(chunk_data) / self._split_size)

            for chunk_number in range(max_chunks):
                result = f(
                    Timeseries(
                        (chunk_id, chunk_number),
                        chunk_kind,
                        chunk_data.iloc[
                            chunk_number
                            * self._split_size : (chunk_number + 1)
                            * self._split_size
                        ],
                    ),
                    **kwargs
                )

                yield from result

        return self._root_ts_data.apply(wrapped_f, **kwargs)

    def pivot(self, results):
        """Pivoting can be copied from the root ts object"""
        return self._root_ts_data.pivot(results)
