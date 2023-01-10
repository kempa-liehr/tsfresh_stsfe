from tsfresh.feature_dynamics_extraction.feature_dynamics_utils import (
    derive_features_dictionaries,
    interpret_feature_dynamic,
)


def test_derive_features_dictionaries():
    x = [
        'y2||linear_trend|attr_"slope"@window_3__linear_trend__attr_"intercept"',
        "y1||percentage_of_reoccurring_values_to_all_values@window_3__energy_ratio_by_chunks__num_segments_10__segment_focus_0",
        'y2||linear_trend|attr_"slope"@window_10__linear_trend__attr_"intercept"',
        'y2||linear_trend|attr_"slope"@window_3__linear_trend__attr_"slope"',
        'y2||linear_trend|attr_"slope"@window_10__linear_trend__attr_"intercept"__attr_"slope"__attr_"slope"',
    ]
    feature_timeseries_dict, feature_dynamics_dict = derive_features_dictionaries(x)

    # Get unique window lengths from this
    print("TESTING")
    for window in feature_timeseries_dict.keys():
        print(window)
        ts_kind_dict = feature_timeseries_dict[window]
        fd_dict = feature_dynamics_dict[window]
        print(ts_kind_dict)
        print(fd_dict)


def test_interpret_feature_dynamic():
    x = [
        'y2||linear_trend|attr_"slope"@window_3__linear_trend__attr_"intercept"',
        "y1||percentage_of_reoccurring_values_to_all_values@                                                 window_3__energy_ratio_by_chunks__num_segments_10__segment_focus_0",
    ]

    stuff = interpret_feature_dynamic(x[0])
    print("Printing stuff")
    print(stuff)


if __name__ == "__main__":
    #test_derive_features_dictionaries()
    test_interpret_feature_dynamic()
