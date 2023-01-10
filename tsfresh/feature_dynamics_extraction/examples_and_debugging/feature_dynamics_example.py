import dask.dataframe as dd
import pandas as pd
import numpy as np

from tsfresh.feature_dynamics_extraction.feature_dynamics_extraction import (
    extract_feature_dynamics,
)
from tsfresh.feature_extraction.settings import (
    MinimalFCParameters,
    EfficientFCParameters,
)
from tsfresh.feature_selection import select_features

# Not in tsfresh published package yet
from tsfresh.feature_dynamics_extraction.feature_dynamics_utils import (
    derive_features_dictionaries,
    gen_pdf_for_feature_dynamics,
    engineer_input_timeseries,
)

# NOTE: The intent of this file is NOT to be a test suite but more of a "debug playground"


def gen_example_timeseries_data(container_type="pandas"):

    assert container_type == "pandas" or container_type == "dask"

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


def controller(
    run_dask,
    run_pandas,
    run_efficient,
    run_minimal,
    run_select,
    run_extract_on_selected,
    engineer_more_ts,
    run_pdf,
):

    assert (
        run_dask + run_pandas < 2 and run_dask + run_pandas > 0
    ), "select one of run_dask and run_pandas"
    if run_dask:
        container_type = "dask"
    elif run_pandas:
        container_type = "pandas"

    assert (
        run_efficient + run_minimal < 2 and run_efficient + run_minimal > 0
    ), "select one of run_efficient and run_minimal"
    if run_efficient:
        # Ignore time-based feature calculators "linear_trend_timewise"
        sub_default_fc_parameters = EfficientFCParameters()
        default_fc_parameters = EfficientFCParameters()
    elif run_minimal:
        sub_default_fc_parameters = MinimalFCParameters()
        default_fc_parameters = MinimalFCParameters()

    # run_extract_on_selected ----> run_select
    assert (
        not (run_extract_on_selected) or run_select
    ), "must select features if you want to extract on selected features"

    config_dict = {
        "Container": container_type,
        "Feature Calculators": {
            "Feature Timeseries": sub_default_fc_parameters,
            "Feature Dynamics": default_fc_parameters,
        },
        "Select": run_select,
        "Extract On Selected": run_extract_on_selected,
        "Engineer More Timeseries": engineer_more_ts,
        "Explain Features with pdf": run_pdf,
    }

    return config_dict


##############################################################################################

if __name__ == "__main__":

    ###############################
    ###############################
    # Control variables here
    run_dask = False
    run_pandas = True
    run_efficient = True
    run_minimal = False
    run_select = True
    run_extract_on_selected = True
    engineer_more_ts = False
    run_pdf = True
    ###############################
    ###############################

    # Set up config
    config = controller(
        run_dask,
        run_pandas,
        run_efficient,
        run_minimal,
        run_select,
        run_extract_on_selected,
        engineer_more_ts,
        run_pdf,
    )

    # generate the data
    container_type = "dask" if run_dask else "pandas"
    ts, response = gen_example_timeseries_data(container_type=container_type)

    # Engineer some input timeseries
    if engineer_more_ts:
        if run_dask:
            ts = ts.compute()

        ts = engineer_input_timeseries(
            timeseries=ts,
            column_sort="t",
            column_id="measurement_id",
            compute_differences_within_series=True,
            compute_differences_between_series=True,
        )

        # Include second order differences:.
        ts = ts.merge(
            engineer_input_timeseries(timeseries=ts[["dt_y1", "dt_y2", "dt_y3"]])
        )
        print(ts)

        if run_dask:
            # turn pandas back to dask after engineering more input timeseries
            ts = dd.from_pandas(ts, npartitions=3)

    print(f"\nTime series input:\n\n{ts}")
    print(f"\nTime series response vector:\n\n{response}")
    window_length_1 = 4
    window_length_2 = 5

    X = extract_feature_dynamics(
        timeseries_container=ts,
        n_jobs=0,
        feature_timeseries_fc_parameters={
            window_length_1: config["Feature Calculators"]["Feature Timeseries"],
            window_length_2: config["Feature Calculators"]["Feature Timeseries"],
        },
        feature_dynamics_fc_parameters={
            window_length_1: config["Feature Calculators"]["Feature Dynamics"],
            window_length_2: config["Feature Calculators"]["Feature Dynamics"],
        },
        column_id="measurement_id",
        column_sort="t",
        column_kind=None,
        column_value=None,
        show_warnings=False,
    )
    if config["Select"]:
        # select_features does not support dask dataframes
        if config["Container"] == "dask":
            X = X.compute()

        X_filtered = select_features(X, response, fdr_level=0.95)

        # Now get names of the features
        rel_feature_names = list(X_filtered.columns)

        # Now generate a dictionary(s) to extract JUST these features
        feature_time_series_dict, feature_dynamics_dict = derive_features_dictionaries(
            rel_feature_names
        )

        # interpret feature dynamics
        if config["Explain Features with pdf"]:
            subset_of_rel_feature_names = rel_feature_names[0:100]
            gen_pdf_for_feature_dynamics(
                feature_dynamics_names=subset_of_rel_feature_names,
            )

        if config["Extract On Selected"]:
            X = extract_feature_dynamics(
                timeseries_container=ts,
                n_jobs=0,
                feature_timeseries_kind_to_fc_parameters=feature_time_series_dict,
                feature_dynamics_kind_to_fc_parameters=feature_dynamics_dict,
                column_id="measurement_id",
                column_sort="t",
                column_kind=None,
                column_value=None,
                show_warnings=False,
            )
            print(f"Relevant Feature Dynamics Matrix{X}")

        print("Success")
