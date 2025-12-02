"""
Preprocessing pipeline for BMW sales data using Kedro.

This module defines a complete Kedro pipeline and its associated nodes for loading, validating, cleaning, feature-engineering, and exporting sales data. The pipeline is intended to automate preprocessing steps required before analytical modelling, reporting, or database creation.


Pipelines
---------
- `create_preprocessing_full_pipeline`:
    Creates the full preprocessing pipeline

- `create_preprocessing_no_feature_engineering_pipeline`:
    Creates the preprocessing sub pipeline eschewing the feature engineering node. The data loaded into the llm would be just the single cleaned dataset.


Nodes
-----
- `load_xlsx_to_df`:
    Loads and validates the Excel dataset.

- `check_and_warn_percent_missing_values`:
    Evaluates missing data and outputs a warning file.

- `clean_data`:
    Cleans the DataFrame based on the chosen missing-value strategy.

- `engineer_features`:
    Produces domain-driven features and aggregated summary tables.

- `export_df_to_csv`:
    Saves cleaned or feature-engineered datasets to CSV files.

- `export_df_to_sqldb`:
    Saves cleaned or feature-engineered datasets to an SQL Database.

Notes
-----
This preprocessing pipeline is designed to support downstream analytics, visualization, and report-generation workflows by structuring data consistently and safely before use.
"""

from pathlib import Path
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from kedro.pipeline import Pipeline, node


# ============================
# Pipelines
# ============================


def create_preprocessing_full_pipeline(
    **kwargs,
) -> Pipeline:
    """Kedro Pipeline created to house nodes that runs and conducts data preprocessing and creation of csv subtables."""
    return Pipeline(
        [
            node(
                func=load_xlsx_to_df,
                inputs=["params:raw_data_path", "params:required_cols"],
                outputs="df_full",
                name="load_xlsx_to_df",
            ),
            node(
                func=check_and_warn_percent_missing_values,
                inputs=["df_full", "params:missing_upper_limit", "params:working_path"],
                outputs=None,
                name="check_and_warn_percent_missing_values",
            ),
            node(
                func=clean_data,
                inputs=["df_full", "params:missing_handling"],
                outputs="df_full_cleaned",
                name="clean_data",
            ),
            node(
                func=engineer_features,
                inputs=["df_full_cleaned", "params:feature_engineering"],
                outputs="df_list_to_export",
                name="engineer_features",
            ),
            node(
                func=export_df_to_csv,
                inputs=["df_list_to_export", "params:working_path"],
                outputs=None,
                name="export_df_to_csv",
            ),
        ]
    )


def create_preprocessing_sqldb_pipeline(
    **kwargs,
) -> Pipeline:
    """Kedro Pipeline created to house nodes that runs and conducts data preprocessing and creation of csv subtables. This pipeline eschews feature engineering."""
    return Pipeline(
        [
            node(
                func=load_xlsx_to_df,
                inputs=["params:raw_data_path", "params:required_cols"],
                outputs="df_full",
                name="load_xlsx_to_df",
            ),
            node(
                func=check_and_warn_percent_missing_values,
                inputs=["df_full", "params:missing_upper_limit", "params:working_path"],
                outputs=None,
                name="check_and_warn_percent_missing_values",
            ),
            node(
                func=clean_data,
                inputs=["df_full", "params:missing_handling"],
                outputs="df_full_cleaned",
                name="clean_data",
            ),
            node(
                func=export_df_to_sqldb,
                inputs=["df_to_export", "params:working_path"],
                outputs=None,
                name="export_df_to_sqldb",
            ),
        ]
    )


# ============================
# Nodes
# ============================


def load_xlsx_to_df(
    raw_data_path: str, required_cols: set[str]
) -> tuple[str, pd.DataFrame]:
    """
    Loads a single Excel (.xlsx) file from a specified directory into a pandas DataFrame and validates the presence of required columns.

    This function expects the directory to contain exactly one Excel file. It reads the file, sanitizes the filename for use as a dataset name, and ensures that all required columns are present in the DataFrame.

    Args:
        raw_data_path (str): Path to the folder containing the Excel file.
        required_cols (set[str]): Set of column names that must exist in the Excel file.

    Raises:
        FileNotFoundError: If no Excel (.xlsx) file is found in the specified folder.
        AssertionError: If more than one Excel file is found in the folder.
        ValueError: If the Excel file is missing any of the required columns.

    Returns:
        tuple[str, pd.DataFrame]: A tuple containing:
            - filename (str): Sanitized name of the Excel file (spaces and dashes replaced with underscores, parentheses removed).
            - df (pd.DataFrame): The loaded DataFrame.
    """
    raw_data_path = Path(raw_data_path)

    xlsx_files = list(raw_data_path.glob("*.xlsx"))

    if not xlsx_files:
        raise FileNotFoundError(
            f"No xlsx file found in the raw data folder {str(raw_data_path)}. Please input data before proceeding."
        )
    if len(xlsx_files) > 1:
        raise AssertionError(
            f"{str(raw_data_path)} expected to only contain 1 .xlsx file. Please check the folder before proceeding."
        )
    data = xlsx_files[0]
    filename = (
        data.name.split(".")[0]
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
    )
    df = pd.read_excel(data)
    df_full = (filename, df)

    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"{data.name} does not contain required columns: {', '.join(missing_cols)}"
        )

    return df_full


def check_and_warn_percent_missing_values(
    df_full: tuple[str, pd.DataFrame], missing_upper_limit: float, working_path: str
) -> None:
    """
    Checks the percentage of missing values in a DataFrame and generates a warning txt file if the percentage exceeds a specified threshold.

    This function calculates the proportion of missing values in the given DataFrame. If the proportion exceeds `missing_upper_limit`, it writes a warning message to a CSV file named `missing_data_warning.csv` in the specified working directory.

    Args:
        df_full (tuple[str, pd.DataFrame]): A tuple containing:
            - filename (str): Name of the DataFrame/file for reporting.
            - df (pd.DataFrame): The DataFrame to check for missing values.
        missing_upper_limit (float): Maximum allowed proportion of missing
            values (between 0 and 1 inclusive). Raises an AssertionError if the value is outside this range or not a number.
        working_path (str): Directory path where the warning CSV file will be
            saved.

    Raises:
        AssertionError: If `missing_upper_limit` is not a number between 0 and
            1.

    Returns:
        None

    Side Effects:
        Writes a txt file `missing_data_warning.txt` in `working_path` containing the warning message if the missing value proportion exceeds the limit or a blank file if there is none.
    """
    if (
        not isinstance(missing_upper_limit, (int, float))
        or not 0 <= missing_upper_limit <= 1
    ):
        raise AssertionError(
            f"missing_upper_limit parameter must be an int or float between 0 and 1 inclusive. It is currently {str(missing_upper_limit)}."
        )

    # Missing_data_warning.txt will be utilized during the report generation
    filename, df = df_full
    count_missing_values = df.isna().sum().sum()
    num_rows, num_cols = df.shape
    percent_missing_values = count_missing_values / (num_rows * num_cols)
    if percent_missing_values > missing_upper_limit:
        missing_data_warning = f"Warning: {filename} contains {percent_missing_values*100}% missing values, exceeding upper limit of {missing_upper_limit*100}%. Interpret generated report with caution."
        with open(
            working_path + "/Missing_data_warning.txt", "w", encoding="utf-8"
        ) as f:
            f.write(missing_data_warning)

    return None


def clean_data(
    df_full: tuple[str, pd.DataFrame], missing_handling: str
) -> tuple[str, pd.DataFrame]:
    """
    Cleans a DataFrame by handling missing values according to the specified strategy.

    This function takes a tuple containing a filename and a DataFrame, and applies a missing value handling strategy. Currently, it supports either dropping rows with any missing values or ignoring missing values.

    Args:
        df_full (tuple[str, pd.DataFrame]): A tuple containing:
            - filename (str): The name of the dataset or file.
            - df (pd.DataFrame): The DataFrame to clean.
        missing_handling (str): Strategy for handling missing values. Must be
            one of:
            - "drop_rows": Remove all rows containing any missing values.
            - "ignore": Keep all rows, including those with missing values.

    Raises:
        AssertionError: If `missing_handling` is not "drop_rows" or "ignore".

    Returns:
        tuple[str, pd.DataFrame]: A tuple containing the filename and the cleaned DataFrame.
    """
    if missing_handling not in ["drop_rows", "ignore"]:
        # ignore is there to ensure user knows what they want
        raise AssertionError(
            f"missing_handling parameter must be either 'drop_rows' or 'ignore'. It is currently {missing_handling}."
        )

    filename, df = df_full

    if missing_handling == "drop_rows":
        df = df.dropna()
    df_full_cleaned = (filename, df)

    return df_full_cleaned


def engineer_features(
    df_full: tuple[str, pd.DataFrame], feature_engineering_params: dict
) -> list[tuple[str, pd.DataFrame]]:
    """
    Perform feature engineering on a cleaned sales dataset and generate summary tables for analysis.

    This function applies domain-specific transformations and aggregations to the input DataFrame, such as calculating fuel efficiency, binning numeric features into ranges, computing revenue, and summarizing sales performance over time. It also prepares multiple DataFrames for export, including yearly sales and sales by categorical variables.

    Args:
        df_full (tuple[str, pd.DataFrame]):
            A tuple containing the filename (str) and the cleaned DataFrame (pd.DataFrame) to process.
        feature_engineering_params (dict):
            A dictionary containing parameters for feature engineering:
                - "target_variables" (list[str]): Columns to treat as target variables (e.g., "Sales_Volume", "Revenue_USD").
                - "feature_params" (dict): Parameters for feature creation, including:
                    - "fuel_efficiency_bin_labels" (list[str]): Labels for fuel efficiency bins.
                    - "price_range_bin_labels" (list[str]): Labels for price tier bins.
                    - "categorical_variables" (list[str]): Columns to aggregate by for yearly summaries.

    Returns:
        list[tuple[str, pd.DataFrame]]:
            A list of tuples where each tuple contains a descriptive name (str) and the corresponding DataFrame (pd.DataFrame) prepared for export or analysis.
            The returned DataFrames include:
                - Yearly aggregated sales and revenue.
                - Yearly sales and revenue by each categorical variable.
    """
    df = df_full[1]

    target_variables = feature_engineering_params["target_variables"]
    feature_params = feature_engineering_params["feature_params"]

    # customers might be influcenced by a car's fuel efficiency. Bin for easier processing
    df["Fuel_Efficiency_KMperL"] = (df["Mileage_KM"] / df["Engine_Size_L"]).astype(int)
    df["Fuel_Efficiency_Range"] = pd.cut(
        df["Fuel_Efficiency_KMperL"],
        bins=len(feature_params["fuel_efficiency_bin_labels"]),
        labels=feature_params["fuel_efficiency_bin_labels"],
    )

    # customers may be influenced by the tier of car they can afford.
    df["Price_Tier"] = pd.cut(
        df["Price_USD"],
        bins=len(feature_params["price_range_bin_labels"]),
        labels=feature_params["price_range_bin_labels"],
    )

    # certain model series might perform better than others in terms of sales
    model_series_map = feature_params["model_series_map"]

    model_series_lookup = (
        pd.Series(model_series_map)
        .explode()
        .rename_axis("Model_Series")
        .reset_index(name="Model")
    )
    df = df.merge(model_series_lookup, on="Model", how="left")

    # business unit should be interested in total revenue
    df["Revenue_USD"] = df["Price_USD"] * df["Sales_Volume"]

    # treat "Sales_Volume" and "Revenue_USD" as potential target variables so drop any NaN in those columns regardless of missing_handling parameter.
    df = df.dropna(subset=target_variables)

    # handle "Year" separately
    categorical_variables = feature_params["categorical_variables"]

    df_list_to_export = []

    # Overall sales performance trend over time
    df_yearly = (
        df.groupby(["Year"], observed=True)
        .agg(
            Sales_Volume_summed=("Sales_Volume", "sum"),
            Total_Revenue_USD=("Revenue_USD", "sum"),
        )
        .reset_index()
    )
    df_list_to_export.append(("Overall_sales_yearly", df_yearly))

    # Aggregate tables grouped by each chosen categorical variable
    for var in categorical_variables:
        df_var_yearly = (
            df.groupby([var], observed=True)
            .agg(
                Sales_Volume_summed=("Sales_Volume", "sum"),
                Sales_Volume_mean=("Sales_Volume", "mean"),
                Total_Revenue_USD=("Revenue_USD", "sum"),
                Mean_Revenue_USD=("Revenue_USD", "mean"),
            )
            .astype({"Sales_Volume_mean": int, "Mean_Revenue_USD": int})
            .reset_index()
        )
        df_list_to_export.append((f"Aggregated_{var}_sales_yearly", df_var_yearly))

    # Include a sample of the engineered full dataframe for the llm to understand the full table schema
    df_sample = df.head(10)
    df_list_to_export.append(("Sample_rows_for_full_table", df_sample))

    return df_list_to_export


def export_df_to_csv(
    df_list_or_tuple: list[tuple[str, pd.DataFrame]] | tuple[str, pd.DataFrame],
    working_path: str,
) -> None:
    """
    Export one or multiple DataFrames to CSV format.

    This function accepts either:
    - A single `(filename, DataFrame)` tuple, or
    - A list of such tuples.

    Each DataFrame is exported as a CSV file to the specified working directory,
    with the filename derived from the provided tuple. The function distinguishes
    between list and tuple inputs using `isinstance` and handles each case accordingly.

    Args:
        df_list_or_tuple (list[tuple[str, pd.DataFrame]] | tuple[str, pd.DataFrame]):
            Either a single tuple containing a filename and DataFrame, or a list of
            such tuples. Each filename (str) is used to name the output CSV file, and
            each DataFrame (pd.DataFrame) is written to disk.
        working_path (str):
            The directory path where the CSV files will be saved.

    Raises:
        AssertionError:
            If `df_list_or_tuple` is neither a tuple nor a list of tuples.

    Returns:
        None
            This function does not return anything. It writes CSV files to disk.
    """
    if isinstance(df_list_or_tuple, list):
        # list detected. Feature engineered. Subtables generated.
        for filename, df in df_list_or_tuple:
            df.to_csv(f"{working_path}/{filename}.csv", index=False)
    elif isinstance(df_list_or_tuple, tuple):
        # tuple detected. Full dataframe used.
        filename, df = df_list_or_tuple
        df.to_csv(f"{working_path}/{filename}.csv", index=False)
    else:
        raise AssertionError(
            "Input variable df_list_or_tuple accepts only lists or tuple"
        )


def export_df_to_sqldb(
    df_list_or_tuple: list[tuple[str, pd.DataFrame]] | tuple[str, pd.DataFrame],
    working_path: str,
) -> None:
    """
    Export one or multiple pandas DataFrames to an SQLite database.

    This function accepts either:
    - A list of `(table_name, DataFrame)` tuples, or
    - A single `(table_name, DataFrame)` tuple.

    It writes each DataFrame into an SQLite database located at
    `<working_path>/sqldb.db`. If the database does not exist, it is created.
    Each DataFrame is stored as a table using its associated name.

    Args:
        df_list_or_tuple (list[tuple[str, pd.DataFrame]] | tuple[str, pd.DataFrame]):
            A single `(table_name, DataFrame)` tuple or a list of such tuples.
            Each tuple specifies the name of the SQL table and the DataFrame
            that will be written into the database.
        working_path (str):
            Directory path where the SQLite database file (`sqldb.db`)
            will be stored or created.

    Returns:
        None: This function does not return anything. It writes tables directly
        into the SQLite database.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: If any issue occurs while writing to the
            SQLite database.
        ValueError: If the provided input is not a tuple or list of tuples.

    """
    db_path = working_path + "/sqldb.db"
    db_path = f"sqlite:///{db_path}"

    engine = create_engine(db_path)

    if isinstance(df_list_or_tuple, list):
        # list detected. Feature engineered. Subtables generated.
        for filename, df in df_list_or_tuple:
            df.to_sql(filename, engine, index=False)
    elif isinstance(df_list_or_tuple, tuple):
        # tuple detected. Full dataframe used.
        filename, df = df_list_or_tuple
        df.to_sql(filename, engine, index=False)
