"""This is a test file for the pipelines_preprocessing.py file"""

import pytest
import pandas as pd
from pathlib import Path
from pandas.testing import assert_frame_equal
from sqlalchemy import create_engine, inspect
import os

# Import your functions
from src.casestudy_bmw_business_reporting.pipelines.pipelines_preprocessing import (
    load_xlsx_to_df,
    check_and_warn_percent_missing_values,
    clean_data,
    engineer_features,
    export_df_to_csv,
    export_df_to_sqldb,
)


# --------------------------
# Fixtures
# --------------------------
@pytest.fixture
def sample_df():
    """Return a sample DataFrame for testing purposes."""
    return pd.DataFrame(
        {
            "Year": [2020, 2021, 2022],
            "Model": ["M8", "i3", "i5"],
            "Price_USD": [20000, 30000, 25000],
            "Sales_Volume": [10, 20, 30],
            "Mileage_KM": [1000, 1500, 2000],
            "Engine_Size_L": [2, 3, 2.5],
        }
    )


# --------------------------
# Tests for load_xlsx_to_df
# --------------------------
def test_load_xlsx_to_df_success(tmp_path, sample_df):
    """Test successful loading of a single Excel file with all required columns."""
    file_path = tmp_path / "sample file.xlsx"
    sample_df.to_excel(file_path, index=False)

    filename, df = load_xlsx_to_df(tmp_path, required_cols=set(sample_df.columns))
    assert filename == "sample_file"
    assert_frame_equal(df, sample_df)


def test_load_xlsx_to_df_no_file(tmp_path):
    """Test that FileNotFoundError is raised when no Excel file exists."""
    with pytest.raises(FileNotFoundError):
        load_xlsx_to_df(tmp_path, required_cols={"A"})


def test_load_xlsx_to_df_multiple_files(tmp_path, sample_df):
    """Test that AssertionError is raised when multiple Excel files are found."""
    sample_df.to_excel(tmp_path / "file1.xlsx", index=False)
    sample_df.to_excel(tmp_path / "file2.xlsx", index=False)

    with pytest.raises(AssertionError):
        load_xlsx_to_df(tmp_path, required_cols=set(sample_df.columns))


def test_load_xlsx_to_df_missing_columns(tmp_path, sample_df):
    """Test that ValueError is raised when required columns are missing."""
    file_path = tmp_path / "sample.xlsx"
    sample_df.to_excel(file_path, index=False)

    with pytest.raises(ValueError):
        load_xlsx_to_df(tmp_path, required_cols=set(sample_df.columns) | {"MissingCol"})


# --------------------------
# Tests for check_and_warn_percent_missing_values
# --------------------------
def test_check_and_warn_percent_missing_values_warning(tmp_path):
    """Test that a warning file is created when missing values exceed the limit."""
    df = pd.DataFrame({"A": [1, None], "B": [None, 2]})
    df_full = ("test", df)

    # Upper limit 0.3 -> should trigger warning
    check_and_warn_percent_missing_values(
        df_full, missing_upper_limit=0.3, working_path=str(tmp_path)
    )

    warning_file = tmp_path / "Missing_data_warning.txt"
    assert warning_file.exists()
    content = warning_file.read_text()
    assert "Warning" in content


def test_check_and_warn_percent_missing_values_no_warning(tmp_path):
    """Test that no warning file is created when missing values are below the limit."""
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df_full = ("test", df)

    check_and_warn_percent_missing_values(
        df_full, missing_upper_limit=0.5, working_path=str(tmp_path)
    )

    warning_file = tmp_path / "Missing_data_warning.txt"
    # File should not be created because no missing values exceed limit
    assert not warning_file.exists()


def test_check_and_warn_percent_missing_values_invalid_limit(tmp_path):
    """Test that AssertionError is raised when missing_upper_limit is invalid."""
    df = pd.DataFrame({"A": [1]})
    df_full = ("test", df)
    with pytest.raises(AssertionError):
        check_and_warn_percent_missing_values(
            df_full, missing_upper_limit=1.5, working_path=str(tmp_path)
        )


# --------------------------
# Tests for clean_data
# --------------------------
def test_clean_data_drop_rows(sample_df):
    """Test that rows with missing values are dropped when 'drop_rows' is specified."""
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "Price_USD"] = None
    df_full = ("test", df_with_nan)

    filename, df_cleaned = clean_data(df_full, missing_handling="drop_rows")
    assert df_cleaned.shape[0] == df_with_nan.shape[0] - 1


def test_clean_data_ignore(sample_df):
    """Test that all rows are kept when 'ignore' is specified."""
    df_full = ("test", sample_df)
    filename, df_cleaned = clean_data(df_full, missing_handling="ignore")
    assert_frame_equal(df_cleaned, sample_df)


def test_clean_data_invalid_option(sample_df):
    """Test that AssertionError is raised for an invalid missing_handling option."""
    df_full = ("test", sample_df)
    with pytest.raises(AssertionError):
        clean_data(df_full, missing_handling="invalid")


# --------------------------
# Tests for engineer_features
# --------------------------
def test_engineer_features(sample_df):
    """Test feature engineering produces expected tables and columns."""
    params = {
        "target_variables": ["Sales_Volume", "Revenue_USD"],
        "feature_params": {
            "fuel_efficiency_bin_labels": ["Low", "High"],
            "price_range_bin_labels": ["Entry", "High-end"],
            "categorical_variables": ["Model"],
            "model_series_map": {"M series": ["M8"], "i Series": ["i3", "i5"]},
        },
    }

    df_full = ("test", sample_df)
    result = engineer_features(df_full, params)
    # Check returned list contains expected number of tables
    assert any(name == "Overall_sales_yearly" for name, _ in result)
    assert any(name.startswith("Aggregated_") for name, _ in result)
    assert any(name == "Sample_rows_for_full_table" for name, _ in result)


# --------------------------
# Tests for export_df_to_csv
# --------------------------
def test_export_df_to_csv_list(tmp_path, sample_df):
    """Test exporting a list of DataFrames to CSV."""
    df_list = [("df1", sample_df)]
    export_df_to_csv(df_list, working_path=str(tmp_path))
    assert (tmp_path / "df1.csv").exists()


def test_export_df_to_csv_tuple(tmp_path, sample_df):
    """Test exporting a single DataFrame tuple to CSV."""
    export_df_to_csv(("df2", sample_df), working_path=str(tmp_path))
    assert (tmp_path / "df2.csv").exists()


def test_export_df_to_csv_invalid_input(tmp_path):
    """Test that AssertionError is raised for invalid input type."""
    with pytest.raises(AssertionError):
        export_df_to_csv("invalid_input", working_path=str(tmp_path))


# --------------------------
# Tests for export_df_to_sqldb
# --------------------------
def test_export_df_to_sqldb_list(tmp_path, sample_df):
    """Test exporting a list of DataFrames to SQLite database."""
    df_list = [("df1", sample_df)]
    export_df_to_sqldb(df_list, working_path=str(tmp_path))
    db_file = tmp_path / "sqldb.db"
    assert db_file.exists()

    # Check table exists
    engine = create_engine(f"sqlite:///{db_file}")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "df1" in tables


def test_export_df_to_sqldb_tuple(tmp_path, sample_df):
    """Test exporting a single DataFrame tuple to SQLite database."""
    export_df_to_sqldb(("df2", sample_df), working_path=str(tmp_path))
    db_file = tmp_path / "sqldb.db"
    assert db_file.exists()

    engine = create_engine(f"sqlite:///{db_file}")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "df2" in tables
