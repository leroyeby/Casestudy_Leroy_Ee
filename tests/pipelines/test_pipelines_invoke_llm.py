"""This is a test file for the pipelines_invoke_llm.py file."""

import json
import os
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.casestudy_bmw_business_reporting.pipelines.pipelines_invoke_llm import (
    load_subtables_full,
    load_subtables,
    write_prompt,
    call_google_gemini_llm,
    write_llm_response_as_json,
)

# -----------------------------
# Fixtures
# -----------------------------


@pytest.fixture
def temp_csv_dir(tmp_path):
    """Creates a temporary directory with sample CSV files for testing."""
    df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df2 = pd.DataFrame({"X": [10, 20], "Y": [30, 40]})

    file1 = tmp_path / "file1.csv"
    file2 = tmp_path / "file2.csv"

    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)

    return tmp_path


# -----------------------------
# Tests for load_subtables_full
# -----------------------------


def test_load_subtables_full_success(tmp_path):
    """Test successful loading of multiple CSV files into JSON-serializable dict."""
    # Create sample CSVs
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"x": ["p", "q"]})

    df1.to_csv(tmp_path / "file1.csv", index=False)
    df2.to_csv(tmp_path / "file2.csv", index=False)

    result = load_subtables_full(
        working_path=str(tmp_path), start_invoke_signal="ready to invoke"
    )

    # Files should be present as keys
    assert "file1.csv" in result
    assert "file2.csv" in result

    # Values should be lists of dicts
    assert result["file1.csv"] == [{"a": 1}, {"a": 2}]
    assert result["file2.csv"] == [{"x": "p"}, {"x": "q"}]


def test_load_subtables_full_wrong_signal(tmp_path):
    """Test that AssertionError is raised if the start signal is incorrect."""
    with pytest.raises(AssertionError):
        load_subtables_full(working_path=str(tmp_path), start_invoke_signal="not_ready")


def test_load_subtables_full_no_csv(tmp_path):
    """Test that FileNotFoundError is raised when no CSV files are present."""
    # tmp_path is empty → no CSVs
    with pytest.raises(FileNotFoundError):
        load_subtables_full(
            working_path=str(tmp_path), start_invoke_signal="ready to invoke"
        )


# -----------------------------
# Tests for load_subtables
# -----------------------------


def test_load_subtables_success(tmp_path):
    """Test successful loading of multiple CSV files into JSON-serializable dict."""
    # Create sample CSVs
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"x": ["p", "q"]})

    df1.to_csv(tmp_path / "file1.csv", index=False)
    df2.to_csv(tmp_path / "file2.csv", index=False)

    result = load_subtables(working_path=str(tmp_path))

    # Files should be present as keys
    assert "file1.csv" in result
    assert "file2.csv" in result

    # Values should be lists of dicts
    assert result["file1.csv"] == [{"a": 1}, {"a": 2}]
    assert result["file2.csv"] == [{"x": "p"}, {"x": "q"}]


def test_load_subtables_no_csv(tmp_path):
    """Test that FileNotFoundError is raised when no CSV files are present."""
    # tmp_path is empty → no CSVs
    with pytest.raises(FileNotFoundError):
        load_subtables(working_path=str(tmp_path))


# -----------------------------
# Tests for write_prompt
# -----------------------------


def test_write_prompt_returns_string():
    """Test that write_prompt returns a string."""
    json_input = {"table1.csv": [{"a": 1, "b": 2}]}
    result = write_prompt(json_input)

    assert isinstance(result, str)
    assert len(result) > 0


def test_write_prompt_embeds_json_dict():
    """Test that the JSON dict appears inside the returned prompt."""
    json_input = {"sales.csv": [{"col": 123}]}

    result = write_prompt(json_input)

    # JSON string should be inside the final prompt
    assert str(json_input) in result


def test_write_prompt_contains_required_sections():
    """Test that critical prompt instructions exist."""
    json_input = {"dummy.csv": []}
    result = write_prompt(json_input)

    # Check that essential headings and text exist. Useful only if prompt is later edited.
    required_substrings = [
        "You are an expert business analyst and data storyteller.",
        "ANALYSIS REQUIREMENTS",
        "SECTION 1 — Sales Performance Trends",
        "SECTION 2 — Top/Bottom Model Performance",
        "SECTION 3 — Regional Performance",
        "SECTION 4 — Key Sales Drivers",
        "OUTPUT FORMAT RULES",
        '"executive_summary": string',
        "DATA INPUT",
    ]

    for substring in required_substrings:
        assert substring in result


def test_write_prompt_includes_python_snippet_rules():
    """Test the prompt includes mandatory python snippet formatting requirements."""
    result = write_prompt({"x": []})

    assert "df = pd.DataFrame(data)" in result
    assert "plt.savefig(buf, format='png')" in result
    assert "plot_img = buf.getvalue()" in result


def test_write_prompt_handles_empty_dict():
    """Test that function handles empty data input gracefully."""
    result = write_prompt({})

    assert "{}" in result
    assert isinstance(result, str)
    assert len(result) > 0


# -----------------------------
# Tests for call_google_gemini_llm
# -----------------------------


@patch(
    "src.casestudy_bmw_business_reporting.pipelines.pipelines_invoke_llm.genai.Client"
)
@patch("src.casestudy_bmw_business_reporting.pipelines.pipelines_invoke_llm.os.getenv")
@patch(
    "src.casestudy_bmw_business_reporting.pipelines.pipelines_invoke_llm.load_dotenv"
)
def test_call_google_gemini_llm_returns_text(
    mock_load_dotenv, mock_getenv, mock_client
):
    """Test that the function correctly extracts JSON text from Gemini response."""
    mock_getenv.side_effect = lambda key: {
        "GEMINI_API_KEY": "key",
        "GEMINI_MODEL": "model",
    }[key]

    # Mock the client response
    mock_response = MagicMock()
    mock_response.text = '```json{"key": "value"}```'
    mock_client.return_value.models.generate_content.return_value = mock_response

    prompt = "dummy prompt"
    result = call_google_gemini_llm(prompt)

    # Check that it returns cleaned JSON string
    assert result == '{"key": "value"}'
    mock_client.return_value.models.generate_content.assert_called_once_with(
        model="model", contents=prompt
    )


# -----------------------------
# Tests for write_llm_response_as_json
# -----------------------------


def test_write_llm_response_as_json_creates_file(tmp_path):
    """Test that the function writes a valid JSON file to the given directory."""
    response_str = '{"key": "value"}'
    write_llm_response_as_json(response_str, str(tmp_path))

    file_path = tmp_path / "response.json"
    assert file_path.exists()

    with open(file_path) as f:
        data = json.load(f)
    assert data["key"] == "value"


def test_write_llm_response_as_json_invalid_json(tmp_path):
    """Test that the function raises JSONDecodeError for invalid JSON input."""
    invalid_json = '{"key": "value"'  # Missing closing brace
    with pytest.raises(json.JSONDecodeError):
        write_llm_response_as_json(invalid_json, str(tmp_path))


def test_write_llm_response_as_json_outputs_str(tmp_path):
    """Test that the function outputs the correct signal string."""
    response_str = '{"key": "value"}'
    result = write_llm_response_as_json(response_str, str(tmp_path))

    assert result == "ready to generate report"
