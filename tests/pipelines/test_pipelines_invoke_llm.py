"""This is a test file for the pipelines_invoke_llm.py file."""

import json
import os
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.casestudy_bmw_business_reporting.pipelines.pipelines_invoke_llm import (
    load_subtables_and_write_prompt,
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
# Tests for load_subtables_and_write_prompt
# -----------------------------


def test_load_subtables_and_write_prompt_creates_prompt(temp_csv_dir):
    """Test that the prompt includes all CSV filenames and required instructions."""
    prompt = load_subtables_and_write_prompt(str(temp_csv_dir))
    assert "file1.csv" in prompt
    assert "file2.csv" in prompt
    assert "You are an expert business analyst" in prompt


def test_load_subtables_and_write_prompt_no_csv(tmp_path):
    """Test that the function raises FileNotFoundError if no CSV files exist."""
    with pytest.raises(FileNotFoundError):
        load_subtables_and_write_prompt(str(tmp_path))


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
