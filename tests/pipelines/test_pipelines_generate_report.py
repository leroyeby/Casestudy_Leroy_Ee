"""This is a test file for the pipelines_generate_report.py file"""

import json
import pytest
from pathlib import Path
from kedro.pipeline import Pipeline

from src.casestudy_bmw_business_reporting.pipelines.pipelines_generate_report import (
    create_generate_report_full_pipeline,
    read_response_json,
    read_missing_data_warning_txt,
    construct_report_from_llm_response_str,
    _build_analysis_section,
)


# ----------------------------
# Tests for read_response_json
# ----------------------------
def test_read_response_json_reads_file(tmp_path):
    """Test that `read_response_json` correctly returns the contents of response.json."""
    file = tmp_path / "response.json"
    file.write_text('{"key": "value"}', encoding="utf-8")

    result = read_response_json(tmp_path)

    assert result == '{"key": "value"}'


def test_read_response_json_missing_file(tmp_path):
    """Test that `read_response_json` raises FileNotFoundError if response.json is missing."""
    with pytest.raises(FileNotFoundError):
        read_response_json(tmp_path)


# ---------------------------------------
# Tests for read_missing_data_warning_txt
# ---------------------------------------
def test_read_missing_data_warning_txt_exists(tmp_path):
    """Test that missing-data warning text is read if the file exists."""
    file = tmp_path / "Missing_data_warning.txt"
    file.write_text("Warning!", encoding="utf-8")

    result = read_missing_data_warning_txt(tmp_path)

    assert result == "Warning!"


def test_read_missing_data_warning_txt_not_exists(tmp_path):
    """Test that function returns empty string when warning file does not exist."""
    result = read_missing_data_warning_txt(tmp_path)
    assert result == ""


# ----------------------------------
# Tests for _build_analysis_section
# ----------------------------------
def test_build_analysis_section_adds_section(tmp_path):
    """Test `_build_analysis_section` builds markdown lines and writes plot PNGs."""
    # Arrange
    working = tmp_path / "working"
    working.mkdir()
    output = tmp_path / "output"
    output.mkdir()

    # Fake CSV
    csv_path = working / "sample.csv"
    csv_path.write_text("a,b\n1,2", encoding="utf-8")

    # Fake python snippet
    python_snippet = "plot_img = b'TESTPNG'"

    section_dict = {
        "title": "Test Section",
        "introduction": "Intro text",
        "narrative": "Narrative text",
        "plots": [
            {
                "id": "plot1",
                "data_source": "sample.csv",
                "python_snippet": python_snippet,
                "description": "A test plot",
            }
        ],
    }

    md_lines = []

    # Act
    updated = _build_analysis_section(md_lines, section_dict, working, output)

    # Assert md_lines contains expected markdown
    assert "## Test Section" in updated[0]
    assert any("![plot1]" in line for line in updated)

    # Check PNG written
    assert (output / "plot1.png").exists()
    assert (output / "plot1.png").read_bytes() == b"TESTPNG"


# -------------------------------------------------
# Tests for construct_report_from_llm_response_str
# -------------------------------------------------
def test_construct_report_from_llm_response_str_creates_md(tmp_path):
    """Test that `construct_report_from_llm_response_str` generates a report.md file."""
    # Define paths
    working = tmp_path / "working"
    output = tmp_path / "output"
    working.mkdir()
    output.mkdir()

    # Create dummy CSV for the plot
    csv_path = working / "sales_data.csv"
    csv_path.write_text("a,b\n1,2", encoding="utf-8")

    # Minimal JSON with empty sections (tests markdown generation only)
    response = {
        "executive_summary": "Summary text",
        "sections": [
            {
                "title": "Sales Performance Trends",
                "introduction": "Intro text",
                "plots": [
                    {
                        "data_source": "sales_data.csv",
                        "id": "plot1",
                        "description": "A test plot",
                        "python_snippet": "plot_img = b'TESTPNG'",
                    }
                ],
                "narrative": "Narrative text",
            },
            {
                "title": "Top/Bottom Model Performance",
                "introduction": "Intro 2",
                "plots": [
                    {
                        "data_source": "sales_data.csv",
                        "id": "plot2",
                        "description": "A test plot",
                        "python_snippet": "plot_img = b'TESTPNG'",
                    }
                ],
                "narrative": "Narrative 2",
            },
            {
                "title": "Regional Performance",
                "introduction": "Intro 3",
                "plots": [
                    {
                        "data_source": "sales_data.csv",
                        "id": "plot3",
                        "description": "A test plot",
                        "python_snippet": "plot_img = b'TESTPNG'",
                    }
                ],
                "narrative": "Narrative 3",
            },
            {
                "title": "Key Sales Drivers",
                "introduction": "Intro 4",
                "plots": [
                    {
                        "data_source": "sales_data.csv",
                        "id": "plot4",
                        "description": "A test plot",
                        "python_snippet": "plot_img = b'TESTPNG'",
                    }
                ],
                "narrative": "Narrative 4",
            },
        ],
        "recommendations": ["Do X", "Do Y"],
    }
    response_str = json.dumps(response)

    md_text = construct_report_from_llm_response_str(
        response_str=response_str,
        missing_data_warning_str="",
        working_path=str(working),
        output_path=str(output),
    )

    # Assert — check expected markdown content
    assert "## Executive Summary" in md_text
    assert "Summary text" in md_text

    # Section content
    assert "## Sales Performance Trends" in md_text
    assert "Intro text" in md_text
    assert "Narrative text" in md_text

    # Plot reference
    assert "![plot1]" in md_text

    # Recommendations
    assert "- Do X" in md_text
    assert "- Do Y" in md_text

    # Check that markdown file was created
    outfile = output / "report.md"
    assert outfile.exists()
    written_content = outfile.read_text()
    assert "Summary text" in written_content
    assert "Sales Performance Trends" in written_content

    # Check that plot PNG was written
    expected_png = output / "plot1.png"
    assert expected_png.exists()
    assert expected_png.read_bytes() == b"TESTPNG"
