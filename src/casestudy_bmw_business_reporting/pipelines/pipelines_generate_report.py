"""
Report Generation Pipeline using Kedro

This module defines a Kedro pipeline and associated nodes for generating a structured business report from LLM (Large Language Model) response data. It provides functionality to read input JSON responses, handle missing data warnings and construct markdown reports with analysis sections and plots.

The generated report includes:
- Executive summary
- Analysis sections (e.g., Sales Performance Trends, Top/Bottom Model Performance, Regional Performance, Key Sales Drivers)
- Recommendations
- Embedded plots generated from CSV data sources


Pipelines
---------
- `create_generate_report_full_pipeline`:
    Creates the full Kedro pipeline for report generation. Used in full pipeline.

- `create_generate_report_only_pipeline`:
    Creates the full Kedro pipeline for report generation. Used when only generating report.

Nodes
-----
- `read_response_json_full`:
    Reads an LLM response JSON file and returns its raw string content. For use in full pipeline.

- `read_response_json`:
    Reads an LLM response JSON file and returns its raw string content. For use when only generating report.

- `read_missing_data_warning_txt`:
    Reads an optional missing data warning text file, returning its content or an empty string.

- `construct_report_from_llm_response_str`:
    Generates a markdown report from the LLM response string and saves it to disk.


Helper Functions
----------------
- `_build_analysis_section`:
    Helper function that builds a markdown section with plots and narratives for the report.


Notes
-----
Plots are generated dynamically from code snippets provided in the LLM response.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from kedro.pipeline import Pipeline, node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# Pipelines
# ============================


def create_generate_report_full_pipeline(
    **kwargs,
) -> Pipeline:
    """Kedro Pipeline created to house nodes that loads subtables for input and invokes the llm for the generation of a structured output. This pipeline is used when invoking the full process of preprocessing into invoking into report generation."""
    return Pipeline(
        [
            node(
                func=read_response_json_full,
                inputs=["params:working_path", "start_report_gen_signal"],
                outputs="response_str_to_report",
                name="read_response_json",
            ),
            node(
                func=read_missing_data_warning_txt,
                inputs="params:working_path",
                outputs="missing_data_warning_str",
                name="read_missing_data_warning_txt",
            ),
            node(
                func=construct_report_from_llm_response_str,
                inputs=[
                    "response_str_to_report",
                    "missing_data_warning_str",
                    "params:working_path",
                    "params:output_path",
                ],
                outputs="business_report_md",
                name="construct_report_from_llm_response_dict",
            ),
        ]
    )


def create_generate_report_only_pipeline(
    **kwargs,
) -> Pipeline:
    """Kedro Pipeline created to house nodes that loads subtables for input and invokes the llm for the generation of a structured output. This pipeline is used when generating report by itself. It presumes invoke llm pipeline has already been executed."""
    return Pipeline(
        [
            node(
                func=read_response_json,
                inputs="params:working_path",
                outputs="response_str_to_report",
                name="read_response_json",
            ),
            node(
                func=read_missing_data_warning_txt,
                inputs="params:working_path",
                outputs="missing_data_warning_str",
                name="read_missing_data_warning_txt",
            ),
            node(
                func=construct_report_from_llm_response_str,
                inputs=[
                    "response_str_to_report",
                    "missing_data_warning_str",
                    "params:working_path",
                    "params:output_path",
                ],
                outputs="business_report_to_export",
                name="construct_report_from_llm_response_dict",
            ),
        ]
    )


# ============================
# Nodes
# ============================


def read_response_json_full(working_path: str, start_report_gen_signal: str) -> str:
    """
    Read the contents of the LLM response JSON file from the specified directory and return it as a raw JSON string. For use in the full preprocessing into invoking into report generation pipeline.

    Args:
        working_path (str): Path to the directory containing the ``response.json`` file.
        start_report_gen_signal (str): String signalling that llm invoking is complete and json file is in the working directory ready to read.
    Returns:
        str: The raw JSON string read from ``response.json``.

    Raises:
        ValueError: If start_report_gen_signal variable received an unexpected str.
        FileNotFoundError: If ``response.json`` does not exist in the specified directory.
        OSError: If the file cannot be opened or read.
    """
    if start_report_gen_signal != "ready to generate report":
        raise ValueError(
            f"start_report_gen_signal variable is supposed to read 'ready to generate report'. Received signal '{start_report_gen_signal}' instead."
        )

    working_path = Path(working_path)
    response_json_filepath = working_path / "response.json"

    if not response_json_filepath.exists():
        raise FileNotFoundError(
            f"No json file found in the working data folder {str(working_path)}. Please run invoke llm pipeline before proceeding."
        )

    with open(response_json_filepath, "r", encoding="utf-8") as f:
        json_str = f.read()

    return json_str


def read_response_json(working_path: str) -> str:
    """
    Read the contents of the LLM response JSON file from the specified directory and return it as a raw JSON string. For use in 'generate_report_only' pipeline.

    Args:
        working_path (str): Path to the directory containing the ``response.json`` file.

    Returns:
        str: The raw JSON string read from ``response.json``.

    Raises:
        FileNotFoundError: If ``response.json`` does not exist in the specified directory.
        OSError: If the file cannot be opened or read.
    """
    working_path = Path(working_path)
    response_json_filepath = working_path / "response.json"

    if not response_json_filepath.exists():
        raise FileNotFoundError(
            f"No json file found in the working data folder {str(working_path)}. Please run invoke llm pipeline before proceeding."
        )

    with open(response_json_filepath, "r", encoding="utf-8") as f:
        json_str = f.read()

    return json_str


def read_missing_data_warning_txt(working_path: str) -> str:
    """
    Read the optional ``Missing_data_warning.txt`` file from the specified directory and return its contents as a string.

    If the file does not exist, an empty string is returned.

    Args:
        working_path (str): Path to the directory that may contain
            ``Missing_data_warning.txt``.

    Returns:
        str: The contents of the missing data warning file, or an empty string
        if the file is not present.

    Raises:
        OSError: If the file exists but cannot be opened or read.
    """
    working_path = Path(working_path)
    missing_data_warning_file = working_path / "Missing_data_warning.txt"

    missing_data_warning_str = ""
    if missing_data_warning_file.exists():
        with open(missing_data_warning_file, "r", encoding="utf-8") as f:
            missing_data_warning_str = f.read()

    return missing_data_warning_str


def construct_report_from_llm_response_str(
    response_str: str,
    missing_data_warning_str: str,
    working_path: str,
    output_path: str,
) -> None:
    """
    Constructs a complete markdown report from a JSON-formatted LLM response string.

    The function parses the LLM response string, generates markdown sections including an executive summary, multiple analysis sections, and recommendations, incorporates any missing data warnings, and saves the final report as `report.md` in the specified output path.

    Args:
        response_str (str): JSON-formatted string containing the LLM response with keys:
            - 'executive_summary' (str)
            - 'sections' (list[dict])
            - 'recommendations' (list[str])
        missing_data_warning_str (str): Optional warning text regarding missing data to be appended in relevant sections.
        working_path (str): Path to the directory containing source CSV files for plots.
        output_path (str): Path to the directory where the markdown report and plot images will be saved.

    Returns:
        None: The function writes the report to disk and does not return a value.
    """
    response_dict = json.loads(response_str)

    md_lines = []
    # Report title
    md_lines.append(
        '<div align="center">\n\n'
        "# BMW Sales Report\n"
        f"Generated: {datetime.utcnow().isoformat()}Z\n\n"
        "</div>\n"
    )

    # Executive summary section
    md_lines.append("## Executive Summary")
    md_lines.append(response_dict["executive_summary"])
    if missing_data_warning_str:
        md_lines.append("\n")
        md_lines.append(missing_data_warning_str)
    md_lines.append('<div style="page-break-after: always;"></div>')

    # Analysis sections
    section_list = response_dict["sections"]
    ## Sales Performance Trends
    section = next(
        (
            dict
            for dict in section_list
            if dict.get("title") == "Sales Performance Trends"
        ),
        None,
    )
    md_lines = _build_analysis_section(md_lines, section, working_path, output_path)
    ## Top/Bottom Model Performance
    section = next(
        (
            dict
            for dict in section_list
            if dict.get("title") == "Top/Bottom Model Performance"
        ),
        None,
    )
    md_lines = _build_analysis_section(md_lines, section, working_path, output_path)
    ## Regional Performance
    section = next(
        (dict for dict in section_list if dict.get("title") == "Regional Performance"),
        None,
    )
    md_lines = _build_analysis_section(md_lines, section, working_path, output_path)
    ## Key Sales Drivers
    section = next(
        (dict for dict in section_list if dict.get("title") == "Key Sales Drivers"),
        None,
    )
    md_lines = _build_analysis_section(md_lines, section, working_path, output_path)

    # Recommendations section
    md_lines.append("## Recommendations")
    for recommendation in response_dict["recommendations"]:
        md_lines.append(f"- {recommendation}")

    md_text = "\n\n".join(md_lines)

    output_path = Path(output_path)
    open(output_path / "report.md", "w", encoding="utf8").write(md_text)

    return md_text


# ============================
# Helper Functions
# ============================


def _build_analysis_section(
    md_lines: list[any],
    section_dict: dict[str:any],
    working_path: str,
    output_path: str,
) -> list[any]:
    """
    Builds a markdown section for a report based on the provided section dictionary, including plots,
    narratives, and references to CSV data sources.

    This function updates the provided markdown lines list by appending:
    - The section title and introduction.
    - Plots generated from CSV data sources using provided Python snippets, saved as PNGs.
    - Plot descriptions and narratives.
    - A page break at the end of the section.

    Args:
        md_lines (list[any]): List of markdown lines to which the section content will be appended.
        section_dict (dict[str, any]): Dictionary containing section metadata, including:
            - 'title' (str): Section title.
            - 'introduction' (str): Introduction text for the section.
            - 'narrative' (str): Main narrative text for the section.
            - 'data_source' (str): Subdirectory name under `output_path` where plots will be saved.
            - 'plots' (list[dict], optional): List of plot dictionaries, each containing:
                - 'id' (str): Unique identifier for the plot.
                - 'data_source' (str): CSV filename containing the source data for the plot.
                - 'python_snippet' (str): Python code snippet to generate the plot; must assign output to `plot_img`.
                - 'description' (str): Description of the plot for the markdown report.
        working_path (str): Path to the directory containing source CSV files.
        output_path (str): Path to the base directory where plot PNGs will be saved.

    Returns:
        list[any]: The updated list of markdown lines including the newly added section content.
    """
    working_path = Path(working_path)
    output_path = Path(output_path)

    md_lines.append(f"## {section_dict['title']}")
    md_lines.append(section_dict["introduction"])
    for plot in section_dict.get("plots", []):
        # load the source data from csv
        source_data_path = working_path / plot["data_source"]
        data = pd.read_csv(source_data_path)
        # generate the plot and save as a variable.
        # llm instructed to assign the plot image to variable plot_img within the python snippet.
        local_vars = {"data": data, "plot_img": None}
        exec(plot["python_snippet"], {}, local_vars)
        plot_img = local_vars["plot_img"]
        #  Save plot as a .png
        filename = f"{plot['id']}.png"
        file_path = output_path / filename

        with open(file_path, "wb") as f:
            f.write(plot_img)
        # reference the .png file in the report
        md_lines.append(f"![{plot['id']}](./{filename})")
        # add plot description
        md_lines.append(f"{plot['description']}\n")

    md_lines.append(section_dict["narrative"])
    md_lines.append('<div style="page-break-after: always;"></div>')

    return md_lines
