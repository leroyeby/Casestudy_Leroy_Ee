"""
LLM invoking pipeline for analysing processed BMW sales data using Kedro.

This module defines the Kedro pipeline and node functions responsible for preparing LLM prompt, invoking the Google Gemini model, and writing the model output to chosen directory in JSON format.


Pipelines
---------
- `create_invoke_llm_full_pipeline`:
    Builds and returns the full Kedro pipeline. Used in full pipeline.

- `create_invoke_llm_only_pipeline`:
    Builds and returns the full Kedro pipeline. Used on its own.


Functions
---------
- `load_subtables_full`:
    Loads the feature engineered subtables. For use in full pipeline.

- `load_subtables`:
    Loads the feature engineered subtables. For use when only invoking llm.

- `write_prompt`:
    Writes the prompt for downstream calling.

- `call_google_gemini_llm`:
    Invokes the google gemini llm and returns a json readable string.

- `write_llm_response_as_json`:
    Writes the json readable llm response into a json file.

Notes
-----
This pipeline is used to generate structured analytical output (executive summaries, sectioned analysis, recommendations, and plot specifications) from engineered subtables (or cleaned full table).
"""

import json
import logging
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from google import genai
from kedro.pipeline import Pipeline, node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# Pipelines
# ============================


def create_invoke_llm_full_pipeline(
    **kwargs,
) -> Pipeline:
    """Kedro Pipeline created to house nodes that loads subtables for input and invokes the llm for the generation of a structured output. This pipeline is used when invoking the full process of preprocessing into invoking into report generation."""
    return Pipeline(
        [
            node(
                func=load_subtables_full,
                inputs=["params:working_path", "start_invoke_signal"],
                outputs="loaded_subtables",
                name="load_subtables_full",
            ),
            node(
                func=write_prompt,
                inputs="loaded_subtables",
                outputs="prompt_from_csv",
                name="write_prompt",
            ),
            node(
                func=call_google_gemini_llm,
                inputs="prompt_from_csv",
                outputs="llm_response_json_readable",
                name="call_google_gemini_llm",
            ),
            node(
                func=write_llm_response_as_json,
                inputs=["llm_response_json_readable", "params:working_path"],
                outputs="start_report_gen_signal",
                name="write_llm_response_as_json",
            ),
        ]
    )


def create_invoke_llm_only_pipeline(
    **kwargs,
) -> Pipeline:
    """Kedro Pipeline created to house nodes that loads subtables for input and invokes the llm for the generation of a structured output. This pipeline is used when invoking llm by itself. It presumes preprocessing pipeline has already been executed."""
    return Pipeline(
        [
            node(
                func=load_subtables,
                inputs="params:working_path",
                outputs="loaded_subtables",
                name="load_subtables",
            ),
            node(
                func=write_prompt,
                inputs="loaded_subtables",
                outputs="prompt_from_csv",
                name="write_prompt",
            ),
            node(
                func=call_google_gemini_llm,
                inputs="prompt_from_csv",
                outputs="llm_response_json_readable",
                name="call_google_gemini_llm",
            ),
            node(
                func=write_llm_response_as_json,
                inputs=["llm_response_json_readable", "params:working_path"],
                outputs="start_report_gen_signal",
                name="write_llm_response_as_json",
            ),
        ]
    )


# ============================
# Nodes
# ============================


def load_subtables_full(working_path: str, start_invoke_signal: str) -> dict:
    """
    Load all CSV subtables from a directory, convert them into JSON-serializable
    dictionaries, and assemble a formatted analysis prompt embedding the data.

    Args:
        working_path (str): Path to the directory containing CSV subtables to load.

    Raises:
        FileNotFoundError: If no csv (.csv) file is found in the specified folder.

    Returns:
        dict: JSON of data loaded from the csv.
    """
    if start_invoke_signal != "ready to invoke":
        raise AssertionError(
            f"start_invoke_signal variable is supposed to read 'ready to invoke'. Received signal '{start_invoke_signal}' instead."
        )

    working_path = Path(working_path)

    csv_files = list(working_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No csv file found in the working data folder {str(working_path)}. Please run preprocessing pipeline before proceeding."
        )

    json_dict = {}
    for file in csv_files:
        df = pd.read_csv(file)
        json_dict[file.name] = df.to_dict(orient="records")

    return json_dict


def load_subtables(working_path: str) -> dict:
    """
    Load all CSV subtables from a directory, convert them into JSON-serializable
    dictionaries, and assemble a formatted analysis prompt embedding the data.

    Args:
        working_path (str): Path to the directory containing CSV subtables to load.

    Raises:
        FileNotFoundError: If no csv (.csv) file is found in the specified folder.

    Returns:
        dict: JSON of data loaded from the csv.
    """
    working_path = Path(working_path)

    csv_files = list(working_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No csv file found in the working data folder {str(working_path)}. Please run preprocessing pipeline before proceeding."
        )

    json_dict = {}
    for file in csv_files:
        df = pd.read_csv(file)
        json_dict[file.name] = df.to_dict(orient="records")

    return json_dict


def write_prompt(json_dict: dict) -> str:
    """
    Assemble a formatted analysis prompt embedding the data.

    Args:
        json_dict (dict): JSON of data loaded from the csv.

    Returns:
        str: Formatted prompt with data embedded within.
    """
    prompt = f"""
    You are an expert business analyst and data storyteller. You will receive 
    aggregate tables and sample rows from a feature-engineered BMW sales dataset.

    Your task is to generate:
    1) an executive summary, 
    2) structured analysis sections, 
    3) business recommendations,
    4) JSON-only output following the schema.

    Your analysis must use ONLY the data provided in the JSON below 
    (do not infer or hallucinate new columns).

    ======================
    ANALYSIS REQUIREMENTS
    ======================

    For each section below, include:
    - clear, concise business narrative
    - 1–3 key insights
    - plot specifications (using matplotlib Python snippets)

    SECTION 1 — Sales Performance Trends
    - Identify and describe overall sales volume and revenue trends over time.
    - Include two line plots:
    • Sales Volume over time  
    • Revenue over time  
    - Use the Year column (or equivalent) as the x-axis.

    SECTION 2 — Top/Bottom Model Performance
    - Identify 3 top-performing and 3 underperforming models by:
    • sales volume
    • revenue
    - Include bar plots comparing top vs. underperformers.

    SECTION 3 — Regional Performance
    - Identify 3 top and 3 bottom regions using the same logic as Section 2.
    - Include bar plots.

    SECTION 4 — Key Sales Drivers
    - Identify up to 3 highest-impact sales drivers 
    (e.g., price tier, market segment, model type).
    - Use the aggregates only.
    - Include bar plots for each driver.

    ======================
    OUTPUT FORMAT RULES
    ======================

    Return **exactly one JSON object** with this schema:

    {{
    "executive_summary": string,
    "sections": [
        {{
        "title": string,
        "introduction": string,
        "plots": [
            {{
            "data_source": string,
            "id": string,
            "description": string,
            "python_snippet": string
            }}
        ],
        "narrative": string,
        }}
    ],
    "recommendations": [string]
    }}

    Strict rules:
    - Output ONLY valid JSON. No text before or after.
    - Do NOT guess column names — use exactly what the JSON provides.
    - "python_snippet" must contain valid, minimal matplotlib code loading from a DataFrame named df. 
    - The last part of each "python_snippet" must be "\nbuf = io.BytesIO()\nplt.savefig(buf, format='png')\nplt.close()\nplot_img = buf.getvalue()\n".
    - df = pd.DataFrame(data) must be one of the lines in the "python_snippet"
    - Assume the data_source table information is in a dataframe df.
    - The y-axis of all plots must start at 0.
    - Keep narrative concise, business-focused, and non-technical.

    ======================
    DATA INPUT
    ======================
    {json_dict}
    """

    logger.info(f"prompt: \n{prompt}")

    return prompt


def call_google_gemini_llm(prompt: str) -> str:
    """
    Send a text prompt to a Google Gemini LLM and return the parsed JSON response.

    This function loads API credentials from environment variables, initializes
    the Gemini client, sends the prompt to the specified model, logs the raw response, and returns it as a JSON readable string.

    Args:
        prompt (str): The text prompt to send to the LLM.

    Returns:
        str: The LLM's JSON readable response text.

    Raises:
        KeyError: If required environment variables (GEMINI_API_KEY or GEMINI_MODEL) are missing.
        genai.error.GenAIError: If the Gemini API call fails for any reason.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    response_text = response.text
    logger.info(f"{model} response: {response_text}")

    # remove json code fence typical of gemini models
    # ```json ```
    # response_json_readable_text = response_text[8:-4]
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    response_json_readable_text = text.strip()

    return response_json_readable_text


def write_llm_response_as_json(llm_response: str, working_path: str) -> str:
    """
    Parse a JSON-formatted LLM response string and write it to a file in the specified directory.

    Args:
        llm_response (str): A string containing a JSON-formatted response from an LLM.
        working_path (str): The directory path where the JSON file ("response.json") will be saved.

    Returns:
        str: String signalling the generate report pipeline can start.

    Raises:
        json.JSONDecodeError: If `llm_response` is not a valid JSON string.
    """
    parsed_response_dict = json.loads(llm_response)

    working_path = Path(working_path)
    json_filepath = working_path / "response.json"

    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(parsed_response_dict, f, indent=4)

    return "ready to generate report"
