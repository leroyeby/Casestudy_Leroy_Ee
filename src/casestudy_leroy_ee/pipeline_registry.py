import logging
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from utils.kedro_utils import get_kedro_context

from .pipelines.pipelines_preprocessing import (
    create_preprocessing_full_pipeline,
    create_preprocessing_sqldb_pipeline,
)
from .pipelines.pipelines_invoke_llm import (
    create_invoke_llm_full_pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

context = get_kedro_context()


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    # ----------------------------------
    # Data Preprocessing Pipelines
    # ----------------------------------
    # ----------------------------------
    ### Data Preprocessing Main Pipeline
    # ----------------------------------
    pipelines["data_preprocessing_full"] = create_preprocessing_full_pipeline()

    # ----------------------------------
    ### Data Preprocessing Sub Pipeline
    # ----------------------------------
    pipelines["data_preprocessing_sqldb"] = create_preprocessing_sqldb_pipeline()

    # ----------------------------------
    # Invoke LLM Pipelines
    # ----------------------------------
    # ----------------------------------
    ### Invoke LLM Main Pipeline
    # ----------------------------------
    pipelines["invoke_llm_full"] = create_invoke_llm_full_pipeline()

    # ----------------------------------
    # Full Main Pipeline
    # ----------------------------------
    pipelines["__default__"] = (
        create_preprocessing_full_pipeline()
        + create_invoke_llm_full_pipeline()
        # + create_generate_report_full_pipeline()
    )

    logger.info(pipelines.values())

    return pipelines
