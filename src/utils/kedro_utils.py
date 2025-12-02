"""Generic utilities that will be used throughout our kedro project."""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


def get_kedro_context() -> dict[str, any]:  # pragma: no cover noqa
    """Retrieves the Kedro context object to access project-specific
    configuration, parameters, and settings. This function initializes the
    Kedro environment by bootstrapping the project from the current working
    directory and creates a session to manage the lifecycle of a Kedro run.

    Returns:
        KedroContext (dict[str, any]): An object that provides access to Kedro project
            configurations and data catalog.

    Raises:
        KedroSessionError: If an error occurs during the session creation or context
            loading.
    """
    bootstrap_project(Path.cwd())
    with KedroSession.create() as session:
        context = session.load_context()
        return context
