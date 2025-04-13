"""Utility functions for loading Python modules from file paths."""

import sys
from typing import Callable
import importlib.util
from pathlib import Path


def load_module_from_path(file_path: str) -> Callable:
    """Load a Python module from a file path.

    Arguments
    ---------
    file_path : str
        Path to the Python file to load

    Returns
    -------
    module : module
        The loaded Python module

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the specified file is not a Python file.
    ImportError
        If the module cannot be loaded or executed.

    Examples
    --------
    >>> import os; this_dir = os.path.dirname(__file__)
    >>> module = load_module_from_path(f"{this_dir}/config.py")
    >>> isinstance(module.eval_cfg, module.EvaluationConfig)
    True
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Module file not found: {file_path}")

    if not file_path.suffix == ".py":
        raise ValueError(f"Expected a Python file (.py), got: {file_path}")

    # Get the module name from the file name (without .py extension)
    module_name = file_path.stem

    # Load the module specification
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module: {file_path}")

    # Create the module from the spec
    module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules
    sys.modules[module_name] = module

    # Execute the module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Error executing module {file_path}: {e}") from e

    return module
