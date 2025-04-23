"""Tests for checking the docstrings of functions and classes.

Header info specific to ESP
"""

from tests.utils.check_docstrings import check_docstrings


# @pytest.mark.skip(reason="Skip this test for now, some issues with .venv files")
def test_docstrings_exist(
    base_folder: str = "beans_zero", folders_to_check: list = None
) -> None:
    """Check that all class and functions contain a docstring.
    Numpy-style is used.

    Arguments
    ---------
    base_folder: str, optional
        Path to the base folder where this function is executed.
    folders_to_check: list[str], optional
        Folders name that must be checked, by default, all of them.

    """
    assert check_docstrings(base_folder, folders_to_check)
