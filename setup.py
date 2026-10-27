"""Package configuration for Gym-Duckiematrix."""

import ast
from pathlib import Path

from setuptools import find_packages, setup  # type: ignore[import-untyped]

package_name = "gym-duckiematrix"
library_webpage = "<LIBRARY_WEB_PAGE>"
maintainer = "Liam Paull"
maintainer_email = "liam@duckietown.com"
short_description = "This is the gym environment wrapper for the Duckiematrix"
full_description = """
This is the gym environment wrapper for the Duckiematrix
"""

if "<" in package_name:
    msg = "Please fill in the project data in setup.py."
    raise ValueError(msg)


# Read version from the __init__ file
def get_version_from_source(filename: str) -> str:
    """Read the package version without importing the package."""
    version_value = None
    source_path = Path(filename)
    with source_path.open(encoding="utf-8") as source_file:
        for line in source_file:
            if line.startswith("__version__"):
                assignment = ast.parse(line).body[0]
                if not isinstance(assignment, ast.Assign):
                    continue
                value = assignment.value
                if isinstance(value, ast.Constant) and isinstance(
                    value.value,
                    str,
                ):
                    version_value = value.value
                    break
        else:
            error_message = f"No version found in {filename!r}."
            raise ValueError(error_message)
    if version_value is None:
        raise ValueError(filename)
    return version_value


version = get_version_from_source("src/gym_duckiematrix/__init__.py")

# read project dependencies
# NO - dependencies.txt is for testing dependiences - EVERYTHING PINNED
# The requirements here must be broad.

install_requires = [
    "duckietown-messages>=0.1.0,<0.2",
    "duckietown-sdk>=0.3.0,<0.4",
    "gymnasium>=1.2.0,<2",
    "matplotlib",
    "numpy",
]
# compile description
underline = "=" * (len(package_name) + len(short_description) + 2)
description = f"""
{package_name}: {short_description}
{underline}

{full_description}
"""

console_scripts: list[str] = []

# setup package
setup(
    name=package_name,
    author=maintainer,
    author_email=maintainer_email,
    url=library_webpage,
    python_requires=">=3.12",
    install_requires=install_requires,
    package_dir={"": "src"},
    packages=find_packages("./src"),
    long_description=description,
    version=version,
    entry_points={"console_scripts": console_scripts},
)
