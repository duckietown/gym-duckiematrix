"""Sphinx configuration for the Gym-Duckiematrix documentation."""

import sys
from pathlib import Path

# -- Project information --

project = "Duckietown Pondcleaner"
copyright = "Duckietown"
author = "Duckietown"

# -- General configuration --

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

sys.path.insert(0, str(Path.cwd()))

# Add any paths that contain templates here, relative to this directory.
templates_path: list[str] = []
html_static_path: list[str] = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []

# -- Options for HTML output --

html_theme = "sphinx_rtd_theme"
