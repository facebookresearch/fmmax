"""Generates the tutorial markdown files from the in-repo docstrings.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import os

import nbformat
from nbconvert import MarkdownExporter


def export_notebooks(notebook_dir: str, output_dir: str = None) -> None:
    """
    Scans a directory for IPython notebooks and exports them to markdown files.

    Args:
        notebook_dir (str): The path to the directory containing the notebooks.
        output_dir (str, optional): The path to the directory where the markdown files should be saved. If not specified, the markdown files will be saved in the same directory as the original notebooks.
    """
    # Check if the output directory exists
    if output_dir and not os.path.exists(output_dir):
        # Create the output directory
        os.makedirs(output_dir)

    # Loop through all files in the directory
    for filename in os.listdir(notebook_dir):
        # Check if the file is an IPython notebook
        if filename.endswith(".ipynb"):
            # Load the notebook
            with open(os.path.join(notebook_dir, filename)) as f:
                notebook = nbformat.read(f, as_version=4)

            # Create a Markdown exporter
            exporter = MarkdownExporter()

            # Export the notebook to markdown
            body, resources = exporter.from_notebook_node(notebook)

            # Save the markdown to a file
            output_filename = filename[:-6] + ".md"
            output_filepath = os.path.join(output_dir or notebook_dir, output_filename)
            print(output_filepath)
            with open(output_filepath, "w") as f:
                f.write(body)

            # Save the images to the output directory
            for image_filename, image_data in resources["outputs"].items():
                with open(
                    os.path.join(output_dir or notebook_dir, image_filename), "wb"
                ) as f:
                    f.write(image_data)


if __name__ == "__main__":
    # Set the path to the directory containing the notebooks
    notebook_dir = "../notebooks"

    # Set the path to the output directory (optional)
    output_dir = "./docs/Tutorials"

    # Call the function to export the notebooks
    export_notebooks(notebook_dir, output_dir)
