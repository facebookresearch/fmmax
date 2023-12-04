"""Generates the API markdown files from the in-repo docstrings.

Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import ast
import glob
import os
from typing import Any, Dict

from docstring_parser import parse


def write_to_file(output_dir: str, filename: str, data: str, mode: str = "w") -> None:
    """
    Writes data to a file.
    Args:
        output_dir (str): The directory to write the file to.
        filename (str): The name of the file.
        data (str): The data to write to the file.
        mode (str, optional): The mode in which to open the file. Defaults to "w".
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), mode) as f:
        f.write(data)


def process_file(output_dir: str, filename: str, toc: Dict[str, Any]) -> None:
    """
    Processes a file and updates the table of contents.
    Args:
        output_dir (str): The directory to write the output to.
        filename (str): The name of the file to process.
        toc (Dict[str, Any]): The table of contents to update.
    """
    with open(filename, "r") as f:
        module = ast.parse(f.read())
    submodule_name = os.path.splitext(os.path.basename(filename))[0]
    toc[submodule_name] = {}
    for node in module.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            process_class(output_dir, node, toc[submodule_name], submodule_name)
        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            process_function(
                output_dir, node, toc[submodule_name], submodule_name, "def"
            )


def write_header(header: str) -> str:
    """
    Writes a header.
    Args:
        header (str): The header to write.
    Returns:
        str: The written header.
    """
    return f"""---
id: {header}
---

    """


def process_class(
    output_dir: str, node: ast.AST, toc: Dict[str, Any], submodule_name: str
) -> None:
    """
    Processes a class node.
    Args:
        output_dir (str): The directory to write the output to.
        node (ast.AST): The node to process.
        toc (Dict[str, Any]): The table of contents to update.
        submodule_name (str): The name of the submodule.
    """
    docstring = ast.get_docstring(node)
    init_docstring = ""
    init_args = ""
    for sub_node in node.body:
        if isinstance(sub_node, ast.FunctionDef) and sub_node.name == "__init__":
            init_docstring = ast.get_docstring(sub_node)
            init_args = ", ".join(
                [arg.arg for arg in sub_node.args.args if arg.arg != "self"]
            )
            break
    if docstring or init_docstring:
        filename = f"{submodule_name}.{node.name}.md"
        markdown = (
            docstring_to_markdown(docstring)
            + "\n"
            + docstring_to_markdown(init_docstring)
        )
        class_id = f"{submodule_name}.{node.name}"
        write_to_file(
            output_dir,
            filename,
            write_header(class_id)
            + f"### `Class {class_id}({init_args}):`\n{markdown}",
        )
        toc[node.name] = {
            "__doc__": {"link": filename, "desc": get_first_line(docstring)}
        }

    for sub_node in node.body:
        if isinstance(sub_node, ast.FunctionDef) and not sub_node.name.startswith("_"):
            process_function(
                output_dir,
                sub_node,
                toc[node.name],
                f"{submodule_name}.{node.name}",
                "Method",
            )


def process_function(
    output_dir: str,
    node: ast.AST,
    toc: Dict[str, Any],
    parent_name: str,
    is_method: bool = False,
) -> None:
    """
    Processes a function node.
    Args:
        output_dir (str): The directory to write the output to.
        node (ast.AST): The node to process.
        toc (Dict[str, Any]): The table of contents to update.
        parent_name (str): The name of the parent.
        is_method (bool, optional): Whether the node is a method. Defaults to False.
    """
    docstring = ast.get_docstring(node)
    if docstring:
        filename = (
            f"{parent_name}.{node.name}.md"
            if "." not in parent_name
            else f'{parent_name.split(".")[-1]}.md'
        )
        markdown = docstring_to_markdown(docstring)

        with open(os.path.join(output_dir, filename), "a") as f:
            if is_method:
                f.write(write_header(f"{parent_name}.{node.name}"))
            f.write(f"\n### `{parent_name}.{node.name}`\n")
            f.write(markdown)
        toc[node.name] = {"link": filename, "desc": get_first_line(docstring)}


def get_first_line(docstring: str) -> str:
    """
    Gets the first line of a docstring.
    Args:
        docstring (str): The docstring to process.
    Returns:
        str: The first line of the docstring.
    """
    return docstring.split("\n")[0]


def docstring_to_markdown(docstring: str) -> str:
    """
    Converts a docstring to markdown.
    Args:
        docstring (str): The docstring to convert.
    Returns:
        str: The converted markdown.
    """
    if docstring is None:
        return ""

    doc = parse(docstring)
    if doc.short_description is None:
        return docstring

    markdown = doc.short_description + "\n\n"

    if doc.long_description:
        markdown += doc.long_description + "\n"

    if doc.params:
        markdown += "\n#### Args:\n"
        for param in doc.params:
            # TODO (smartalecH) eventually we want to include type information
            # too... it's a bit hairy rn since ast is *supposed* to allow this
            # now (and typed_ast is deprecated) but there isn't a ton of
            # documentation yet...
            markdown += f"- **{param.arg_name}**: {param.description}\n"

    if doc.returns:
        markdown += "\n#### Returns:\n"
        markdown += f"- **{doc.returns.type_name}**: {doc.returns.description}\n"
    return markdown


def traverse_directory(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """
    Traverses a directory and processes all Python files.
    Args:
        input_dir (str): The directory to traverse.
        output_dir (str): The directory to write the output to.
    Returns:
        Dict[str, Any]: The table of contents.
    """
    # Check if the output dir exists, create it if it doesn't.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Remove any existing doc files.
    files = glob.glob(os.path.join(output_dir, "*.md"))
    for f in files:
        os.remove(f)

    # Walk through the file tree and process each file.
    toc = {}
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".py"):
                process_file(output_dir, os.path.join(root, file), toc)
    return toc


def write_toc(output_dir: str, toc: Dict[str, Any]) -> None:
    """
    Writes the table of contents to a file.
    Args:
        output_dir (str): The directory to write the file to.
        toc (Dict[str, Any]): The table of contents to write.
    """
    # ... rest of the function ...
    with open(os.path.join(output_dir, "TOC.md"), "w") as f:
        f.write("# API Reference\n")
        for submodule_name, classes in sorted(toc.items()):
            for class_name, functions in sorted(classes.items()):
                if class_name == "__doc__":
                    continue
                else:
                    f.write(f"\n## {submodule_name}.{class_name}\n")
                    f.write(
                        f'- [{class_name}]({functions["__doc__"]["link"]}): {functions["__doc__"]["desc"]}\n'
                    )
                    for function_name, function in sorted(functions.items()):
                        if function_name != "__doc__":
                            f.write(
                                f'  - [{function_name}]({function["link"]}): {function["desc"]}\n'
                            )


if __name__ == "__main__":
    input_dir = "../src/fmmax"
    output_dir = "./docs/API"
    toc = traverse_directory(input_dir, output_dir)

    # The TOC output is currently broken and not really needed for now...
    # write_toc(output_dir, toc)
