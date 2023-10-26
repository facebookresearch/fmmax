import os
from typing import List, Tuple
from pydocstring import parse
def generate_markdown_files(src_dir: str, output_dir: str):
    """Generate markdown files for a Docusaurus site based on the docstrings of a Python library."""
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Traverse the source directory and extract the docstrings of functions and classes
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".py"):
                # Parse the module and extract its functions and classes
                with open(os.path.join(root, file), "r") as f:
                    module = ast.parse(f.read())
                    functions_and_classes = []
                    for node in module.body:
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            functions_and_classes.append((node.name, node))
                
                # Generate a markdown file for each class and group of functions
                with open(os.path.join(output_dir, f"{file[:-3]}.md"), "w") as f:
                    f.write("# " + file[:-3] + "\n\n")
                    for name, obj in functions_and_classes:
                        if not name.startswith("_"):
                            docstring = parse(obj).to_markdown()
                            if isinstance(obj, ast.ClassDef):
                                # Write the class docstring to the file
                                f.write("## " + name + "\n\n" + docstring + "\n\n")
                            else:
                                # Write the function docstring to the file
                                f.write("### " + name + "\n\n" + docstring + "\n\n")

if __name__ == "__main__":
    src_dir = "../fmmax/"  # Replace with the name of your Python library
    output_dir = "docs/API"  # Replace with the path to your Docusaurus site's API docs directory
    generate_markdown_files(src_dir, output_dir)