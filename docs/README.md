# Docs

Here live the source markdown files and generation code for the FMMAX docs.

## Quickstart

To run a docs server locally, simply run the following (in the current directory):

```bash
$ make install
$ make all
```

(Note the above assumes you already have `python` and `npm` installed on your system)

The routine will automatically generate the API reference markdown files from the FMMAX source code, and the Tutorials markdown files from the FMMAX notebooks.

To test changes on the fly, you simply need to save the markdown file you are working on and the server should update the page in real time.

To *regenerate* API or tutorial markdown files from their source, simply run `make all` (no need to reinstall).

To clean the docs directory back to the upstream state, simply run `make clean`. This will delete all generated markdown files and the docusaurus dependencies.

## Adding tutorials

To add a new tutorial, simply add a new Jupyter notebook in the `../notebooks` directory. To ensure the output is also displayed in the docs, run the notebook once before committing.

The notebook will automatically be exported as a markdown file as described above.

To add the new tutorial to the Tutorials menu bar, add the name of the markdown file (without the suffix) to the `sidebars.js` file. This allows you to specify the order in which the tutorials appear.