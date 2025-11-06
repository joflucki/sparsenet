Developer guide
=====================================

The developer guide is aimed at developers and software engineers working on improving and extending the capabilites of the SparseNet package.
It contains explanation of the file structure, the package setup, and more.

.. toctree::
   :maxdepth: 1

   File structure <file_structure>
   Python package <python_package>
   HappyLambda specification <happy_lambda>
   Building the documentation <build_doc>

To start working on the SparseNet package, you must first install the SparseNet repository, and the dependencies specific the development.

Python dependencies specific to development are defined in the ``pyproject.toml`` as "dev" dependencies, in the optional dependecies section.

To install thoses dependecies, you can install the package with the following command, run from the repository root folder:

.. code:: bash

   pip install .[dev]

For more developer information, please refer to the documents tied with the Bachelor's Thesis