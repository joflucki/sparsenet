File structure
===============

The SparseNet repository is set up with many folders.
Here is an overview of the top level folders and files:

.. code::

    sparsenet/
    ├─ benchmarks/
    ├─ data/
    ├─ docs/
    ├─ examples/
    ├─ src/sparsenet/
    ├─ tests/
    ├─ .gitignore
    ├─ LICENSE
    ├─ README.md
    ├─ pyproject.toml
    ├─ setup.py


* ``benchmarks``: Runnable scripts, more specifically benchmarking scripts, that record the performance of the code, display graphs, etc.
* ``data``: Multiple datasets for training.
* ``docs``: The package's documentation.
* ``examples``: Multiple example usage of the SparseNet package.
* ``src/sparsenet``: The source code of the SparseNet package.
* ``tests``: Multiple tests for the SparseNet package.
* ``.gitignore``: The list of files and folders ignored by Git.
* ``LICENSE``: The selected license for the SparseNet repository.
* ``README.md``: The README file for the SparseNet repository.
* ``pyproject.toml``: The main configuration file for the package. Contains the package name, author, dependencies, etc.
* ``setup.py``: The configuration file for Setuptools. Currently empty.

The ``data`` folder
---------------------

The ``data`` folder is used to store datasets and various data useful for examples and tests.

.. code::

    data/
    ├─ simulations/
    ├─ unige/

* ``simulations``: The datasets generated via simulation.
* ``unige``: The dataset provided by the UniGE.

The ``docs`` folder
---------------------

The ``docs`` folder contains all the documentation, including the user guide, developer guide and API documentation.

It is written using ReStructedText, and built using the Sphinx tool.

.. code::

    docs/
    ├─ developer_guide/
    ├─ installation/
    ├─ sparsenet/
    ├─ user_guide/
    ├─ Makefile
    ├─ conf.py
    ├─ index.rst
    ├─ make.bat

* ``developer_guide``: The pages of the developer guide.
* ``installation``: The pages of the installation guide.
* ``sparsenet``: The API Documentation.
* ``user_guide``: The pages of the user guide.
* ``Makefile``: Makefile for Sphinx build.
* ``conf.py``: Configuration file for Sphinx.
* ``make.bat``: Build script for Sphinx.


The ``examples`` folder
---------------------

The ``examples`` folder contains many code examples, usign the various tools of the SparseNet package.

.. code::

    docs/
    ├─ linear/
    ├─ nn/

* ``linear``: Examples for linear models.
* ``nn``: Examples for neural networks.

The ``src/sparsenet`` folder
---------------------

The ``src/sparsenet`` folder contains the source code of the SparseNet package.

.. code::

    src/sparsenet/
    ├─ happy_lambda/
    ├─ linear/
    ├─ nn/
    ├─ __init__.py

* ``happy_lambda``: Contains the specifications and implementation of the "happy lambda" algorithms.
* ``linear``: Source code for tools and componenets useful for linear models.
* ``nn``: Source code for tools and components useful for neural networks.
* ``__init__.py``: Package initialization file.

The ``tests`` folder
---------------------

The ``tests`` folder contains unit tests for the SparseNet package.

.. code::

    src/sparsenet/
    ├─ happy_lambda/
    ├─ linear/
    ├─ nn/
    ├─ __init__.py

* ``happy_lambda``: Unit tests for the ``happy_lambda`` module.
* ``linear``: Unit tests for the ``linear`` module.
* ``nn``: Unit tests for the ``nn`` module.
* ``__init__.py``: Package initialization file.