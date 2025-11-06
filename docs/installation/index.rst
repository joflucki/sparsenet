Installation
=============

The installation guide contains step-by-step instructions to install and set up a Python program with the SparseNet package.

It is recommendedto have prior knowledge in Python development and virtual environments.
The downloaded SparseNet repository is also required.

Downloading the Repo
--------------------

You can download the SparseNet repository and place in the directory of your choice. For example ``$HOME/sparsenet``.

Once you have downloaded the repository, move to the root of the repository.

Setting Up a Virtual Environment
--------------------------------

It is highly recommended to use a virtual environment to manage your dependencies. To set up a virtual environment, navigate to the your project directory and create a virtual environment using ``venv``:

.. code-block:: bash

   cd $HOME/sparsenet
   python -m venv venv

Activate the virtual environment and update pip:

- On Windows:

  .. code-block:: bash

    venv\Scripts\activate
    python -m pip install --upgrade pip

- On macOS and Linux:

  .. code-block:: bash

    source venv/bin/activate
    python3 -m pip install --upgrade pip

Once the virtual environment is activated, you should see ``(venv)`` at the beginning of your command prompt.

Next, install the SparseNet package and its required dependencies using ``pip``:

.. code-block:: bash

   pip install .

This will install SparseNet along with all its dependencies.

If you wish to install the development dependencies (necessary to build the documentation, and to run unit tests),
you can install them using the following command:


.. code-block:: bash

   pip install .[dev]

Running an example
-------------------------------

With the virtual environment set up and dependencies installed, you can now start using the SparseNet package.
To use an example, simply use the command line to launch a script from the root of the repository.

- On Windows:

  .. code-block:: bash

    python examples\linear\training.py

- On macOS and Linux:

  .. code-block:: bash

    python3 examples/linear/training.py


Importing and Using the Package
-------------------------------

You can now start using the SparseNet package. To import SparseNet in your Python scripts, simply use:

.. code-block:: python

   import sparsenet

You can then start using the various tools and functions provided by SparseNet. For example, to create an MLP simulation:

.. code-block:: python

   from sparsenet.nn import Simulation
   simulation = Simulation(parameters)

Refer to the User Guide for detailed examples and use cases.

Setting Up an IDE (Optional)
----------------------------

While setting up an IDE is optional, it can greatly enhance your development experience. Here are some steps to set up a common IDE, Visual Studio Code (VS Code), for working with SparseNet:

1. **Open VS Code**: Launch Visual Studio Code.

2. **Open the Project Folder**: Go to ``File -> Open Folder`` and select your project directory.

3. **Select the Python Interpreter**: Press ``Ctrl+Shift+P`` (or ``Cmd+Shift+P`` on macOS) to open the command palette. Type ``Python: Select Interpreter`` and choose the interpreter located in your virtual environment (``.venv/bin/python`` or ``venv\Scripts\python.exe``).

4. **Install Python Extension**: If you haven't already, install the Python extension for VS Code, which provides rich support for Python. This includes the "Python" extension, but also the "Pylance" and "Python Debugger" extensions.

5. **Set Up Debugging Configuration**: You can set up a debugging configuration to run your scripts directly from VS Code. Go to the Debug view, click on the gear icon to open the ``launch.json`` file, and add a configuration for Python.

6. **Enable Black Formatting**: SparseNet uses `Black` for code formatting. To enable Black in VS Code, you can install the Black extension.

7. **Enable Strict Type Checking**: For better code quality, enable strict type checking. To do so, open VS Code settings with ``CTRL+,`` and search ``Python type checking mode``. In the first result, chose a type checking mode. We recommend ``standard`` or ``strict``.

By following these additional steps, you'll ensure consistent code formatting and improve type checking in your SparseNet project.
