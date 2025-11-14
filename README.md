# SparseNet — Efficient sparse models with PyTorch :runner:

SparseNet is a compact library built on PyTorch designed to facilitate the training of sparse models using algorithms such as ISTA. Developed in the context of research at the University of Geneva's Section of Mathematics, this library was implemented by Jonas Flückiger from the HE-Arc University of Applied Sciences as part of his bachelor's thesis.

SparseNet offers tools for creating sparse datasets, training linear models and neural networks, and provides numerous examples and use cases. These tools are available as a Python package, while the examples and scripts are hosted in the Git repository.

This documentation includes instructions for installing and using the SparseNet library, as well as comprehensive API documentation for developers. Each module is described in detail within the API documentation, with additional information on the implementation of specific algorithms where necessary.

## Getting started

### Downloading the repo


You can download the SparseNet repository and place it in the directory of your choice.

```
git clone https://github.com/joflucki/sparsenet
```

Once you have downloaded the repository, move the the root directory of the projet.

### Setting up a virtual environment

It is highly recommended to use a virtual environment to manage your dependencies. To set up a virtual environment, navigate to your project root directory and create a virtual environment using ``venv``:

```bash
python -m venv venv
```

Activate the virtual environment and update pip:

- On Windows:

    ```bash
    venv\Scripts\activate
    python -m pip install --upgrade pip
    ```

- On macOS and Linux:

    ```bash
    source venv/bin/activate
    python3 -m pip install --upgrade pip
    ```

Once the virtual environment is activated, you should see `(venv)` at the beginning of your command prompt.

Next, install the SparseNet package and its required dependencies using `pip`:

```bash
pip install .
```
This will install SparseNet along with all its dependencies.

### Running an example

With the virtual environment set up and dependencies installed, you can now start using the SparseNet package.
To use an example, simply use the command line to launch a script from the root of the repository.

- On Windows:

    ```bash
    python examples\linear\training.py
    ```


- On macOS and Linux:

    ```bash
    python3 examples/linear/training.py
    ```

### Importing and Using the Package

You can now start using the SparseNet package. To import SparseNet in your Python script, simply use:

```python
import sparsenet
```
You can then start using the various tools and functions provided by SparseNet. For example, to create an MLP simulation:

```python
from sparsenet.nn import Simulation
simulation = Simulation(parameters)
```

Refer to the User Guide for detailed examples and use cases.

### Setting Up an IDE (Optional)

While setting up an IDE is optional, it can greatly enhance your development experience. Here are some steps to set up a common IDE, Visual Studio Code (VS Code), for working with SparseNet:

1. **Open VS Code**: Launch Visual Studio Code.

2. **Open the Project Folder**: Go to `File -> Open Folder` and select your project directory.

3. **Select the Python Interpreter**: Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) to open the command palette. Type `Python: Select Interpreter` and choose the interpreter located in your virtual environment (`.venv/bin/python` or `venv\Scripts\python.exe`).

4. **Install Python Extension**: If you haven't already, install the Python extension for VS Code, which provides rich support for Python. This includes the "Python" extension, but also the "Pylance" and "Python Debugger" extensions.

5. **Set Up Debugging Configuration**: You can set up a debugging configuration to run your scripts directly from VS Code. Go to the Debug view, click on the gear icon to open the `launch.json` file, and add a configuration for Python.

6. **Enable Black Formatting**: SparseNet uses `Black` for code formatting. To enable Black in VS Code, you can install the Black extension.

7. **Enable Strict Type Checking**: For better code quality, enable strict type checking. To do so, open VS Code settings with `CTRL+,` and search `Python type checking mode`. In the first result, chose a type checking mode. We recommend `standard` or `strict`.

By following these additional steps, you'll ensure consistent code formatting and improve type checking in your SparseNet project.

## License

The software was developed by Jonas Flückiger (HE-Arc), in collaboration with Cédric Bilat (HE-Arc) and Sylvain Sardy (UniGE), all rights reserved.
