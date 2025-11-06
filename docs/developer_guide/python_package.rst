Python package
===============

The Python package is set up using the rather new ``pyproject.toml`` configuration file.
This file is used by build tools such as ``setuptools`` to gather the necessary informations on the package. 
Things such as the package version, package dependencies, dev dependencies, etc.

SparseNet uses this file along ``setuptools`` to configure and build the package, which is why a ``setup.py`` file is still necessary to configure ``setuptools``.
Currently, the package only contains pure Python code, and so the ``setup.py`` file is empty.

Modules
--------

The SparseNet is composed of sub-modules, such as ``linear`` or ``nn``. These modules are useful to keep the code separated, and makes it easier to maintain.
To create a module, it is important to add a ``__init__.py`` file in the new folder. This file indicates to Python that it contains some Python code that can be imported.

To make a class or method importable directly from the parent module, it is necessary to import the desired component in the ``__init__.py`` file to make it accessible.

For example, let's consider this file structure:

.. code::
    my_module/
    ├─ __init__.py
    ├─ a.py
    ├─ b.py
    ├─ c.py

If the ``__init__.py`` is empty, importing the ``A`` class from the ``a.py`` file would be done like so:

.. code:: python

    from sparsenet.my_module.a import A

To avoid having to specify the name of the file in which the ``A`` class is defined, we can modify the ``__init__.py``:

.. code:: python
    
    from .a import A

This makes the following import possible:

.. code:: python

    from sparsenet.my_module import A

.. warning::
    All internal imports must be done with **RELATIVE** imports. Imports such as ``from .a import A`` or ``from . import a`` will work,
    but imports like ``from sparsenet.my_module.a import A`` will cause issue internally.

You can also define the ``__all__`` variable in the ``__init__.py``. This variable specifies which objects are imported when using a wildcard ``*`` import.