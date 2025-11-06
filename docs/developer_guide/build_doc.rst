Building the documentation
===========================

To build the documentation, it is imperative to have all optional dependecies installed.
To install dev dependecies, run the following command at the root of the repository:

.. code:: bash

    pip install .[dev]

It is also required to have Make installed. To install Make on Ubuntu, use the following command:

.. code:: bash

    sudo apt install build-essential

Once the python dependecies and Make are installed you can used Make to build the documentation.

Run the following command in the ``docs`` folder:

.. code:: bash

    make html

This will generate a ``_build/html`` folder, with the generated documentation as HTML files.
You can the open then ``index.html`` file and view the documentation.