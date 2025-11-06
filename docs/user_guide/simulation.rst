Generating a dataset through simulation
============================================

SparseNet includes a way to run simulations, that will provide a dataset as well as a target to train linear models or MLPs.
These simulations can be saved on disk and reused for later training.

Creating a simulation
----------------------

Linear simulation
~~~~~~~~~~~~~~~~~~~~~~~~

To create a linear simulation, the ``Simulation`` class is available in the ``sparsenet.linear`` package.

.. code-block:: python

    import torch
    from sparsenet.linear import Simulation

    simulation = Simulation(
        sample_count=1_000,
        feature_count=500,
    )
    simulation.run()

The sample count corresponds to the number of samples in the dataset, while the feature count corresponds to the number of features per sample.
By default, the linear simulation will run on the GPU if it is available.

You can also directly provide the device on which to run the simulation, as well as a specific random number generator, to ensure reproducibility.

.. code-block:: python

    import torch
    from sparsenet.linear import Simulation

    device = torch.device("cuda")
    generator = torch.Generator(device).manual_seed(100)
    simulation = Simulation(
        sample_count=1_000,
        feature_count=500,
        generator=generator,
        device=device,
    )
    simulation.run()

This will create a random number generator with a seed of 100, and explicitly runs on the GPU.

MLP simulation
~~~~~~~~~~~~~~~~~~~~~~~~

MLP simulations are similar to linear simulations, with the addition of new parameters.
This time, the class is located in the ``sparsenet.nn`` package.

.. code-block:: python

    import torch
    from sparsenet.nn import Simulation, sigma

    simulation = Simulation(
        sample_count=100,
        feature_count=20,
        neuron_counts=[10, 1],
        activation_fn=sigma,
    )
    simulation.run()

With MLP simulations, you must describe the shape of the neural net by providing the number neurons per layer (``neuron_counts``)
as well as the activation function used in the network (``activation_fn``).

MLP simulations will simulate a "standard" MLP, where weights and biases are applied to each layer, and the activation function is applied the each layer,
except the last one.


Accessing the dataset
------------------------

Once the simulation has been created and run, you can access the different generated values in the simulation.

Linear simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset = simulation.dataset
    weights = simulation.weights
    target = simulation.target
    error = simulation.error

    print(dataset)
    print(weights)
    print(target)
    print(error)


MLP simulation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    dataset = simulation.dataset
    weights = simulation.weights
    biases = simulation.biases
    target = simulation.target
    
    print(dataset)
    print(weights)
    print(biases)
    print(target)

Saving a simulation to disk
---------------------------------

In both cases, linear or MLP, you can save you dataset to a folder using the ``save`` method:

.. code-block:: python

    simulation.save("data/simulations/Steve", format="csv")

This will create a folder in the location ``data/simulations/Steve``, and save the different elements as files.

Loading a simulation
------------------------------------

If you wish to load a simulation that was previously saved on the disk, you can use the static ``load`` method.


Linear simulation
~~~~~~~~~~~~~~~~~~~
.. code-block:: python

    import torch
    from sparsenet.linear import Simulation

    dataset, weights, error, target = Simulation.load("data/simulations/Steve", format="csv")

MLP simulation
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from sparsenet.nn import Simulation
    
    dataset, weights, biases, target = Simulation.load("data/simulations/Steve", format="csv")

For detailed API documentation, see the :doc:`linear simulation API Documentation <../sparsenet/linear/simulation>` or the :doc:`MLP simulation API Documentation <../sparsenet/nn/simulation>`