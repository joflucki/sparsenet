Simulation
==========

Simulations are used to generate entire datasets, including the input, outputs, weights, and errors. Utilities are also provided to save and load simulation files.

To run a simuation, create a ``Simulation`` instance by supplying the necessary parameters, and run the ``run`` method to run the simulations.
Once you've ran the simulation, all properties will be populated and available, as described in the API documentation.

Example
---------

.. code-block:: python

   device = torch.device("cuda")
   simulation = Simulation(
       sample_count=1_000,
       feature_count=500,
       generator=torch.Generator(device).manual_seed(100),
       device=device,
       non_zero_first=False,
   )
   simulation.run()
   simulation.save("data/simulations/Steve", format="csv")


API Documentation
-----------------

.. automodule:: sparsenet.linear.simulation
   :members:
   :undoc-members:
   :show-inheritance: