Computing the optimal penalty weight
======================================

When training a sparse model, it is common to apply a penalty on the parameters of the model.
The weight of this penalty can have a great impact on the training of the model.

SparseNet provides a way to compute the optimal penalty weight, a.k.a the "happy lambda", based on the dataset and shape of the model.



Computing the value for linear models
---------------------------------------

.. code-block:: python

    import torch
    from sparsenet.happy_lambda import HappyLambdaLinear

    happy_lambda = HappyLambdaLinear(dataset=dataset, distribution_count=1000).run()
    print(happy_lambda)

In this case, the dataset was generated using a linear simulation, and the distribution count indicates
the number of gaussian distributions to draw when computing the penalty weight. The bigger the number, the more precise the result.



Computing the value for MLPs
---------------------------------------

.. code-block:: python

    import torch
    from sparsenet.happy_lambda import HappyLambdaMLP

    happy_lambda = HappyLambdaMLP(
        dataset=dataset,
        distribution_count=1000,
        neuron_count=[10, 1],
    ).run()
    print(happy_lambda)

In this case, the dataset was generated using an MLP simulation, and the distribution count indicates
the number of gaussian distributions to draw when computing the penalty weight. The bigger the number, the more precise the result.
The neuron count indicates the number of neurons in each layer of the model, and is used to compute the penalty weight.