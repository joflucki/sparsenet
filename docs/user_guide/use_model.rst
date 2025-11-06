Using a model
=======================

Once a model is created, it is possible to use it on input data to produce an output.
The method used to create an output is different between linear models and neural nets.

Using a linear model
--------------------

Because a linear model is a simple matrix operations, we can easily compute the output using PyTorch's tensor operations.

.. code-block:: python

    output = torch.matmul(input, weight)

Depending on the form of the linear model,
it is possible to add a bias or error component using ``torch.add()`` or more simply the ``+`` operator between tensors.


Using a neural net
------------------

With neural nets, once the class is defined and an instance is created, you can simply
call the ``forward`` method on the input data, or call the object directly.

.. code-block:: python

    output = model.forward(input)
    # or
    output = model(input)

The two instructions are equivalent, though the ``forward`` method will provide better type hints.