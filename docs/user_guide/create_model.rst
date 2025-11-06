Creating a model
==================

Creating a linear model
---------------------------

For linear models, the model can be represented using tensors. For example, a 2D tensors for weights.
You can initialize these tensors with values from any desired distribution, such as a normal distribution.

.. code-block:: python

    import torch

    N = 500
    weight = torch.randn([N, 1])

Or fill it with a specific value:

.. code-block:: python

    import torch

    N = 500
    weight = torch.full([N, 1], 10) # Fill with value '10'


Creating an MLP
---------------------------

To create a functional neural net using PyTorch,
you must define and initialize the neural net,
and specify how data will pass through your model.

Defining the neural net
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import sparsenet


    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(501, 20)
            self.fc2 = sparsenet.nn.NormalizedLinear(20, 1)

Here, a class named ``Net`` is created (the name does not matter), and in its contructor (the ``__init__`` method), two layers of neurons are defined.
The first layer is a  linear layer from PyTorch, with 501 incoming features, and 20 outgoing features.
This means the input of the model has 501 features, and that the first layer is composed of 20 neurons.

Then, a second layer is defined. This layer is a normalized linear layer, where the weights of the neurons are normalized.
This layer has 20 incoming features (the 20 neurons from the 1st layer) and 1 outgoing feature, the final scalar value.

Specifying how data passes through
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the shape of the neural net is defined, it is necessary to indicate how data will pass through,
by implementing the ``forward`` method inside the model. This method will be called when performing forward passes with the model.

.. code-block:: python

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            a = sparsenet.nn.sigma(self.fc1(input))
            b = self.fc2(a)
            return b

Here, the input tensor is passed through the first layer of the model with ``self.fc1(input)``.
Then, the result is passed through the ``sigma`` activation function.

Eventually, the data is passed through the second layer, and returned.


Using ``torch.nn.Sequential``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch includes a simpler way to define a neural net without the use of Python classes.

The ``torch.nn.Sequential`` is a series of `sequential` operations that are applied in series to input data.

Using the same architecture as before, a ``torch.nn.Sequential`` version would be:

.. code:: python 

    model = torch.nn.Sequential(
        torch.nn.Linear(501, 20),
        sparsenet.nn.Sigma(),
        sparsenet.nn.NormalizedLinear(20, 1),
    )





Specifying which parameters are penalized
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the case of sparse models, it is common to apply a penalty to some of the parameters of the neural net.
To indicate which parameters are penalized or not, we can use parameter groups.

Parameter groups are a way to group different parameters of a neural net, and attach metadata to these groups.
For example, if you want to use different learning rates for different parameters in your neural net,
you might use parameter groups.

In the case of sparse models, parameters groups can be used as a convenient way to indicate wich parameters are penalized.
For example, one might implement a method in the neural net that returns the different parameter groups:

.. code-block:: python

        def param_groups(self):
            return [
                {
                    "params": [self.fc1.weight, self.fc1.bias],
                    "penalized": True,
                },
                {
                    "params": [self.fc2.weight, self.fc2.bias],
                    "penalized": False,
                },
            ]

Or one might define an object outside of the neural net:

.. code-block:: python

    model = Net()
    params_groups = [
        {
            "params": [model.fc1.weight, model.fc1.bias],
            "penalized": True,
        },
        {
            "params": [model.fc2.weight, model.fc2.bias],
            "penalized": False,
        },
    ]


In its entirety, a neural net might look like this:

.. code-block:: python

    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(501, 20)
            self.fc2 = sparsenet.nn.NormalizedLinear(20, 1)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            a = sparsenet.nn.sigma(self.fc1(input))
            b = self.fc2(a)
            return b

        def param_groups(self):
            return [
                {
                    "params": [self.fc1.weight, self.fc1.bias],
                    "penalized": True,
                },
                {
                    "params": [self.fc2.weight, self.fc2.bias],
                    "penalized": False,
                },
            ]

    model = Net()

It is possible to change the number of neurons in each layer, the type of layer used, and how the data passes through activation functions or not.

There are many activation functions and neuron types available in the PyTorch library, which you can learn more about in their documentation.