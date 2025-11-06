Evaluating a model
===================

Once the model is capable of producing an output, it is important to evaluate the quality of this output.
To do so, multiple loss functions are available in the PyTorch library, as well as an additionnal loss function
from the SparseNet package.

The method does not differ between linear models and neural nets, because the operation is carried out on the output
and target tensors, regardless of how they were produced.

Creating a criterion
---------------------

A criterion is an object used to evaluate the quality of an output.
For example, the ``MSELoss`` from PyTorch is a widely used criterion.

.. code-block:: python

    criterion = torch.nn.MSELoss()


In the case of sparse neural nets, you might want to use a criterion that includes a penalty.
For example, the ``PenalizedL2Loss`` applies an additionnal penalty on the L2 Loss.

For this penalty, you have to indicate a penalty weight, and which parameters are penalized.

.. code-block:: python

    criterion = sparsenet.nn.PenalizedL2Loss(
        params=[model.fc1.weight, model.fc1.bias],
        happy_lambda=happy_lambda,
    )

Using the criterion
--------------------

Once the criterion is created, you can use it on the ouptut and target tensors to produce a loss.

Because criterions derive from the ``torch.nn.Module`` class, you can either use the ``forward`` method,
or call the object directly.

.. code-block:: python

    loss = criterion.forward(output, target)
    # or
    loss = criterion(output, target)

In the case of the ``PenalizedL2Loss``, the ``forward`` method returns the loss as two values, the L2 loss and the L1 penalty.
These two components are returned separatly to allow to compute partial derivatives if needed.
In the case where partial derivatives are not needed, you can simply add the two values together to find the final loss value.

For example:

.. code-block:: python

    l2_loss, l1_penalty = criterion.forward(output, target)
    l2_loss.backward() # Compute gradient for L2 loss only

    complete_loss = l2_loss + l1_penalty