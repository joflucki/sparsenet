Optimizing a model
=======================

PyTorch proposes a way to optimize models easily, using loss functions, automatic gradient computation, and optimization algorithms.

Generally speaking, the optimization process follows a few steps:

#. Compute the output of the model (forward pass)
#. Compute the loss
#. Compute the gradients
#. Make on optimization step
#. Repeat

Each step is necessary and can not be omitted. Without the forward pass and the loss, PyTorch's autograd is not able to compute gradients,
and without gradients, the optimization algorithm can not function.

To perform an optimization step, PyTorch uses the ``torch.optim.Optimizer``, which is the parent class to all optimization algorithms.
To optimize a specific set of inputs, you must pass them to an optimizer object.

.. code:: python

    import torch

    # These are the parameters to optimize, the input
    parameters = torch.randn([100, 200])

    # Create an optimizer object, in this case the Stochastic Gradient Descent algorithm
    optimizer = torch.optim.SGD([parameters])

    # Depending on the algorithm, you might want to specify additionl hyperparameters
    optimizer = torch.optim.SGD([parameters], lr=0.0001, momentum=0.005)


Once the optimizer object is created, you can follow the steps described earlier (compute loss, compute the gradients, perform an optimization step).
The optimization step is always a single step of optimization, an will not completely optimize the function. For that, you must do many optimization step.



Optimizing a linear model
---------------------------

Using PyTorch and a linear model of form :math:`y = ax`, this process might look like this:

.. code:: python

    import torch

    # Create dummy data, input of 1 x 20, target of 1 x 10
    input = torch.randn([1, 20]) # or "x"
    target = torch.randn([1, 10]) # or "y"

    # Create model weights, 20 x 10
    weights = torch.randn([20, 10], requires_grad=True) # or "a"

    # Create a criterion, or "loss function"
    criterion = torch.nn.MSELoss()

    # Create an optimizer object, which is linked to the parameters to optimize (weights)
    optimizer = torch.optim.SGD([weights], lr=0.1)

    # Training loop, perfomed 10 times
    for _ in range(10):
        # 0. Clear any pre-existing gradients
        optimizer.zero_grad()

        # 1. Compute the output
        output = torch.matmul(input, weights)

        # 2. Compute the loss
        loss = criterion.forward(output, target)

        # 3. Compute the gradients
        loss.backward()

        # 4. Optimize
        optimizer.step()

        # Print loss for visualization
        print(loss.item())

Here, the tensor to optimize is simply a single tensor, containing the weight of the linear model.
The same 4 steps are present inside the training loop. 

Using some SparseNet components, the training loop might look like this:

.. code:: python

    import torch
    import sparsenet

    # Create dummy data, input of 1 x 20, target of 1 x 10
    input = torch.randn([1, 20])  # or "x"
    target = torch.randn([1, 10])  # or "y"

    # Define training weights
    weights = torch.randn([20, 10], requires_grad=True)  # or "a"

    # Compute lambda
    happy_lambda = sparsenet.happy_lambda.HappyLambdaLinear(
        input,
        distribution_count=1000,
    ).run()

    # Define loss
    criterion = sparsenet.nn.PenalizedL2Loss([weights[1:]], happy_lambda)

    # Define optimizer
    optimizer = sparsenet.linear.ISTA([weights], happy_lambda)


    # Define the closure.
    # The closure is a function used by the optimizer to quickly
    # re-evaluate the loss of the model.
    def closure():
        with torch.no_grad():
            output = torch.matmul(input, weights)
            loss, penalty = criterion(output, target)
            return (loss + penalty).item()


    # Training loop, perfomed 10 times
    for _ in range(10):
        # 0. Clear any pre-existing gradients
        optimizer.zero_grad()

        # 1. Compute the output
        output = torch.matmul(input, weights)

        # 2. Compute the loss
        loss, penalty = criterion(output, target)

        # 3. Compute the gradients
        loss.backward()

        # 4. Optimize
        optimizer.step(closure, (loss + penalty).item())

        # Print loss for visualization
        print((loss + penalty).item())



Optimizing a neural net
-----------------------

In the case of a neural net, the parameters to optimize might be many, and the process differs slightly:

.. code:: python

    import torch

    # Create dummy data
    input = torch.randn(20, requires_grad=True)
    target = torch.randn(10)

    # Create a neural network
    model = torch.nn.Linear(20, 10)

    # Create a criterion, or "loss function"
    criterion = torch.nn.MSELoss()

    # Create an optimizer object, which is linked to the parameters to optimize
    optimizer = torch.optim.SGD(model.parameters())

    # Training loop, perfomed 100 times
    for _ in range(100):
        # Clear any pre-existing gradients
        optimizer.zero_grad()

        # Compute the output
        output = model(input)

        # Compute the loss
        loss = criterion(output, target)

        # Compute the gradients
        loss.backward()

        # Optimize
        optimizer.step()

Again, the 4 basic steps are present in the training loop.

SparseNet additionnal components are PyTorch components, and they integrate well with existing PyTorch programs.
However, their usage differs slightly from a PyTorch standard training loop.

Here is how they might be used:

.. code:: python

    import torch
    import sparsenet


    # Define a neural net using SparseNet components
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = torch.nn.Linear(20, 5)
            self.fc2 = sparsenet.nn.NormalizedLinear(5, 10) # Normalized neurons from SparseNet

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            a = sparsenet.nn.sigma(self.fc1(input)) # Sigma activation function from SparseNet
            b = self.fc2(a)
            return b

        # Define a helper method to access parameter groups
        def param_groups(self):
            return [
                {
                    "params": [self.fc1.weight, self.fc1.bias],
                    "penalized": True,  # Used by ISTA and PenalizedL2Loss
                },
                {
                    "params": [self.fc2.weight, self.fc2.bias],
                    "penalized": False,  # Used by ISTA and PenalizedL2Loss
                },
            ]


    # Create dummy data
    input = torch.randn([1, 20])
    target = torch.randn([1, 10])

    # Create model
    model = Net()

    # Compute Happy Lambda
    hl = sparsenet.happy_lambda.HappyLambdaMLP(input, 1_000, [5, 10]).run()

    # Create a criterion, or "loss function"
    criterion = sparsenet.nn.PenalizedL2Loss(
        model.param_groups()[0]["params"],  # Indicate "penalized" parameters
        hl,  # Indicate optimal regularization parameter
    )

    # Create an optimizer object, which is linked to the parameters to optimize
    optimizer = sparsenet.nn.ISTA(model.param_groups(), hl)


    # Define the closure.
    # The closure is a function used by the optimizer to quickly
    # re-evaluate the loss of the model.
    def closure():
        with torch.no_grad():
            output = model.forward(input)
            loss, penalty = criterion.forward(output, target)
            return (loss + penalty).item()


    # Training loop, perfomed 10 times
    for _ in range(10):
        # 0. Clear any pre-existing gradients
        optimizer.zero_grad()

        # 1. Compute the output
        output = model.forward(input)

        # 2. Compute the loss
        loss, penalty = criterion.forward(output, target)

        # 3. Compute the gradients
        loss.backward()

        # 4. Optimize
        optimizer.step(closure, (loss + penalty).item())

        # Print loss for visualization
        print(loss.item())

In this example, we define a class to shape our model. In the case of SparseNet, it is often useful to have a class and to add helper methods to define
which parameters are penalized or not.

Additionnally we compute the optimal regularization parameter using the ``HappyLambdaMLP`` class. This value is then used
with the ``ISTA`` optimizer and ``PenalizedL2Loss``.

An extra function was also defined, the ``closure`` function, wich is a a function used to reevaluate the loss of the model.
Some PyTorch optimizer require such a function to make multiple evaluations in their optimization step, which is the case of the 
``ISTA`` optimizer.

Then, there is the training loop. The traning loop in itself is similar, but the loss computation, and ``step`` call are different.

The loss is computed in two parts. The L2 loss itself, and the L1 penalty.
This is the case to allow partial differentiation, which is shown later in the training loop, where only the L2 loss is differentiated.

For the ``step`` method, two additionnal arguments are added. The ``closure`` function, which the optimizer needs to reevaluate the model, and the
``initial_loss`` parameter. This parameter is used to indicate the starting loss, or initial loss, before making the optimization step. This is useful
to avoid recomputing the loss at the beginning of each optimization step, but the parameter is optional and will be recomputed if needed.