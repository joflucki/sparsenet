Using Tensors
===============================

PyTorch is an open-source deep learning framework that provides a range of tools and libraries for building machine learning models.
At the core of PyTorch is the Tensor, a multi-dimensional array that is similar to NumPy arrays
but with additional capabilities for GPU acceleration and automatic differentiation.

Basic Principles
----------------

A Tensor is the fundamental building block in PyTorch.
It is a data structure that represents a multi-dimensional array,
and it can be used to store and manipulate data of various types.
Tensors can run on both CPU and GPU, making it easier to perform fast computations.

Creating Tensors
----------------

You can create Tensors in various ways. Here are a few examples:

1. **From a Python list:**

   .. code-block:: python

      import torch

      # Creating a tensor from a list
      data = [[1, 2], [3, 4]]
      tensor = torch.tensor(data)
      print(tensor)
      
   This will creating a 2 by 2 2D Tensor with the first row containing the values 1 and 2, and the second row containing the values 3 and 4.

2. **Using built-in functions:**

   .. code-block:: python

      # Creating a 2x2 tensor with random values
      random_tensor = torch.rand(2, 2)
      print(random_tensor)

      # Creating a 2x2 tensor filled with zeros
      zeros_tensor = torch.zeros(2, 2)
      print(zeros_tensor)

      # Creating a 2x2 tensor filled with ones
      ones_tensor = torch.ones(2, 2)
      print(ones_tensor)

Basic Operations
----------------

Tensors support a variety of operations. Here are some common operations:

1. **Arithmetic Operations:**

   .. code-block:: python

      # Addition
      tensor_a = torch.tensor([[1, 2], [3, 4]])
      tensor_b = torch.tensor([[5, 6], [7, 8]])
      tensor_sum = tensor_a + tensor_b
      print(tensor_sum)

      # Multiplication
      tensor_product = tensor_a * tensor_b
      print(tensor_product)

      # Matrix multiplication
      tensor_matmul = torch.matmul(tensor_a, tensor_b)
      print(tensor_matmul)

2. **Indexing and Slicing:**

   .. code-block:: python

      # Indexing (first row, second value)
      print(tensor_a[0, 1])  # Output: 2

      # Slicing (all rows, second value)
      print(tensor_a[:, 1])  # Output: tensor([2, 4])

3. **Reshaping:**

   .. code-block:: python

      # Reshaping a tensor to a 4-value 1D tensor.
      tensor_reshaped = tensor_a.view(4)
      print(tensor_reshaped)  # Output: tensor([1, 2, 3, 4])

4. **GPU Acceleration:**

   .. code-block:: python

      # Moving a tensor to the GPU
      if torch.cuda.is_available(): # If the GPU is available
          tensor_gpu = tensor_a.to('cuda') # Send the tensor to GPU memory
          print(tensor_gpu)

   When a Tensor is on GPU memory, all operations are run in parallel on the GPU.

Tensors and Gradients
---------------------

One of the powerful features of PyTorch Tensors is their ability to track gradients. This is essential for training neural networks using backpropagation. By setting ``requires_grad=True``, PyTorch will automatically compute gradients for the tensor during the backward pass. By default, ``requires_grad`` is set to ``False``.

   .. code-block:: python

      # Creating a tensor with gradient tracking
      x = torch.tensor([[2., 3.]], requires_grad=True)

      # Performing a simple operation
      y = x + 2
      z = y * y * 3
      out = z.mean()

      # Computing the gradients
      out.backward()
      print(x.grad)  # Output: tensor([[10.5000, 13.5000]])

This introduction covers the basic principles of PyTorch Tensors, how to create them, perform basic operations, and use them with gradients. For more detailed information and advanced operations, refer to the PyTorch documentation and tutorials.
