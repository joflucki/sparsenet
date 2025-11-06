SparseNet API Documentation
===========================

SparseNet is composed of three core modules: ``linear``, ``nn`` and ``happy_lambda``.

``linear`` is used to train linear models of form :math:`y = ax + b` so that :math:`x` is sparse.

``nn`` on the other side, is used to train sparse neural networks, using PyTorch's neural net features.

``happy_lambda`` contains the algorithms used to compute the optimal penalty weight, a.k.a the "happy lambda".

.. toctree::
   :maxdepth: 1

   sparsenet.happy_lambda <happy_lambda/index>
   sparsenet.linear <linear/index>
   sparsenet.nn <nn/index>
