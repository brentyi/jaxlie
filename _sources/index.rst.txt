jaxlie documentation
==========================================

|build| |nbsp| |mypy| |nbsp| |lint| |nbsp| |coverage|

`jaxlie` is a Lie theory library for rigid body transformations in Jax. Implements
pytree-compatible SO(2), SO(3), SE(2), and SE(3) dataclasses with support for
(exp, log, adjoint, multiply, inverse, identity) operations. Borrows heavily from
the C++ library `Sophus <https://github.com/strasdat/Sophus>`_.

Source code on `Github <https://github.com/brentyi/jaxlie>`_.

.. toctree::
   :caption: API Reference
   :maxdepth: 3
   :titlesonly:
   :glob:

   api/jaxlie/index


.. toctree::
   :maxdepth: 5
   :caption: Example usage

   se3_overview


.. |build| image:: https://github.com/brentyi/jaxlie/workflows/build/badge.svg
   :alt: Build status icon
.. |mypy| image:: https://github.com/brentyi/jaxlie/workflows/mypy/badge.svg?branch=master
   :alt: Mypy status icon
.. |lint| image:: https://github.com/brentyi/jaxlie/workflows/lint/badge.svg
   :alt: Lint status icon
.. |coverage| image:: https://codecov.io/gh/brentyi/jaxlie/branch/master/graph/badge.svg
   :alt: Test coverage status icon
   :target: https://codecov.io/gh/brentyi/jaxlie
.. |nbsp| unicode:: 0xA0
   :trim:
