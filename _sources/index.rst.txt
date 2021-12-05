jaxlie documentation
==========================================

|build| |nbsp| |mypy| |nbsp| |lint| |nbsp| |coverage|


:code:`jaxlie` is a Lie theory library for rigid body transformations and
optimization in JAX.


.. autoapi-inheritance-diagram:: jaxlie.SO2 jaxlie.SO3 jaxlie.SE2 jaxlie.SE3
   :top-classes: jaxlie.MatrixLieGroup


Current functionality:

- SO(2), SE(2), SO(3), and SE(3) Lie groups implemented as high-level
  dataclasses.

- :code:`exp()`, :code:`log()`, :code:`adjoint()`, :code:`multiply()`,
  :code:`inverse()`, and :code:`identity()` implementations for each Lie group.

- Pytree registration for all dataclasses.

- Helpers + analytical Jacobians for on-manifold optimization
  (:code:`jaxlie.manifold`).

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
   vmap_usage


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
