# jaxlie

![build](https://github.com/brentyi/jaxlie/workflows/build/badge.svg)
![mypy](https://github.com/brentyi/jaxlie/workflows/mypy/badge.svg?branch=master)
![lint](https://github.com/brentyi/jaxlie/workflows/lint/badge.svg)
[![codecov](https://codecov.io/gh/brentyi/jaxlie/branch/master/graph/badge.svg)](https://codecov.io/gh/brentyi/jaxlie)

**[ [API reference](https://brentyi.github.io/jaxlie) ]**
**[ [PyPI](https://pypi.org/project/jaxlie/) ]**

`jaxlie` is a Lie theory library for rigid body transformations and optimization
in JAX.

Current functionality:

- SO(2), SE(2), SO(3), and SE(3) Lie groups implemented as high-level
  dataclasses.
- **`exp()`**, **`log()`**, **`adjoint()`**, **`multiply()`**, **`inverse()`**,
  and **`identity()`** implementations for each Lie group.
- Pytree registration for all dataclasses.
- Helpers + analytical Jacobians for on-manifold optimization
  (**`jaxlie.manifold`**).

---

##### Install (Python >=3.6)
```bash
pip install jaxlie
```

---

##### Example usage

```python
import numpy as onp

from jaxlie import SE3

#############################
# (1) Constructing transforms
#############################

# We can compute a w<-b transform by integrating over an se(3) screw, equivalent
# to `SE3.from_matrix(expm(wedge(twist)))`
twist = onp.array([1.0, 0.0, 0.2, 0.0, 0.5, 0.0])
T_w_b = SE3.exp(twist)
p_b = onp.random.randn(3)

# We can print the (quaternion) rotation term; this is a `SO3` object:
print(T_w_b.rotation)

# Or print the translation; this is a simple array with shape (3,):
print(T_w_b.translation)

# Or the underlying parameters; this is a length-7 (translation, quaternion) array:
print(T_w_b.xyz_wxyz)  # SE3-specific field
print(T_w_b.parameters)  # Alias shared by all groups

# There are also other helpers to generate transforms, eg from matrices:
T_w_b = SE3.from_matrix(T_w_b.as_matrix())

# Or from explicit rotation and translation terms:
T_w_b = SE3.from_rotation_and_translation(
    rotation=T_w_b.rotation,
    translation=T_w_b.translation,
)

# Or with the dataclass constructor + the underlying length-7 parameterization:
T_w_b = SE3(xyz_wxyz=T_w_b.xyz_wxyz)


#############################
# (2) Applying transforms
#############################

# Transform points with the `@` operator:
p_w = T_w_b @ p_b
print(p_w)

# or `.apply()`:
p_w = T_w_b.apply(p_b)
print(p_w)

# or the homogeneous matrix form:
p_w = (T_w_b.as_matrix() @ onp.append(p_b, 1.0))[:-1]
print(p_w)


#############################
# (3) Composing transforms
#############################

# Compose transforms with the `@` operator:
T_b_a = SE3.identity()
T_w_a = T_w_b @ T_b_a
print(T_w_a)

# or `.multiply()`:
T_w_a = T_w_b.multiply(T_b_a)
print(T_w_a)


#############################
# (4) Misc
#############################

# Compute inverses:
T_b_w = T_w_b.inverse()
identity = T_w_b @ T_b_w
print(identity)

# Compute adjoints:
adjoint_T_w_b = T_w_b.adjoint()
print(adjoint_T_w_b)

# Recover our twist, equivalent to `vee(logm(T_w_b.as_matrix()))`:
twist = T_w_b.log()
print(twist)
```

---

##### Misc

`jaxlie` is heavily inspired by the C++ library
[Sophus](https://github.com/strasdat/Sophus).
