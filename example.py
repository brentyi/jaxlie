import numpy as onp
from jax import numpy as jnp

import jaxlie

# Compute a w<-b transform by integrating over an se(3) screw
twist = jnp.array([1.0, 0.0, 0.2, 0.0, 0.5, 0.0])
T_w_b = jaxlie.SE3.exp(twist)
p_b = onp.random.randn(3)

# We can print the underlying (quaternion) rotation term:
print(T_w_b.rotation)

# Or translation:
print(T_w_b.translation)

# Transform points with the `@` operator, `.apply()`, or the matrix form:
p_w = T_w_b @ p_b
print(p_w)

p_w = T_w_b.apply(p_b)
print(p_w)

p_w = (T_w_b.as_matrix() @ jnp.append(p_b, 1.0))[:-1]
print(p_w)

# Compose transforms with the `@` operator or `.product()`:
T_b_a = jaxlie.SE3.identity()

T_w_a = T_w_b @ T_b_a
print(T_w_a)

T_w_a = T_w_b.product(T_b_a)
print(T_w_a)

# Compute inverses:
T_b_w = T_w_b.inverse()
identity = T_w_b @ T_b_w
print(identity)

# Recover our twist a la `vee(logm(T_w_b.as_matrix()))`:
twist = T_w_b.log()
print(twist)
