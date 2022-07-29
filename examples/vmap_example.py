"""Examples of vectorizing transformations via vmap.

Omitted for brevity here, but note that in practice we usually want to JIT after
vmapping!"""

import jax
import numpy as onp

from jaxlie import SO3

N = 100

#############################
# (1) Setup.
#############################

# We start by creating two rotation objects:
# - R_single contains a standard single rotation.
# - R_stacked contained `N` rotations stacked together! Note that all Lie group objects
#   are PyTrees, so this has the same structure as R_single but with a batch axis in the
#   contained parameters array.

R_single = SO3.from_x_radians(onp.pi / 2.0)
assert R_single.wxyz.shape == (4,)

R_stacked = jax.vmap(SO3.from_x_radians)(
    onp.random.uniform(low=-onp.pi, high=onp.pi, size=(N,))
)
assert R_stacked.wxyz.shape == (N, 4)

# We can also create two arrays containing points: one is a single point, the other is
# `N` points stacked.
p_single = onp.random.uniform(size=(3,))
p_stacked = onp.random.uniform(size=(N, 3))

#############################
# (2) Applying 1 transformation to 1 point.
#############################

# Recall that these two approaches to transforming a point:
p_transformed_single = R_single @ p_single
assert p_transformed_single.shape == (3,)
p_transformed_single = R_single.apply(p_single)
assert p_transformed_single.shape == (3,)

# Are just syntactic sugar for calling:
p_transformed_single = SO3.apply(R_single, p_single)
assert p_transformed_single.shape == (3,)


#############################
# (3) Applying 1 transformation to N points.
#############################

# This follows standard vmap semantics!
p_transformed_stacked = jax.vmap(R_single.apply)(p_stacked)
assert p_transformed_stacked.shape == (N, 3)

# Note that this is equivalent to:
p_transformed_stacked = jax.vmap(lambda p: SO3.apply(R_single, p))(p_stacked)
assert p_transformed_stacked.shape == (N, 3)

#############################
# (4) Applying N transformations to N points.
#############################

# R_stacked and p_stacked both have an (N,) batch dimension compared to their "single"
# counterparts. We can therefore vmap over both arguments of SO3.apply:
p_transformed_stacked = jax.vmap(SO3.apply)(R_stacked, p_stacked)
assert p_transformed_stacked.shape == (N, 3)

#############################
# (5) Applying N transformations to 1 point.
#############################

p_transformed_stacked = jax.vmap(lambda R: SO3.apply(R, p_single))(R_stacked)
assert p_transformed_stacked.shape == (N, 3)

#############################
# (6) Multiplying transformations.
#############################

# The same concepts as above apply to other operations!
# For multiplication, these are all the same:
assert (R_single @ R_single).wxyz.shape == (4,)
assert (R_single.multiply(R_single)).wxyz.shape == (4,)
assert (SO3.multiply(R_single, R_single)).wxyz.shape == (4,)

# And therefore we can also do 1 x N multiplication:
assert (jax.vmap(R_single.multiply)(R_stacked)).wxyz.shape == (N, 4)
assert (jax.vmap(lambda R: SO3.multiply(R_single, R))(R_stacked)).wxyz.shape == (N, 4)

# Or N x N multiplication:
assert (jax.vmap(SO3.multiply)(R_stacked, R_stacked)).wxyz.shape == (N, 4)

# Or N x 1 multiplication:
assert (jax.vmap(lambda R: SO3.multiply(R, R_single))(R_stacked)).wxyz.shape == (N, 4)
