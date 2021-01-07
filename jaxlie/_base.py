import abc
from typing import TypeVar, overload

from jax import numpy as jnp

from . import types

T = TypeVar("T", bound="MatrixLieGroup")


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    # Class properties
    # > These will be set in `_utils.register_lie_group()`

    matrix_dim: int = 0
    """Dimension of square matrix output from `.as_matrix()`."""

    parameters_dim: int = 0
    """Dimension of underlying parameters, `.parameters`."""

    tangent_dim: int = 0
    """Dimension of tangent space."""

    space_dim: int = 0
    """Dimension of coordinates that can be transformed."""

    def __init__(self, parameters: jnp.ndarray):
        """Construct a group object from its underlying parameters."""

        # Note that this method is implicitly overriden by the dataclass decorator and
        # should _not_ be marked abstract.
        raise NotImplementedError()

    # Shared implementations

    @overload
    def __matmul__(self: T, other: T) -> T:
        ...

    @overload
    def __matmul__(self: T, other: types.Vector) -> types.Vector:
        ...

    def __matmul__(self, other):
        """Overload for the `@` operator.

        Switches between the group action (`.apply()`) and multiplication
        (`.multiply()`) based on the type of `other`.
        """
        if isinstance(other, types.Vector):
            return self.apply(target=other)
        if isinstance(other, MatrixLieGroup):
            return self.multiply(other=other)
        else:
            assert False, "Invalid argument"

    # Factory

    @staticmethod
    @abc.abstractmethod
    def identity() -> "MatrixLieGroup":
        """Returns identity element.

        Returns:
            types.Matrix: Identity.
        """

    @staticmethod
    @abc.abstractmethod
    def from_matrix(matrix: types.Matrix) -> "MatrixLieGroup":
        """Get group member from matrix representation.

        Args:
            matrix (jnp.ndarray): types.Matrix representaiton.

        Returns:
            T: Group member.
        """

    # Accessors

    @abc.abstractmethod
    def as_matrix(self) -> types.Matrix:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @property
    @abc.abstractmethod
    def parameters(self) -> types.Vector:
        """Get underlying representation."""

    # Operations

    @abc.abstractmethod
    def apply(self: T, target: types.Vector) -> types.Vector:
        """Applies the group action.

        Args:
            target (types.Vector): types.Vector to transform.

        Returns:
            types.Vector: Transformed vector.
        """

    @abc.abstractmethod
    def multiply(self: T, other: T) -> T:
        """Left-multiplies this transformations with another.

        Args:
            other (T): other

        Returns:
            T: self @ other
        """

    @staticmethod
    @abc.abstractmethod
    def exp(tangent: types.TangentVector) -> "MatrixLieGroup":
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent (types.TangentVector): Input.

        Returns:
            MatrixLieGroup: Output.
        """

    @abc.abstractmethod
    def log(self: T) -> types.TangentVector:
        """Computes `vee(logm(transformation matrix))`.

        Returns:
            types.TangentVector: Output. Shape should be `(tangent_dim,)`.
        """

    @abc.abstractmethod
    def adjoint(self: T) -> types.Matrix:
        """Computes the adjoint, which transforms tangent vectors between tangent spaces.

        More precisely, for a transform `T`:
        ```
        T @ exp(omega) = exp(Adj_T @ omega) @ T
        ```

        For robotics, typically used for converting twists, wrenches, and Jacobians
        between our spatial and body representations.

        Returns:
            types.Matrix: Output. Shape should be `(tangent_dim, tangent_dim)`.
        """

    @abc.abstractmethod
    def inverse(self: T) -> T:
        """Computes the inverse of our transform.

        Returns:
            types.Matrix: Output.
        """

    @abc.abstractmethod
    def normalize(self: T) -> T:
        """Normalize/projects values and returns.

        Returns:
            T: Normalized group member.
        """
