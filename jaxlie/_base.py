import abc
import dataclasses
from typing import Tuple, Type, TypeVar

import jax
import numpy as onp
from jax import numpy as jnp
from overrides import overrides

from ._types import Matrix, TangentVector, Vector

T = TypeVar("T", bound="MatrixLieGroup")


class MatrixLieGroup(abc.ABC):

    # Shared implementations

    @classmethod
    def identity(cls: Type[T]) -> T:
        """Returns identity element.

        Args:

        Returns:
            Matrix: Identity.
        """
        return cls.from_matrix(onp.eye(cls.get_matrix_dim()))

    def __matmul__(a: T, b: T) -> T:
        """Operator override: `a @ b` computes `a.product(b)`.

        Args:
            a (T): a
            b (T): b

        Returns:
            T: a @ b
        """
        return a.product(b)

    # Abstract methods

    @property
    @abc.abstractmethod
    def matrix(self) -> Matrix:
        """Get value as a matrix."""

    @property
    @abc.abstractmethod
    def compact(self) -> Vector:
        """Get compact representation."""

    @staticmethod
    @abc.abstractmethod
    def get_matrix_dim() -> int:
        """Get dimension of (square) matrix representation.

        Returns:
            int: Matrix dimensionality.
        """

    @staticmethod
    @abc.abstractmethod
    def get_tangent_dim() -> int:
        """Get dimensionality of tangent space.

        Args:

        Returns:
            int: Tangent space dimension.
        """

    @staticmethod
    @abc.abstractmethod
    def get_compact_dim() -> int:
        """Get dimensionality of compact representation.

        Args:

        Returns:
            int: Compact representation dimension.
        """

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls: Type[T], matrix: Matrix) -> T:
        """Factory for creating a group member from its full square matrix representation."""

    @classmethod
    @abc.abstractmethod
    def from_compact(cls: Type[T], vector: Vector) -> T:
        """Factory for creating a group member from its compact representation."""

    @abc.abstractmethod
    def product(self: T, other: T) -> T:
        """Left-multiplies this transformations with another.

        Args:
            other (T): other

        Returns:
            T: self @ other
        """

    @staticmethod
    @abc.abstractmethod
    def exp(cls: Type[T], tangent: TangentVector) -> T:
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent (TangentVector): Input.

        Returns:
            Matrix: Output.
        """

    @staticmethod
    @abc.abstractmethod
    def log(self: T) -> TangentVector:
        """Computes `logm(vee(tangent))`.

        Args:
            x (Matrix): Input.

        Returns:
            TangentVector: Output.
        """

    @staticmethod
    @abc.abstractmethod
    def inverse(self: T) -> T:
        """Computes the inverse of x.

        Args:
            x (Matrix): Input.

        Returns:
            Matrix: Output.
        """
