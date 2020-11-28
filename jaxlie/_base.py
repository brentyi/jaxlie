import abc
from typing import Type, TypeVar, overload

from ._types import Matrix, TangentVector, Vector

T = TypeVar("T", bound="MatrixLieGroup")


class MatrixLieGroup(abc.ABC):

    # Shared implementations

    @overload
    def __matmul__(self: T, other: T) -> T:
        ...

    @overload
    def __matmul__(self: T, other: Vector) -> Vector:
        ...

    def __matmul__(self, other):
        """Operator overload, for composing transformations and/or applying them to
        points.
        """
        if type(self) is type(other):
            return self.product(T)
        elif isinstance(other, Vector):
            return self.apply(target=other)
        else:
            assert False, "Invalid argument"

    # Abstract factory

    @classmethod
    @abc.abstractmethod
    def identity(cls: Type[T]) -> T:
        """Returns identity element.

        Args:

        Returns:
            Matrix: Identity.
        """

    # Abstract accessors

    @staticmethod
    @abc.abstractmethod
    def matrix_dim() -> int:
        """Get dimension of (square) matrix representation.

        Returns:
            int: Matrix dimensionality.
        """

    @staticmethod
    @abc.abstractmethod
    def compact_dim() -> int:
        """Get dimensionality of compact representation.

        Args:

        Returns:
            int: Compact representation dimension.
        """

    @staticmethod
    @abc.abstractmethod
    def tangent_dim() -> int:
        """Get dimensionality of tangent space.

        Returns:
            int: Tangent space dimension.
        """

    @abc.abstractmethod
    def matrix(self) -> Matrix:
        """Get value as a matrix."""

    @abc.abstractmethod
    def compact(self) -> Matrix:
        """Get compact representation."""

    # Operations

    @abc.abstractmethod
    def apply(self: T, target: Vector) -> Vector:
        """Applies transformation to a vector.

        Args:
            target (Vector): Vector to transform.

        Returns:
            Vector: Transformed vector.
        """

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
    def exp(tangent: TangentVector) -> T:
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

    @abc.abstractmethod
    def normalize(self: T) -> T:
        """Normalize/projects values and returns.

        Returns:
            T: Normalized group member.
        """
