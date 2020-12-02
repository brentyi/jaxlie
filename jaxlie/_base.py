import abc
from typing import TYPE_CHECKING, TypeVar, overload

from ._types import Matrix, TangentVector, Vector

if TYPE_CHECKING:
    from ._se2 import SE2
    from ._se3 import SE3
    from ._so2 import SO2
    from ._so3 import SO3

T = TypeVar("T", bound="MatrixLieGroup")


class MatrixLieGroup(abc.ABC):

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

    # Shared implementations

    @overload
    def __matmul__(self: "SO2", other: "SE2") -> "SE2":
        # Upcast SO2 -> SE2
        ...

    @overload
    def __matmul__(self: "SE2", other: "SO2") -> "SE2":
        # Upcast SO2 -> SE2
        ...

    @overload
    def __matmul__(self: "SO3", other: "SE3") -> "SE3":
        # Upcast SO3 -> SE3
        ...

    @overload
    def __matmul__(self: "SE3", other: "SO3") -> "SE3":
        # Upcast SO3 -> SE3
        ...

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
        if isinstance(other, Vector):
            return self.apply(target=other)
        if isinstance(other, MatrixLieGroup):
            return self.product(other=other)
        else:
            assert False, "Invalid argument"

    # Factory

    @staticmethod
    @abc.abstractmethod
    def identity() -> "MatrixLieGroup":
        """Returns identity element.

        Returns:
            Matrix: Identity.
        """

    @staticmethod
    @abc.abstractmethod
    def from_matrix(matrix: Matrix) -> "MatrixLieGroup":
        """Get group member from matrix representation.

        Args:
            matrix (jnp.ndarray): Matrix representaiton.

        Returns:
            T: Group member.
        """

    # Accessors

    @abc.abstractmethod
    def as_matrix(self) -> Matrix:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @property
    @abc.abstractmethod
    def parameters(self) -> Vector:
        """Get underlying representation."""

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
    def exp(tangent: TangentVector) -> "MatrixLieGroup":
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent (TangentVector): Input.

        Returns:
            Matrix: Output.
        """

    @abc.abstractmethod
    def log(self: T) -> TangentVector:
        """Computes `vee(logm(transformation matrix))`.

        Returns:
            TangentVector: Output.
        """

    @abc.abstractmethod
    def inverse(self: T) -> T:
        """Computes the inverse of our transform.

        Returns:
            Matrix: Output.
        """

    @abc.abstractmethod
    def normalize(self: T) -> T:
        """Normalize/projects values and returns.

        Returns:
            T: Normalized group member.
        """
