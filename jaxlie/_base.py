import abc
from typing import ClassVar, Generic, Type, TypeVar, overload

import numpy as onp
from jax import numpy as jnp
from overrides import EnforceOverrides, final, overrides

from . import hints

GroupType = TypeVar("GroupType", bound="MatrixLieGroup")
SEGroupType = TypeVar("SEGroupType", bound="SEBase")


class MatrixLieGroup(abc.ABC, EnforceOverrides):
    """Interface definition for matrix Lie groups."""

    # Class properties
    # > These will be set in `_utils.register_lie_group()`

    matrix_dim: ClassVar[int]
    """Dimension of square matrix output from `.as_matrix()`."""

    parameters_dim: ClassVar[int]
    """Dimension of underlying parameters, `.parameters()`."""

    tangent_dim: ClassVar[int]
    """Dimension of tangent space."""

    space_dim: ClassVar[int]
    """Dimension of coordinates that can be transformed."""

    def __init__(self, parameters: hints.Vector):
        """Construct a group object from its underlying parameters."""

        # Note that this method is implicitly overriden by the dataclass decorator and
        # should _not_ be marked abstract.
        raise NotImplementedError()

    # Shared implementations

    @overload
    def __matmul__(self: GroupType, other: GroupType) -> GroupType:
        ...

    @overload
    def __matmul__(self: GroupType, other: hints.Vector) -> hints.VectorJax:
        ...

    def __matmul__(self, other):
        """Overload for the `@` operator.

        Switches between the group action (`.apply()`) and multiplication
        (`.multiply()`) based on the type of `other`.
        """
        if isinstance(other, (onp.ndarray, jnp.ndarray)):
            return self.apply(target=other)
        if isinstance(other, MatrixLieGroup):
            return self.multiply(other=other)
        else:
            assert False, "Invalid argument"

    # Factory

    @classmethod
    @abc.abstractmethod
    def identity(cls: Type[GroupType]) -> GroupType:
        """Returns identity element.

        Returns:
            Identity element.
        """

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls: Type[GroupType], matrix: hints.Matrix) -> GroupType:
        """Get group member from matrix representation.

        Args:
            matrix: Matrix representaiton.

        Returns:
            Group member.
        """

    # Accessors

    @abc.abstractmethod
    def as_matrix(self) -> hints.MatrixJax:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @abc.abstractmethod
    def parameters(self) -> hints.Vector:
        """Get underlying representation."""

    # Operations

    @abc.abstractmethod
    def apply(self: GroupType, target: hints.Vector) -> hints.VectorJax:
        """Applies group action to a point.

        Args:
            target: Point to transform.

        Returns:
            Transformed point.
        """

    @abc.abstractmethod
    def multiply(self: GroupType, other: GroupType) -> GroupType:
        """Composes this transformation with another.

        Returns:
            self @ other
        """

    @classmethod
    @abc.abstractmethod
    def exp(cls: Type[GroupType], tangent: hints.TangentVector) -> GroupType:
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent: Tangent vector to take the exponential of.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def log(self: GroupType) -> hints.TangentVectorJax:
        """Computes `vee(logm(transformation matrix))`.

        Returns:
            Output. Shape should be `(tangent_dim,)`.
        """

    @abc.abstractmethod
    def adjoint(self: GroupType) -> hints.MatrixJax:
        """Computes the adjoint, which transforms tangent vectors between tangent spaces.

        More precisely, for a transform `GroupType`:
        ```
        GroupType @ exp(omega) = exp(Adj_T @ omega) @ GroupType
        ```

        In robotics, typically used for converting twists, wrenches, and Jacobians
        between our spatial and body representations.

        Returns:
            Output. Shape should be `(tangent_dim, tangent_dim)`.
        """

    @abc.abstractmethod
    def inverse(self: GroupType) -> GroupType:
        """Computes the inverse of our transform.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def normalize(self: GroupType) -> GroupType:
        """Normalize/projects values and returns.

        Returns:
            GroupType: Normalized group member.
        """

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls: Type[GroupType], key: jnp.ndarray) -> GroupType:
        """Draw a uniform sample from the group. Translations are in the range [-1, 1].

        Args:
            key: PRNG key, as returned by `jax.random.PRNGKey()`.

        Returns:
            Sampled group member.
        """


class SOBase(MatrixLieGroup):
    """Base class for special orthogonal groups."""


ContainedSOType = TypeVar("ContainedSOType", bound=SOBase)


class SEBase(Generic[ContainedSOType], MatrixLieGroup):
    """Base class for special Euclidean groups.

    Each SE(N) group member contains an SO(N) rotation, as well as an N-dimensional
    translation vector."""

    # SE-specific interface

    @classmethod
    @abc.abstractmethod
    def from_rotation_and_translation(
        cls: Type[SEGroupType],
        rotation: ContainedSOType,
        translation: hints.Vector,
    ) -> SEGroupType:
        """Construct a rigid transform from a rotation and a translation.

        Args:
            rotation: Rotation term.
            translation: translation term.

        Returns:
            Constructed transformation.
        """

    @abc.abstractmethod
    def rotation(self) -> ContainedSOType:
        """Returns a transform's rotation term."""

    @abc.abstractmethod
    def translation(self) -> hints.Vector:
        """Returns a transform's translation term."""

    # Overrides

    @final
    @overrides
    def apply(self, target: hints.Vector) -> hints.VectorJax:
        return self.rotation() @ target + self.translation()

    @final
    @overrides
    def multiply(self: SEGroupType, other: SEGroupType) -> SEGroupType:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=(self.rotation() @ other.translation()) + self.translation(),
        )

    @final
    @overrides
    def inverse(self: SEGroupType) -> SEGroupType:
        R_inv = self.rotation().inverse()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    @final
    @overrides
    def normalize(self: SEGroupType) -> SEGroupType:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
        )
