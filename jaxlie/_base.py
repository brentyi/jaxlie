import abc
from typing import ClassVar, Generic, Tuple, TypeVar, Union, overload

import jax
import numpy as onp
from jax import numpy as jnp
from typing_extensions import Self, final, get_args, override

from . import hints


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    # Class properties.
    # > These will be set in `_utils.register_lie_group()`.

    matrix_dim: ClassVar[int]
    """Dimension of square matrix output from `.as_matrix()`."""

    parameters_dim: ClassVar[int]
    """Dimension of underlying parameters, `.parameters()`."""

    tangent_dim: ClassVar[int]
    """Dimension of tangent space."""

    space_dim: ClassVar[int]
    """Dimension of coordinates that can be transformed."""

    def __init__(
        # Notes:
        # - For the constructor signature to be consistent with subclasses, `parameters`
        #   should be marked as positional-only. But this isn't possible in Python 3.7.
        # - This method is implicitly overriden by the dataclass decorator and
        #   should _not_ be marked abstract.
        self,
        parameters: jax.Array,
    ):
        """Construct a group object from its underlying parameters."""
        raise NotImplementedError()

    # Shared implementations.

    @overload
    def __matmul__(self, other: Self) -> Self: ...

    @overload
    def __matmul__(self, other: hints.Array) -> jax.Array: ...

    def __matmul__(self, other: Union[Self, hints.Array]) -> Union[Self, jax.Array]:
        """Overload for the `@` operator.

        Switches between the group action (`.apply()`) and multiplication
        (`.multiply()`) based on the type of `other`.
        """
        if isinstance(other, (onp.ndarray, jax.Array)):
            return self.apply(target=other)
        elif isinstance(other, MatrixLieGroup):
            assert self.space_dim == other.space_dim
            return self.multiply(other=other)
        else:
            assert False, f"Invalid argument type for `@` operator: {type(other)}"

    # Factory.

    @classmethod
    @abc.abstractmethod
    def identity(cls, batch_axes: Tuple[int, ...] = ()) -> Self:
        """Returns identity element.

        Args:
            batch_axes: Any leading batch axes for the output transform.

        Returns:
            Identity element.
        """

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls, matrix: hints.Array) -> Self:
        """Get group member from matrix representation.

        Args:
            matrix: Matrix representaiton.

        Returns:
            Group member.
        """

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> jax.Array:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @abc.abstractmethod
    def parameters(self) -> jax.Array:
        """Get underlying representation."""

    # Operations.

    @abc.abstractmethod
    def apply(self, target: hints.Array) -> jax.Array:
        """Applies group action to a point.

        Args:
            target: Point to transform.

        Returns:
            Transformed point.
        """

    @abc.abstractmethod
    def multiply(self, other: Self) -> Self:
        """Composes this transformation with another.

        Returns:
            self @ other
        """

    @classmethod
    @abc.abstractmethod
    def exp(cls, tangent: hints.Array) -> Self:
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent: Tangent vector to take the exponential of.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def log(self) -> jax.Array:
        """Computes `vee(logm(transformation matrix))`.

        Returns:
            Output. Shape should be `(tangent_dim,)`.
        """

    @abc.abstractmethod
    def adjoint(self) -> jax.Array:
        """Computes the adjoint, which transforms tangent vectors between tangent
        spaces.

        More precisely, for a transform `GroupType`:
        ```
        GroupType @ exp(omega) = exp(Adj_T @ omega) @ GroupType
        ```

        In robotics, typically used for transforming twists, wrenches, and Jacobians
        across different reference frames.

        Returns:
            Output. Shape should be `(tangent_dim, tangent_dim)`.
        """

    @abc.abstractmethod
    def inverse(self) -> Self:
        """Computes the inverse of our transform.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def normalize(self) -> Self:
        """Normalize/projects values and returns.

        Returns:
            Normalized group member.
        """

    @abc.abstractmethod
    def jlog(self) -> jax.Array:
        """
        Computes the Jacobian of the logarithm of the group element when a
        local perturbation is applied.

        This is equivalent to the inverse of the right Jacobian, or:

        ```
        jax.jacrev(lambda x: (T @ exp(x)).log())(jnp.zeros(tangent_dim))
        ```

        where `T` is the group element and `exp(x)` is the tangent vector.

        Returns:
            The Jacobian of the logarithm, having the dimensions `(tangent_dim, tangent_dim,)` or batch of these Jacobians.
        """

    @classmethod
    @abc.abstractmethod
    def sample_uniform(cls, key: jax.Array, batch_axes: Tuple[int, ...] = ()) -> Self:
        """Draw a uniform sample from the group. Translations (if applicable) are in the
        range [-1, 1].

        Args:
            key: PRNG key, as returned by `jax.random.PRNGKey()`.
            batch_axes: Any leading batch axes for the output transforms. Each
                sampled transform will be different.

        Returns:
            Sampled group member.
        """

    @final
    def get_batch_axes(self) -> Tuple[int, ...]:
        """Return any leading batch axes in contained parameters. If an array of shape
        `(100, 4)` is placed in the wxyz field of an SO3 object, for example, this will
        return `(100,)`."""
        return self.parameters().shape[:-1]


class SOBase(MatrixLieGroup):
    """Base class for special orthogonal groups."""


ContainedSOType = TypeVar("ContainedSOType", bound=SOBase)


class SEBase(Generic[ContainedSOType], MatrixLieGroup):
    """Base class for special Euclidean groups.

    Each SE(N) group member contains an SO(N) rotation, as well as an N-dimensional
    translation vector.
    """

    # SE-specific interface.

    @classmethod
    @abc.abstractmethod
    def from_rotation_and_translation(
        cls,
        rotation: ContainedSOType,
        translation: hints.Array,
    ) -> Self:
        """Construct a rigid transform from a rotation and a translation.

        Args:
            rotation: Rotation term.
            translation: translation term.

        Returns:
            Constructed transformation.
        """

    @final
    @classmethod
    def from_rotation(cls, rotation: ContainedSOType) -> Self:
        return cls.from_rotation_and_translation(
            rotation=rotation,
            translation=jnp.zeros(
                (*rotation.get_batch_axes(), cls.space_dim),
                dtype=rotation.parameters().dtype,
            ),
        )

    @final
    @classmethod
    def from_translation(cls, translation: hints.Array) -> Self:
        # Extract rotation class from type parameter.
        assert len(cls.__orig_bases__) == 1  # type: ignore
        return cls.from_rotation_and_translation(
            rotation=get_args(cls.__orig_bases__[0])[0].identity(),  # type: ignore
            translation=translation,
        )

    @abc.abstractmethod
    def rotation(self) -> ContainedSOType:
        """Returns a transform's rotation term."""

    @abc.abstractmethod
    def translation(self) -> jax.Array:
        """Returns a transform's translation term."""

    # Overrides.

    @final
    @override
    def apply(self, target: hints.Array) -> jax.Array:
        return self.rotation() @ target + self.translation()  # type: ignore

    @final
    @override
    def multiply(self, other: Self) -> Self:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=(self.rotation() @ other.translation()) + self.translation(),
        )

    @final
    @override
    def inverse(self) -> Self:
        R_inv = self.rotation().inverse()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    @final
    @override
    def normalize(self) -> Self:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
        )
