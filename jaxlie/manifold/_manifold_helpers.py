from typing import TypeVar

from .. import types
from .._base import MatrixLieGroup

T = TypeVar("T", bound=MatrixLieGroup)


def rplus(transform: T, delta: types.TangentVector) -> T:
    """Manifold right plus.

    Computes `T_wb = T_wa @ exp(delta)`.

    Args:
        transform (T): `T_wa`
        delta (types.TangentVector): `T_ab.log()`

    Returns:
        T: `T_wb`
    """
    return transform @ type(transform).exp(delta)


def rminus(a: T, b: T) -> types.TangentVector:
    """Manifold right minus.

    Computes `delta = (T_wa.inverse() @ T_wb).log()`.

    Args:
        a (T): `T_wa`
        b (T): `T_wb`

    Returns:
        types.TangentVector: `T_ab.log()`
    """
    return (a.inverse() @ b).log()
