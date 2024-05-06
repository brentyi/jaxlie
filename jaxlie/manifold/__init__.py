from ._backprop import grad as grad
from ._backprop import value_and_grad as value_and_grad
from ._backprop import zero_tangents as zero_tangents
from ._deltas import rminus as rminus
from ._deltas import rplus as rplus
from ._deltas import (
    rplus_jacobian_parameters_wrt_delta as rplus_jacobian_parameters_wrt_delta,
)
from ._tree_utils import normalize_all as normalize_all
