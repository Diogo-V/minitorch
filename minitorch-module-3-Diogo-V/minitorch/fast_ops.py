from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        for out_idx in prange(len(out)):
            out_midx = np.zeros(MAX_DIMS, np.int32)
            in_midx = np.zeros(MAX_DIMS, np.int32)
            # convert multidimensional index for the output
            to_index(out_idx, out_shape, out_midx)
            broadcast_index(out_midx, out_shape, in_shape, in_midx)

            in_idx = index_to_position(in_midx, in_strides)
            out[index_to_position(out_midx, out_strides)] = fn(in_storage[in_idx])

    return njit(fn=_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.


    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
        fn: function maps two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        for out_idx in prange(len(out)):
            out_midx = np.zeros(MAX_DIMS, np.int32)
            a_midx = np.zeros(MAX_DIMS, np.int32)
            b_midx = np.zeros(MAX_DIMS, np.int32)
            to_index(out_idx, out_shape, out_midx)
            broadcast_index(out_midx, out_shape, a_shape, a_midx)
            broadcast_index(out_midx, out_shape, b_shape, b_midx)

            a_idx = index_to_position(a_midx, a_strides)
            b_idx = index_to_position(b_midx, b_strides)
            out[index_to_position(out_midx, out_strides)] = fn(
                a_storage[a_idx], b_storage[b_idx]
            )

    return njit(fn=_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
        fn: reduction function mapping two floats to float.

    Returns:
        Tensor reduce function
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_size: int = len(out)
        reduce_size: int = a_shape[reduce_dim]
        # Main loop in parallel
        for i in prange(out_size):
            # All indices use numpy buffers
            # out_index: Index = np.zeros_like(out_shape, dtype=np.int32)
            out_index: Index = np.zeros(MAX_DIMS, dtype=np.int32)
            # The index of out[i]
            to_index(i, out_shape, out_index)
            # The starting position in a to be reduced
            a_ordinal = index_to_position(out_index, a_strides)
            # Initialize the reduced value of a[i]
            reduced_val = out[i]
            # Inner-loop should not call any functions or write non-local variables
            for j in range(reduce_size):
                # Calculate the reduced value of a[i]
                reduced_val = fn(
                    reduced_val,
                    a_storage[a_ordinal + j * a_strides[reduce_dim]],
                )
            # Put the reduced data into out
            out[i] = reduced_val

    return njit(fn=_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # Tensors are [batch, in_features, out_features]
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Exposes indices to make more clear
    COMMON_BETWEEN_MATRICES = a_shape[-1]  # Must be equal to b_shape[-2]
    BATCH_SIZE, OUT_IN_FEATURES, OUT_OUT_FEATURES = out_shape[-3:]

    # We parallelize on the batch dimension since they are independent
    for batch in prange(BATCH_SIZE):
        # We loop twice on the out tensor to access each element independently
        for out_in_feature in range(OUT_IN_FEATURES):
            for out_out_feature in range(OUT_OUT_FEATURES):
                result = 0.0

                # The first section of the addition selects the correct batch section while the second one
                # selects the correct index to start multiplying
                a_ord = batch * a_batch_stride + out_in_feature * a_strides[-2]
                b_ord = batch * b_batch_stride + out_out_feature * b_strides[-1]

                # In were, we accumulate the vector multiplication. Since we already put the pointers
                # for a and b on their correct spots, we only need to move them by the strides of the
                # opposite dimensions during multiplication
                for _ in range(COMMON_BETWEEN_MATRICES):
                    result += a_storage[a_ord] * b_storage[b_ord]
                    a_ord += a_strides[-1]
                    b_ord += b_strides[-2]

                # Finds the correct index in the out tensor and puts the result there
                out_ord = (
                    batch * out_strides[-3]
                    + out_in_feature * out_strides[-2]
                    + out_out_feature * out_strides[-1]
                )
                out[out_ord] = result


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
