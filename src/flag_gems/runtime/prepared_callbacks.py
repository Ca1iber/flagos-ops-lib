# SPDX-License-Identifier: Apache-2.0
"""Optional prepared-operation callbacks; all model state belongs to the caller.

Install handlers before execution/graph capture and keep their resources alive
for every captured graph. Handlers return None to retain the original dispatch.
They must compute directly rather than recursively calling the same wrapper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from torch import Tensor

# Publish the pair together; FlagGems retains only the two callable references.
_prepared_handlers = (None, None)


def set_prepared_handlers(*, matmul=None, linear=None) -> None:
    """Replace both handlers; omitted/None entries clear the corresponding hook.

    matmul(a, b, *, out=None) and linear(input, weight, bias=None) return
    a result tensor or None. Registration owns no weights or model caches.
    """
    if matmul is not None and not callable(matmul):
        raise TypeError("matmul handler must be callable or None")
    if linear is not None and not callable(linear):
        raise TypeError("linear handler must be callable or None")
    global _prepared_handlers
    _prepared_handlers = (matmul, linear)


def try_prepared_matmul(a, b, *, out=None) -> Optional[Tensor]:
    """Call the prepared matmul handler, or return None for normal dispatch."""
    handler = _prepared_handlers[0]
    if handler is None:
        return None
    return handler(a, b, out=out)


def try_prepared_linear(input, weight, bias=None) -> Optional[Tensor]:
    """Call the prepared linear handler, or return None for normal dispatch."""
    handler = _prepared_handlers[1]
    if handler is None:
        return None
    return handler(input, weight, bias)


__all__ = [
    "set_prepared_handlers",
    "try_prepared_matmul",
    "try_prepared_linear",
]
