"""Keep legacy constructor-local dtype selection from changing caller state."""

from functools import wraps

import torch


def preserve_default_dtype(initializer):
    """Restore PyTorch's default dtype after a constructor, including failures."""
    @wraps(initializer)
    def wrapped(self, *args, **kwargs):
        previous = torch.get_default_dtype()
        try:
            return initializer(self, *args, **kwargs)
        finally:
            torch.set_default_dtype(previous)
    return wrapped
