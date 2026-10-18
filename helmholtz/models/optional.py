"""Opt-in integration with externally supplied MGCFNN code."""

import importlib
import os


def load_mgcfnn_model():
    """Return a compatible external MyModel class without redistributing it.

    The upstream MGCFNN repository did not specify a license at release
    preparation time. Users who have permission to use that implementation may
    supply their own importable module with MyModel and a training-safe Fourier
    implementation (do not cache detached weights during training).
    """
    module_name = os.environ.get("NPBS_MGCFNN_MODULE")
    if not module_name:
        raise ImportError(
            "MGCFNN code is not bundled because its upstream repository does "
            "not specify a license. Supply an authorized compatible module "
            "and set NPBS_MGCFNN_MODULE to its import name. See helmholtz/README.md."
        )
    module = importlib.import_module(module_name)
    if hasattr(module, "global_is_solve"):
        module.global_is_solve = False
    if not hasattr(module, "MyModel"):
        raise ImportError(f"{module_name!r} must export a compatible MyModel class")
    return module.MyModel
