"""Neural preconditioners for periodic convection–diffusion–reaction equations."""

from .model import FNOSolver
from .operators import SpectralPDELoss

__all__ = ["FNOSolver", "SpectralPDELoss"]
