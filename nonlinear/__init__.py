"""Newton systems for the nonlinear Dirichlet Poisson equation."""

from .model import ModelConfig, NeuralNewtonSolver
from .operators import DirichletPoissonInverse, NewtonOperator
from .solvers import SolveResult, solve_inner, solve_newton

__all__ = [
    "DirichletPoissonInverse", "ModelConfig", "NeuralNewtonSolver",
    "NewtonOperator", "SolveResult", "solve_inner", "solve_newton",
]
