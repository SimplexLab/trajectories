from abc import ABC, abstractmethod
from functools import partial
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from trajectories.pf_distance import compute_pf_distance, compute_pf_distance_nd


class Objective(ABC):
    def __init__(self, n_params: int, n_values: int):
        self.n_params = n_params
        self.n_values = n_values

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Compute the value of the objective function at x. It has to be a vector."""

    @abstractmethod
    def jacobian(self, x: Tensor) -> Tensor:
        """
        Compute the value of the Jacobian of the objective function at x. It is a matrix of shape
        [n_values, n_params].
        """

    def __str__(self) -> str:
        """Return a string representation of the objective function."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.n_values})"


class WithPFDistanceMixin(ABC):
    """
    Mixin adding the possibility to compute the distance to the Pareto front. Since this function
    may be costly to initialize but cheap to run, we provide a function returning the function that
    can compute the distance to the pareto front.
    """

    @abstractmethod
    def make_pf_distance_fn(self) -> Callable[[Tensor], Tensor]:
        """
        Creates the function that can compute from x the distance between f(x) and the Pareto front.
        """


class WithSPSMappingMixin(ABC):
    """Mixin adding the possibility to get the Strong Pareto stationary mapping."""

    class SPSMapping(ABC):
        N_SAMPLES: int  # Preferred number of samples

        def __call__(self, w: Tensor) -> Tensor:
            """
            Map a vector with (strictly) positive coordinates to another vector.

            :param w: The vector with (strictly) positive coordinates.
            """

            if (w.le(0.0)).any():
                raise ValueError(
                    f"All coordinates of w must be (strictly) positive. Found w = {w}."
                )

            return self._compute(w)

        @abstractmethod
        def _compute(self, w: Tensor) -> Tensor:
            pass

        def sample(self, n_samples: int, eps: float) -> Tensor:
            # TODO: we need to handle the case with more values than 2 (maybe with another subclass)
            ws_np = np.linspace([0 + eps, 1 - eps], [1 - eps, 0 + eps], n_samples)
            ws = torch.tensor(ws_np)
            sps_points = torch.stack([self(w) for w in ws])
            return sps_points

    @property
    @abstractmethod
    def sps_mapping(self) -> SPSMapping:
        pass


class QuadraticForm(Objective):
    def __init__(self, As: list[Tensor], us: list[Tensor]):
        if len(As) != len(us):
            raise ValueError("As and us must have the same length.")

        if len(As) < 1:
            raise ValueError("As and us must have at least one element.")

        super().__init__(n_params=len(us[0]), n_values=len(As))
        # Note that if A is not PSD, the objective is not convex.
        self.As = As
        self.us = us

    def __call__(self, x: Tensor) -> Tensor:
        objective_values = [self.quad(x, A, u) for A, u in zip(self.As, self.us)]
        return torch.stack(objective_values)

    def jacobian(self, x: Tensor) -> Tensor:
        return torch.vstack([2 * (x - u) @ A for A, u in zip(self.As, self.us)])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(As={self.As}, us={self.us})"

    @staticmethod
    def quad(x: Tensor, A: Tensor, u: Tensor):
        x_minus_u = x - u
        return x_minus_u @ A @ x_minus_u


class ConvexQuadraticForm(QuadraticForm, WithSPSMappingMixin, WithPFDistanceMixin):
    def __init__(self, Bs: list[Tensor], us: list[Tensor]):
        self.Bs = Bs
        super().__init__(As=[B @ B.T for B in self.Bs], us=us)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Bs={self.Bs}, us={self.us})"

    class SPSMapping(WithSPSMappingMixin.SPSMapping):
        N_SAMPLES = 100

        def __init__(self, As: list[Tensor], us: list[Tensor]):
            self.As = As
            self.us = us

        def _compute(self, w: Tensor) -> Tensor:
            G = torch.stack([weight * A for weight, A in zip(w, self.As)]).sum(dim=0)
            b = torch.stack([weight * A @ u for weight, A, u in zip(w, self.As, self.us)]).sum(
                dim=0
            )
            return torch.linalg.lstsq(G, b, driver="gelsd").solution

    @property
    def sps_mapping(self) -> SPSMapping:
        return self.SPSMapping(self.As, self.us)

    def make_pf_distance_fn(self) -> Callable[[Tensor], Tensor]:
        sps_points = self.sps_mapping.sample(self.sps_mapping.N_SAMPLES, eps=1e-5)
        pf_points = torch.stack([self(x) for x in sps_points])

        return partial(compute_pf_distance, pf_points)


class ElementWiseQuadratic(Objective, WithSPSMappingMixin, WithPFDistanceMixin):
    # TODO: we should probably make this a subclass of CQF
    def __init__(self, n_dim: int):
        super().__init__(n_params=n_dim, n_values=n_dim)

    def __call__(self, x: Tensor) -> Tensor:
        if len(x) != self.n_values:
            raise ValueError("x must have the same length as the number of values.")
        return x**2

    def jacobian(self, x: Tensor) -> Tensor:
        return torch.diag(torch.stack([2 * x[0], 2 * x[1]]))

    class SPSMapping(WithSPSMappingMixin.SPSMapping):
        N_SAMPLES = 1

        def __init__(self, n_values: int):
            self.n_values = n_values

        def _compute(self, w: Tensor) -> Tensor:
            return torch.zeros(self.n_values)

    @property
    def sps_mapping(self) -> SPSMapping:
        return self.SPSMapping(self.n_values)

    def make_pf_distance_fn(self) -> Callable[[Tensor], Tensor]:
        # TODO: there is some code duplication here
        sps_points = self.sps_mapping.sample(self.SPSMapping.N_SAMPLES, eps=1e-5)
        pf_points = torch.stack([self(x) for x in sps_points])

        return partial(compute_pf_distance, pf_points)


class Multinorm(Objective, WithSPSMappingMixin, WithPFDistanceMixin):
    # TODO: this is actually a convex quadratic form I think
    def __init__(self, a: Tensor):
        n = len(a)
        super().__init__(n_params=n, n_values=n)
        self.a = a

    def __call__(self, x: Tensor) -> Tensor:
        if len(x) != self.n_values:
            raise ValueError("x must have the same length as the number of values.")

        # f_i(x) = a_i * || x - a_i * e_i  ||²
        return self.a * torch.norm(x.expand(len(x), len(x)) - torch.diag(self.a), dim=1) ** 2

    def jacobian(self, x: Tensor) -> Tensor:
        return self.a * 2 * (x.expand(len(x), len(x)) - torch.diag(self.a))

    class SPSMapping(WithSPSMappingMixin.SPSMapping):
        N_SAMPLES = 100

        def __init__(self, n_values: int, a: Tensor):
            self.n_values = n_values
            self.a = a

        def _compute(self, w: Tensor) -> Tensor:
            # return (w * (self.a ** 2)) / (torch.sum(w * self.a))
            return w * self.a

    @property
    def sps_mapping(self) -> SPSMapping:
        return self.SPSMapping(self.n_values, self.a)

    def make_pf_distance_fn(self) -> Callable[[Tensor], Tensor]:
        sps_points = self.sps_mapping.sample(self.SPSMapping.N_SAMPLES, eps=1e-5)
        pf_points = torch.stack([self(x) for x in sps_points])

        return partial(compute_pf_distance_nd, pf_points)
