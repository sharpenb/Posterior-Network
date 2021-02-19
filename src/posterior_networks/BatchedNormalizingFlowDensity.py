import math

from pyro.distributions.util import copy_docs_from
from pyro.distributions.torch_transform import TransformModule
from torch.distributions import Transform, constraints
import torch.distributions as tdist
import torch.nn.functional as F
from torch import nn
import torch


@copy_docs_from(Transform)
class ConditionedRadial(Transform):
    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, params):
        super().__init__(cache_size=1)
        self._params = params
        self._cached_logDetJ = None

    # This method ensures that torch(u_hat, w) > -1, required for invertibility
    def u_hat(self, u, w):
        raise NotImplementedError()
        alpha = torch.matmul(u.unsqueeze(-2), w.unsqueeze(-1)).squeeze(-1)
        a_prime = -1 + F.softplus(alpha)
        return u + (a_prime - alpha) * w.div(w.pow(2).sum(dim=-1, keepdim=True))

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from the base distribution (or the output
        of a previous transform)
        """
        x0, alpha_prime, beta_prime = self._params() if callable(self._params) else self._params

        # Ensure invertibility using approach in appendix A.2
        alpha = F.softplus(alpha_prime)
        beta = -alpha + F.softplus(beta_prime)

        # Compute y and logDet using Equation 14.
        diff = x - x0[:, None, :]
        r = diff.norm(dim=-1, keepdim=True).squeeze()
        h = (alpha[:, None] + r).reciprocal()
        h_prime = - (h ** 2)
        beta_h = beta[:, None] * h

        self._cached_logDetJ = ((x0.size(-1) - 1) * torch.log1p(beta_h) +
                                torch.log1p(beta_h + beta[:, None] * h_prime * r))
        return x + beta_h[:, :, None] * diff

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor
        Inverts y => x. As noted above, this implementation is incapable of
        inverting arbitrary values `y`; rather it assumes `y` is the result of a
        previously computed application of the bijector to some `x` (which was
        cached on the forward call)
        """

        raise KeyError("ConditionedRadial object expected to find key in intermediates cache but didn't")

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian
        """
        x_old, y_old = self._cached_x_y
        if x is not x_old or y is not y_old:
            # This call to the parent class Transform will update the cache
            # as well as calling self._call and recalculating y and log_detJ
            self(x)

        return self._cached_logDetJ


@copy_docs_from(ConditionedRadial)
class Radial(ConditionedRadial, TransformModule):

    domain = constraints.real_vector
    codomain = constraints.real_vector
    bijective = True
    event_dim = 1

    def __init__(self, c, input_dim):
        super().__init__(self._params)

        self.x0 = nn.Parameter(torch.Tensor(c, input_dim,))
        self.alpha_prime = nn.Parameter(torch.Tensor(c,))
        self.beta_prime = nn.Parameter(torch.Tensor(c,))
        self.c = c
        self.input_dim = input_dim
        self.reset_parameters()

    def _params(self):
        return self.x0, self.alpha_prime, self.beta_prime

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.x0.size(1))
        self.alpha_prime.data.uniform_(-stdv, stdv)
        self.beta_prime.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)


class BatchedNormalizingFlowDensity(nn.Module):

    def __init__(self, c, dim, flow_length, flow_type='planar_flow'):
        super(BatchedNormalizingFlowDensity, self).__init__()
        self.c = c
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type

        self.mean = nn.Parameter(torch.zeros(self.c, self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim).repeat(self.c, 1, 1), requires_grad=False)

        if self.flow_type == 'radial_flow':
            self.transforms = nn.Sequential(*(
                Radial(c, dim) for _ in range(flow_length)
            ))
        elif self.flow_type == 'iaf_flow':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, z):
        sum_log_jacobians = 0
        z = z.repeat(self.c, 1, 1)
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(z, z_next)
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(
            self.mean.repeat(z.size(1), 1, 1).permute(1, 0, 2),
            self.cov.repeat(z.size(1), 1, 1, 1).permute(1, 0, 2, 3)
        ).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x
