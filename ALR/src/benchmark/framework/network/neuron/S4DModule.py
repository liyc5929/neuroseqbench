import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .lif import LIFAct


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1. - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, neuron_num, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = neuron_num
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, neuron_num, d_state=64, dropout=0.0, time_step=0, return_state=False, residual=True, recurrent=False, binary = None, threshold=0.5, surro_grad=None, **kernel_args):
        super().__init__()

        self.h = neuron_num
        self.n = d_state
        self.d_output = self.h
        self.time_step = time_step
        self.return_state = return_state
        self.residual = residual

        self.D = nn.Parameter(torch.randn(self.h))
        self.binary = binary

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if self.binary != 'GSU':
            self.output_linear = nn.Sequential(
                nn.Conv1d(self.h, 2*self.h, kernel_size=1),
                nn.GLU(dim=-2),
            )
        else:
            self.w = nn.Parameter(torch.randn(self.h,self.h))
            self.b = nn.Parameter(torch.zeros(self.h))
            self.c = nn.Parameter(torch.zeros(self.h))
            bound = 1 / math.sqrt(self.h)
            nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
            nn.init.uniform_(self.b, -bound, bound)
            nn.init.uniform_(self.c, -bound, bound)
        self.LN = nn.LayerNorm(self.h)
        self.threshold = threshold
        self.surro_grad = surro_grad

    def forward(self, u, state=None, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (L, B, H) """
        u = u.permute(1,2,0) # [B, H, L]
        B,H,L = u.size()
        z = u

        if self.binary != 'GSU':
            u = self.LN(u.transpose(-2,-1)).transpose(-2,-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)
        if state is not None:
            u[...,0] = u[...,0] + state
        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)
        state = y[...,-1]

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        if self.binary == 'binary':
            y = self.dropout(LIFAct.apply(y, 0., 0., self.threshold, self.time_step, self.surro_grad))
        elif self.binary == 'GSU':
            y = self.dropout(y)
        else:
            y = self.dropout(self.activation(y))
        if self.binary != 'GSU':
            y = self.output_linear(y)
        else:
            y = y.permute(2,0,1).reshape(-1, self.h) # B*L,H
            delta_y = 0.15 * torch.max(abs(y), dim=1,keepdim=True).values
            delta_w = 0.15 * abs(self.w).max()
            tri_y = (abs(y) >= delta_y) * y.sign()
            tri_w = (abs(self.w) >= delta_w) * self.w.sign()
            a = torch.matmul(tri_y,self.w)
            y = torch.mul((a + self.b), torch.mm(y,tri_w) + self.c)
            y = y.reshape(L,B,H).permute(1,2,0) # B H L
            y = self.LN(y.transpose(-2,-1)).transpose(-2,-1)
            y = self.activation(y)

        if self.residual:
            y = z + y
        y = y.permute(2, 0, 1) # [B, H, L] ->[L, B, H]
        if self.return_state:
            return y, state # Return a dummy state to satisfy this repo's interface, but this can be modified
        else:
            return y

def setup_optimizer(model, lr, weight_decay, epochs, optim):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    if optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.0)
    elif optim == 'adam':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler