import torch
import torch.nn as nn
import math

from ..trainer import SurrogateGradient as SG
from ..trainer.surrogate import PMSN_surrogate
from .base_neuron import BaseNeuron


class PMSN(BaseNeuron):
    def __init__(
        self,
        rest: float = 0.0,
        decay: float = 0.2,
        threshold: float = 0.3,
        neuron_num: int = 1,
        time_step: int = None,
        surro_grad: SG = None,
        exec_mode: str = "serial",
        recurrent: bool = False,
    ):
        super(PMSN, self).__init__(exec_mode=exec_mode)
        self.rest = rest
        self.decay = decay
        self.threshold = threshold
        self.neuron_num = neuron_num
        self.time_step = time_step
        self.recurrent = recurrent
        self.surro_grad= surro_grad
        if self.recurrent:
            self.recurrent_weight = nn.Linear(self.neuron_num, self.neuron_num)
        self.return_mem = False

        self.kernel = PMSN_kernel(self.neuron_num, N=4)
        self.D = nn.Parameter(torch.randn(self.neuron_num))
        self.thresh = torch.tensor([self.threshold])
        self.bn = nn.BatchNorm1d(self.neuron_num)
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"rest={self.rest}, "
            f"decay={self.decay}, "
            f"threshold={self.threshold}, "
            f"neuron_num={self.neuron_num}, "
            f"time_step={self.time_step}, "
            #f"surrogate_gradient=\"{self.surro_grad.func_name}\", "
            f"execution_mode=\"{self.exec_mode}\", "
            f"recurrent={self.recurrent}"
            f")"
        )

    def _serial_process(self, tx, state=None):
        if isinstance(state, tuple):
            tx[0,] = tx[0,] + state[0]
            return_state = True
        else:
            return_state = False
        step_num = tx.size(0) #(T,B,H)

        tx = self.bn(tx.view(-1, tx.size(-1))).view(step_num, -1, self.neuron_num)
        tx = tx.permute(1,2,0) #(B, H, T)
        # Compute SSM Kernel
        k = self.kernel(L=step_num) # (H T)

        # # Convolution
        k_f = torch.fft.rfft(k, n=2*step_num) # (H T)
        u_f = torch.fft.rfft(tx, n=2*step_num) # (B H T)
        _y = torch.fft.irfft(u_f*k_f, n=2*step_num)[..., :step_num] # (B H T)
        y = _y + (tx * self.D.unsqueeze(-1))
        # proposed reset mechanism
        ty = PMSN_surrogate.apply(y.relu(), self.thresh.to(tx.device))
        ty = ty.permute(2,0,1)

        if return_state:
            return ty, (_y[...,-1], None)
        elif self.return_mem:
            return y[-1,].unsqueeze(0)
        else:
            return ty

class PMSN_kernel(nn.Module):
    def __init__(self, d_model, N=4, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H).uniform_(0, 1) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)  # [H]

        self.log_dt = nn.Parameter(log_dt)
        diag_indices = torch.arange(N)
        sub_diag_indices = diag_indices[:-1] + 1
        super_diag_indices = diag_indices[1:] - 1

        S = torch.zeros(N,N)
        S[diag_indices, diag_indices] = -0.5
        S[diag_indices[:-1], sub_diag_indices] = 5. * ((torch.arange(N-1)+1))
        S[diag_indices[1:], super_diag_indices] = -5. * ((torch.arange(N-1)+1)) # 超对角线

        S_diag = torch.diagonal(S)
        A_real = (torch.mean(S_diag) * torch.ones_like(S_diag)).unsqueeze(0).repeat(H,1)

        A_imag, V = torch.linalg.eigh(S * -1j)  # [N; N,N]
        A_imag = A_imag.unsqueeze(0).repeat(H,1)

        self.mask = torch.zeros(N,N).cuda()
        self.mask[diag_indices, diag_indices] = 1
        self.mask[diag_indices[:-1], sub_diag_indices] = 1


        log_A_real = torch.log(-A_real)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

        B = torch.ones(H, N)
        C= torch.zeros(H,N)
        C[:,-1]=1
        Vinv=V.conj().T  # [N,N]
        CV= torch.einsum('hm,mn->hn',C+0j,V) # [H,N]
        VinvB=torch.einsum('mn,hn->hm',Vinv,B+0j) #[H,N]

        self.VinvB_real = nn.Parameter(VinvB.real)
        self.VinvB_imag = nn.Parameter(VinvB.imag)
        self.CV_real = nn.Parameter(CV.real)
        self.CV_imag = nn.Parameter(CV.imag)

    def forward(self, L):
        # u [B,H,L]
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)
        B = self.VinvB_real + 1j * self.VinvB_imag  #(H,N)
        C = self.CV_real + self.CV_imag * 1j

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H,1)
        A_bar = torch.exp(A*dt.unsqueeze(-1))  #[H N]
        B_bar = (A_bar-1)*B/A
        # Vandermonde multiplication
        logK = (A*dt.unsqueeze(-1)).unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)   e-At
        K = torch.exp(logK)
        KB = torch.einsum('hnl,hn->hnl',K,B_bar)  # e-At*B  # (H N L)
        CKB = torch.einsum('hn, hnl -> hl', C, KB).real #(H L)
        return CKB
