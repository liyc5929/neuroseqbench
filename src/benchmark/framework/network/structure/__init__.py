from .module import MergeDimension, SplitDimension, ANNSequential, Permute
from .base_architecture import BaseArchitecture
from .ff_snn import SpikingNet
from .lm_snn import LMSNN
from .dcls_delays import DCLS_Delays
from .lstm import LSTMNet, LMLSTM
from .tcn import TCN, LMTCN
from .transformer import TransformerNet, LMTransformer
from .spike_driven_transformer import SpkTransformerNet, LMSpkTransformer
