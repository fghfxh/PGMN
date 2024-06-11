
from dataclasses import dataclass
from simple_parsing import mutable_field
# from .models.mntdp import MNTDP_net
# from .models.fixed_extractor import OvAInn
# from .models.hat_1 import HAT
from .models.cnn_independent_experts import ExpertMixture
from .models.PGMN import PGMN_net
from .models.gate_network import HyperGate_Net
from .models.PGMN_components import PGMN_conv_block
from .models.cnn_soft_gated_lifelong_dynamic import CNNSoftGatedLLDynamic

@dataclass
class ModelOptions():
    PGMN: PGMN_net.Options = mutable_field(PGMN_net.Options)
    Module: PGMN_conv_block.Options = mutable_field(PGMN_conv_block.Options)
    Experts: ExpertMixture.Options = mutable_field(ExpertMixture.Options)
    SGNet: CNNSoftGatedLLDynamic.Options = mutable_field(CNNSoftGatedLLDynamic.Options)
    ############################
    Gate: HyperGate_Net.Options = mutable_field(HyperGate_Net.Options)

