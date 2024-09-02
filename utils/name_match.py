from agents.gdumb import Gdumb
from agents.agem import AGEM
from agents.clser import CLSER
from agents.exp_replay import ExperienceReplay

from agents.ewc_pp import EWC_pp
from agents.er_ace import ERACE
from agents.er_ace_lipschitz import ERACE_Lipschitz
from agents.gem import GEM
from agents.cndpm import Cndpm
from agents.derpp import DERPP
from agents.lwf import Lwf
from agents.icarl import Icarl
from agents.pcr import ProxyContrastiveReplay
from agents.pcr_m import ProxyContrastiveReplay_m
from agents.pcr_swav import ProxyContrastiveReplay_swav
from agents.pcr_sub import ProxyContrastiveReplay_sub
from agents.pcr_ca import ProxyContrastiveReplay_ca
from agents.scr import SupContrastReplay
from agents.superposition import Superposition
from agents.superpcr import SuperPCR
from continuum.dataset_scripts.bird100 import BIRD100
from continuum.dataset_scripts.cifar100 import CIFAR100
from continuum.dataset_scripts.cifar10 import CIFAR10
from continuum.dataset_scripts.core50 import CORE50
from continuum.dataset_scripts.food100 import FOOD100
from continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from continuum.dataset_scripts.openloris import OpenLORIS
from continuum.dataset_scripts.places100 import PLACES100
from continuum.dataset_scripts.tinyimagenet import TINYIMAGENET

from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.mir_retrieve import MIR_retrieve, Inverse_MIR_retrieve, Uncertainty_retrieve, Hardness_retrieve, MIX_retrieve
from utils.buffer.gss_greedy_update import GSSGreedyUpdate, InverseGSSGreedyUpdate
from utils.buffer.aser_retrieve import ASER_retrieve
from utils.buffer.aser_update import ASER_update
from utils.buffer.sc_retrieve import Match_retrieve
from utils.buffer.mem_match import MemMatch_retrieve

data_objects = {
    'bird100': BIRD100,
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'core50': CORE50,
    'food100': FOOD100,
    'mini_imagenet': Mini_ImageNet,
    'openloris': OpenLORIS,
    'places100': PLACES100,
    'tinyimagenet': TINYIMAGENET,
}

agents = {
    'AGEM': AGEM,
    'CLSER': CLSER,
    'DERPP': DERPP,
    'ER': ExperienceReplay,
    'ER_ACE': ERACE,
    'ER_ACE_L': ERACE_Lipschitz,
    'EWC': EWC_pp,
    'GEM': GEM,
    'CNDPM': Cndpm,
    'LWF': Lwf,
    'ICARL': Icarl,
    'GDUMB': Gdumb,
    'SCR': SupContrastReplay,
    'SUPER': Superposition,
    'SuperPCR': SuperPCR,
    'PCR': ProxyContrastiveReplay,
    'PCR_m': ProxyContrastiveReplay_m,
    'PCR_swav': ProxyContrastiveReplay_swav,
    'PCR_ca': ProxyContrastiveReplay_ca,
    'PCR_sub': ProxyContrastiveReplay_sub,
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'IMIR': Inverse_MIR_retrieve,
    'random': Random_retrieve,
    'ASER': ASER_retrieve,
    'match': Match_retrieve,
    'mem_match': MemMatch_retrieve,
    'UCR': Uncertainty_retrieve,
    'HDR': Hardness_retrieve,
    'MIX': MIX_retrieve,
}

update_methods = {
    'random': Reservoir_update,
    'GSS': GSSGreedyUpdate,
    'IGSS': InverseGSSGreedyUpdate,
    'ASER': ASER_update
}

