from .registry import register_model, get_model_cls
from .attnet import (
    AttNet,
    attnet_xxs,
    attnet_xs,
    attnet_s,
    attnet_t1,
    attnet_t2,
    attnet_t3,
    attnet_t4,
)
from .fastervit import FasterViT
from .iformer import IFormer, iformer_t, iformer_s, iformer_m, iformer_l
from .lowformer import (
    LowFormer,
    lowformer_backbone_b0,
    lowformer_backbone_b1,
    lowformer_backbone_b2,
    lowformer_backbone_b3,
)
from .mlla import Mlla
from .mobilenet import MobileNetv3, mobilenetv3_large, mobilenetv3_small
from .partialformer import PartialFormer
from .reduceformer import (
    ReduceFormer,
    reduceformer_backbone_b1,
    reduceformer_backbone_b2,
    reduceformer_backbone_b3,
)
from .shvit import SHViT
from .vit import VisionTransformer
from .vssd import Vssd
