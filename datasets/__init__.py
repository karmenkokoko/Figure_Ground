from re import I
from . import KITTI_raw_multi
from .list_utils import *
from .validation import *

KITTI_Raw_Multi_KittiSPlit_Train = KITTI_raw_multi.KITTI_Raw_Multi_KittiSPlit_Train
KITTI_train = KITTI_raw_multi.SequenceFolder
Validation_Flow = ValidationFlow
Validation_Set = ValidationSet
Validation_mask = ValidationMask