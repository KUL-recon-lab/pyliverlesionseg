from . import architectures
from . import components
from . import sampling
from .general import *
from .CNN_liver_lesion_seg_CT_MR_functions import *

# needed to derive version number from git tags
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
