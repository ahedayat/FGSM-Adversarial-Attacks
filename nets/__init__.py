"""
In this Package, some networks is implemented. 

Supported network:
    - MyVGG:
        * 'vgg19': 'VGG19' + 'Batch-Normalization'
        * 'vgg16': 'VGG16' + 'Batch-Normalization'
        * 'vgg13': 'VGG13' + 'Batch-Normalization'
        * 'vgg11': 'VGG11' + 'Batch-Normalization'
"""

from .my_vgg import MyVGG
from .my_net import MyNet

from .nets_utlis import load_net as load
from .nets_utlis import save_net as save


__version__ = '1.0.0'
__author__ = 'Ali Hedayatnia, M.Sc. Student of Artificial Intelligence @ University of Tehran'
