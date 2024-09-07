from jbag.models.unet import build_unet
from jbag.models.unet_plus_plus import build_unet_plus_plus

from models.ga_net import GANet


def build_ga_net(cfg):
    return GANet()


model_zoo = {
    'ga_net': build_ga_net,
    'unet++': build_unet_plus_plus,
    'unet': build_unet
}
