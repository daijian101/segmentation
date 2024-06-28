from jbag.models import UNetPlusPlus

from models.ga_net import GANet


def build_unet_plus_plus(cfg):
    return UNetPlusPlus(in_channels=1, out_channels=2)


def build_ga_net(cfg):
    return GANet()


model_zoo = {
    'ga_net": build_ga_net,
    "unet_plus_plus": build_unet_plus_plus,
}
