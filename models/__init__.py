from jbag.models.network_weight_initialization import initialize_network
from jbag.models.unet import build_unet

from models.ga_net import GANet
from models.unet_plus_plus_new import UNetPlusPlus


def build_unet_plus_plus(cfg):
    network = UNetPlusPlus(in_channels=1, out_channels=2)
    initialize_network(network, cfg)
    return network


def build_ga_net(cfg):
    return GANet()


model_zoo = {
    'ga_net': build_ga_net,
    'unet++': build_unet_plus_plus,
    'unet': build_unet
}
