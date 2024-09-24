from jbag.models.network_weight_initialization import initialize_network
from jbag.models.unet import build_unet
from jbag.models.unet_plus_plus import build_unet_plus_plus
from jbag.models.utils import get_conv_op, get_norm_op, get_non_linear_op
from lazyConfig import Config

from models.ga_net_new import GANet


def build_ganet(network_config: Config):
    conv_op = get_conv_op(network_config.conv_dim)
    norm_op = get_norm_op(network_config.norm_op, network_config.conv_dim)
    non_linear_op = get_non_linear_op(network_config.non_linear)

    params = {'input_channels': network_config.input_channels,
              'num_classes': network_config.num_classes,
              'num_stages': network_config.num_stages,
              'num_features_per_stage': network_config.num_features_per_stage.as_primitive(),
              'conv_op': conv_op,
              'kernel_sizes': network_config.kernel_sizes.as_primitive(),
              'strides': network_config.strides.as_primitive(),
              'num_conv_per_stage_encoder': network_config.num_conv_per_stage_encoder.as_primitive(),
              'num_conv_per_stage_decoder': network_config.num_conv_per_stage_decoder.as_primitive(),
              'conv_bias': network_config.conv_bias,
              'norm_op': norm_op,
              'norm_op_kwargs': network_config.norm_op_kwargs,
              'non_linear': non_linear_op,
              'non_linear_kwargs': network_config.non_linear_kwargs,
              'non_linear_first': network_config.non_linear_first,
              }
    network = GANet(**params)

    initialize_network(network, network_config)

    return network


model_zoo = {
    'ganet': build_ganet,
    'unet++': build_unet_plus_plus,
    'unet': build_unet
}
