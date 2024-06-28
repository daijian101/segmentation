from inference.ga_net_inference import GANetInference
from inference.vanilla_inference import VanillaInference

inference_zoo = {
    'unet': VanillaInference,
    'unet_plus_plus': VanillaInference,
    'ga_net': GANetInference,
}
