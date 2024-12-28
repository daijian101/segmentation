from abc import ABC, abstractmethod

import torch
from jbag.samplers import GridSampler


class Inference(ABC):
    def __init__(self, model, inference_size, device):
        self.model = model
        self.inference_size = inference_size
        self.device = device
        self.model.eval()

    def infer_sample(self, data):
        image = data['data']
        image = image.squeeze()
        if len(image.shape) == 2:
            image = torch.unsqueeze(image, dim=2)
        image = image.permute((2, 0, 1))
        batch_size = self.inference_size if self.inference_size <= image.shape[0] else image.shape[0]
        patch_size = (batch_size, image.size(1), image.size(2))
        sampler = GridSampler(image, patch_size)
        output = []
        for patch in sampler:
            patch = patch.unsqueeze(1)
            output_patch = self.infer_batch(patch)
            output.append(output_patch.to(torch.uint8).cpu())
        output = sampler.restore(output)
        output = output.permute(1, 2, 0)
        return output

    @abstractmethod
    def infer_batch(self, batch):
        ...
