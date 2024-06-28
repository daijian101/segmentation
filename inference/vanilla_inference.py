import torch

from inference.inference import Inference


class VanillaInference(Inference):
    @torch.no_grad()
    def infer_batch(self, batch):
        output = self.model(batch.to(self.device))
        output = torch.argmax(output, dim=1)
        return output
