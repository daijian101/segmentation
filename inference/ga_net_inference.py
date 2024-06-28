import torch

from inference.inference import Inference


class GANetInference(Inference):
    @torch.no_grad()
    def infer_batch(self, batch):
        output, _ = self.model(batch.to(self.device))
        output = torch.argmax(output, dim=1)
        return output
