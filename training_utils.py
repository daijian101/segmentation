import torch
from jbag.samplers import GridSampler


@torch.no_grad()
def infer_3d_volume(input_data, batch_size, network, device):
    input_data = input_data['data']
    _, h, w, _ = input_data.shape
    input_data = input_data.permute((0, 3, 1, 2)).reshape((-1, h, w))
    batch_size = batch_size if batch_size < input_data.shape[1] else input_data.shape[1]
    patch_size = (batch_size, input_data.size(1), input_data.size(2))
    sampler = GridSampler(input_data, patch_size)

    output = []
    for patch in sampler:
        patch = patch.unsqueeze(dim=1)
        output_patch, _ = network(patch.to(device))
        output_patch = torch.argmax(output_patch, dim=1).to(torch.uint8).cpu()
        output.append(output_patch)
    output = sampler.restore(output)

    return output
