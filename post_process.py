import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import reconstruction


class PostProcessTransformCompose:
    def __init__(self, transforms):
        if isinstance(transforms, Filling):
            self.transforms = [transforms, ]
        self.transforms = transforms

    def __call__(self, data, **kwargs):
        for t in self.transforms:
            data = t(data, **kwargs)
        return data


class PostProcessTransform:
    ...


class LargestConnectedRegion(PostProcessTransform):
    def __call__(self, data, **kwargs):
        labeled_region = label(data)
        region_prob = regionprops(labeled_region)
        if region_prob:
            region_prob.sort(key=lambda x: x.area, reverse=True)
            new_segmentation = np.where(labeled_region == region_prob[0].label, True, False).astype(bool)
            return new_segmentation
        return data


class Filling(PostProcessTransform):

    def __call__(self, data, **kwargs):
        for i in range(data.shape[2]):
            image = data[..., i].astype(bool)
            seed = np.copy(image)
            seed[1:-1, 1:-1] = image.max()
            mask = image
            filled = reconstruction(seed, mask, method='erosion')
            data[..., i] = filled
        return data


post_process_methods = dict(filling=Filling, keeping_largest_region=LargestConnectedRegion)
