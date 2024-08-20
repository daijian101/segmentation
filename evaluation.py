import os
import nibabel
import numpy as np
from cavass.ops import read_cavass_file
from cavass.window_transform import cavass_soft_tissue_window
from jbag import MetricSummary
from medpy.metric import dc
from tqdm import tqdm
from jbag.image import overlay
from tensorboardX import SummaryWriter

from visualization import get_tensorboard_3d_images

output1 = '/data/dj/nnUNet_results_nnunet/Dataset001_Dphm/UNetPlusPlusTrainer__nnUNetPlans__2d/fold_0/validation'

target_path = '/data/dj/data/bca/cavass_data/Dphm'

image_path = '/data/dj/data/bca/cavass_data/images'

vis_log_path = '/data/dj/tmp/nnunet_dphm'

vis_log = SummaryWriter(vis_log_path)


dice = MetricSummary(dc)
for each in tqdm(os.listdir(output1)):
    if not each.endswith('.nii.gz'):
        continue
    subject_name = each[:-7]

    target_file = os.path.join(target_path, f'{subject_name}.BIM')

    input_data = nibabel.load(os.path.join(output1, each)).get_fdata()

    target = read_cavass_file(target_file)

    dice_score = dice.update(input_data, target)
    print(dice_score)

    # im0_file = os.path.join(image_path, f'{subject_name}.IM0')
    #
    # image = read_cavass_file(im0_file)
    #
    # image = cavass_soft_tissue_window(image)
    #
    # mask = input_data.astype(np.uint8)
    # image = get_tensorboard_3d_images(image, mask)
    #
    # vis_log.add_images(f'{subject_name}_{dice_score}', image)

print(dice.mean())

vis_log.close()
