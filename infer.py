import os
import traceback

import cavass.ops as cavass
import numpy as np
import torch
from jbag.config import load_config
from jbag.io import read_txt2list, save_json, write_list2txt, read_json
from jbag.model_weights import load_weights
from jbag.mp import fork
from jbag.transforms import ToType
from jbag.transforms.normalization import ZscoreNormalization
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from cfgs.args import parser
from dataset.dataset import ImageDataset
from inference import inference_zoo
from models import model_zoo
from post_process import PostProcessTransformCompose, post_process_methods


def convert2json(study_index, failure_samples):
    json_file = os.path.join(cfg.volume_json_dir, f'{study_index}.json')
    if os.path.exists(json_file):
        return

    # Declare directory structure
    # im0_file = os.path.join(cfg.IM0_dir, study_index, f'{study_index}-CT.IM0')
    im0_file = os.path.join(cfg.IM0_dir, f'{study_index}.IM0')
    try:
        image_data = cavass.read_cavass_file(im0_file)
    except OSError:
        failure_samples.append(study_index)
        traceback.print_exc()
        return

    json_obj = {'data': image_data, 'subject': study_index}
    save_json(json_file, json_obj)


def main():
    if 'inference_samples' in cfg:
        if isinstance(cfg.inference_samples, str):
            studies = read_txt2list(cfg.inference_samples)
        else:
            studies = cfg.inference_samples
    else:
        # Declare directory structure
        studies = [each[:-4] for each in os.listdir(cfg.IM0_dir)]
        # studies = [each for each in os.listdir(cfg.IM0_dir)]
        # studies = [each[:-4] for each in os.listdir(cfg.IM0_dir) if each.find('-CT') != -1 or each.find('-CT-') != -1]

    if cfg.convert_json:
        # convert IM0 to json
        print('======Convert IM0 to JSON======')
        failure_samples = []
        params = []
        for sample in studies:
            params.append((sample, failure_samples))

        fork(convert2json, 8, params)

        if failure_samples:
            studies = [each for each in studies if each not in failure_samples]
            write_list2txt(os.path.join(cfg.output_cavass_dir, 'failure_samples.txt'), failure_samples)

    print('======Infer subjects======')

    json_samples = [f'{each}.json' for each in studies]

    for target_label in tqdm(cfg.inference_labels):
        data_property = read_json(cfg.labels[target_label].data_property)

        tr_transforms = Compose([
            ToType(keys='data', dtype=np.float32),
            ZscoreNormalization(keys='data', mean=data_property['intensity_mean'],
                                std=data_property['intensity_std'],
                                lower_bound=data_property['intensity_0_5_percentile'],
                                upper_bound=data_property['intensity_99_5_percentile']),
        ])

        dataset = ImageDataset(json_samples,
                               cfg.volume_json_dir,
                               transforms=tr_transforms)
        data_loader = DataLoader(dataset, 1)

        model_name = cfg.labels[target_label].model
        network_config = load_config(cfg.labels[target_label].network_config)
        model = model_zoo[model_name](network_config).to(device)
        load_weights(cfg.labels[target_label].pretrained_weights, model)
        inference_method = cfg.labels[target_label].inference_method if 'inference_method' in cfg.labels[target_label] \
            else cfg.labels[target_label].model

        inference_method_extra_args = cfg.labels[target_label].inference_method_args if 'inference_method_args' in \
                                                                                        cfg.labels[target_label] else {}

        if 'batch_size' in cfg.labels[target_label]:
            batch_size = cfg.labels[target_label].batch_size
        else:
            batch_size = cfg.batch_size
        inference_ist = inference_zoo[inference_method](model, batch_size, device,
                                                        **inference_method_extra_args)

        bim_save_path = os.path.join(cfg.output_cavass_dir, f'{target_label}')

        trs = []
        if 'post_process' in cfg.labels[target_label]:
            for each_transform in cfg.labels[target_label]['post_process']:
                trs.append(post_process_methods[each_transform]())
        post_process_compose = PostProcessTransformCompose(trs)

        for batch in tqdm(data_loader):
            study_index = batch['subject'][0]

            output_file_path = os.path.join(bim_save_path, f'{study_index}_{target_label}.BIM')
            if os.path.exists(output_file_path):
                continue
            with autocast():
                result = inference_ist.infer_sample(batch)
            segmentation = result.cpu().numpy().astype(bool)

            segmentation = post_process_compose(segmentation)

            # Declare directory structure
            # im0_file = os.path.join(cfg.IM0_dir, study_index, f'{study_index}-CT.IM0')
            im0_file = os.path.join(cfg.IM0_dir, f'{study_index}.IM0')

            cavass.save_cavass_file(os.path.join(bim_save_path, f'{study_index}_{target_label}.BIM'), segmentation,
                                    True, copy_pose_file=im0_file)


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = load_config(args.cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cfg.gpus)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    main()
