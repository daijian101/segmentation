import os
import traceback

import cavass.ops as cavass
import numpy as np
import torch
from jbag.checkpoint import load_checkpoint
from jbag.config import get_config
from jbag.io import read_txt_2_list, save_json, write_list_2_txt
from jbag.transforms import ToType
from jbag.transforms.normalization import ZscoreNormalization
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from cfgs.args import parser
from dataset.dataset import ImageDataset
from inference import inference_zoo
from materials.meas_stds_const import BCA_MEAN, BCA_STD
from models import model_zoo
from post_process import PostProcessTransformCompose, post_process_methods


def main():
    if 'inference_samples' in cfg:
        if isinstance(cfg.inference_samples, str):
            samples = read_txt_2_list(cfg.inference_samples)
        else:
            samples = cfg.inference_samples
    else:
        samples = [each[:-4] for each in os.listdir(cfg.IM0_path) if each.endswith('.IM0')]
        # samples = [each for each in os.listdir(cfg.IM0_path)]
        # samples = [each[:-4] for each in os.listdir(cfg.IM0_path) if each.find('-CT') != -1 or each.find('-CT-') != -1]

    if cfg.convert_json:
        # convert IM0 to json
        print('======Convert IM0 to JSON======')
        failure_samples = []
        for subject_name in tqdm(samples):
            json_file = os.path.join(cfg.volume_json_path, f'{subject_name}.json')
            if os.path.exists(json_file):
                continue
            # im0_file = os.path.join(cfg.IM0_path, subject_name, f'{subject_name}-CT.IM0")
            im0_file = os.path.join(cfg.IM0_path, f"{subject_name}.IM0")
            try:
                image_data = cavass.read_cavass_file(im0_file)
            except OSError:
                failure_samples.append(subject_name)
                traceback.print_exc()
                continue

            json_obj = {"data": image_data, "subject": subject_name}
            save_json(json_file, json_obj)

        if failure_samples:
            samples = [each for each in samples if each not in failure_samples]
            write_list_2_txt(os.path.join(cfg.result_cavass_path, "failure_samples.txt"), failure_samples)

    print("======Infer subjects======")

    json_samples = [f"{each}.json" for each in samples]

    tr_transforms = Compose([
        ToType(keys="data", dtype=np.float32),
        ZscoreNormalization(keys="data", mean=BCA_MEAN, std=BCA_STD)
    ])
    dataset = ImageDataset(json_samples,
                           cfg.volume_json_path,
                           transforms=tr_transforms)
    data_loader = DataLoader(dataset, 1)

    for target_label in tqdm(cfg.inference_labels):
        model_name = cfg.labels[target_label].model
        model = model_zoo[model_name](cfg.labels[target_label]).to(device)
        load_checkpoint(cfg.labels[target_label].checkpoint, model)
        inference_method = cfg.labels[target_label].inference_method if "inference_method" in cfg.labels[target_label] \
            else cfg.labels[target_label].model

        inference_method_extra_args = cfg.labels[target_label].inference_method_args if "inference_method_args" in \
                                                                                        cfg.labels[target_label] else {}

        if "models" in cfg:
            if model_name in cfg.models:
                if "batch_size" in cfg.models[model_name]:
                    inference_size = cfg.models[model_name].batch_size
        else:
            inference_size = cfg.batch_size
        inference_ist = inference_zoo[inference_method](model, inference_size, device,
                                                        **inference_method_extra_args)

        bim_save_path = os.path.join(cfg.result_cavass_path, f"{target_label}")

        trs = []
        if "post_process" in cfg.labels[target_label]:
            for each_transform in cfg.labels[target_label]["post_process"]:
                trs.append(post_process_methods[each_transform]())
        post_process_compose = PostProcessTransformCompose(trs)

        for batch in tqdm(data_loader):
            subject_name = batch["subject"][0]

            output_file_path = os.path.join(bim_save_path, f"{subject_name}_{target_label}.BIM")
            if os.path.exists(output_file_path):
                continue
            with autocast():
                result = inference_ist.infer_sample(batch)
            segmentation = result.cpu().numpy().astype(bool)

            segmentation = post_process_compose(segmentation)

            # im0_file = os.path.join(cfg.IM0_path, subject_name, f"{subject_name}-CT.IM0")
            im0_file = os.path.join(cfg.IM0_path, f"{subject_name}.IM0")

            # cavass.save_cavass_file(os.path.join(bim_save_path, subject_name, f"{subject_name}_{target_label}.BIM"),
            #                         segmentation, True, reference_file=im0_file)

            cavass.save_cavass_file(os.path.join(bim_save_path, f"{subject_name}_{target_label}.BIM"), segmentation, True, reference_file=im0_file)


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = get_config(args.cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cfg.gpus)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main()
