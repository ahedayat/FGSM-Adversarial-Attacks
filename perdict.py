import os
import numpy as np
from datetime import datetime, date


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

import nets as nets
import utils as utils
import deeplearning as dl
import dataloaders as data


def get_output(model, sample, device):
    model.eval()
    sample = sample.to(device)
    sample = sample.unsqueeze(0)

    output = model(sample)
    init_pred = output.max(1, keepdim=True)[1]

    return init_pred.item()


def _main(args):

    # Hardware
    cuda = True if args.gpu and torch.cuda.is_available() else False
    device = torch.device("cuda") if cuda else torch.device("cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Datasets
    mean, std = (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)

    # Training Datasets
    train_dataset = data.Cifar10Classification(
        data_path=args.train_data_path,
        data_mode="train",
        data_download=True,
        input_normalization=True,
        mean=mean,
        std=std
    )

    # Testing Datasets
    test_dataset = data.Cifar10Classification(
        data_path=args.train_data_path,
        data_mode="test",
        data_download=True,
        input_normalization=True,
        mean=mean,
        std=std
    )

    # CNN Backbone
    model = nets.MyVGG(
        vgg_type=args.vgg_type,
        num_classes=10,
        pretrained=True,
        dropout=args.dropout,
        freeze_backbone=False
    )

    # Loading Model
    if args.ckpt_load_path is not None:
        print("**** Loading Model...")
        model, _ = nets.load(
            ckpt_path=args.ckpt_load_path, model=model, optimizer=None)
        print("**** Model is loaded successfully ****")

    if cuda:
        model = model.cuda()

    model.eval()

    train_sample_id = args.train_data_index
    test_sample_id = args.test_data_index

    datasets = (train_dataset, test_dataset)
    sample_ids = (train_sample_id, test_sample_id)
    modes = ("train", "test")

    cifar10_classes = train_dataset.get_classes()

    for (mode, _dataset, sample_id) in zip(modes, datasets, sample_ids):
        sample, target_id = _dataset[sample_id]

        target = cifar10_classes[target_id]

        predicted_id = get_output(model, sample, device)
        predicted = cifar10_classes[predicted_id]

        save_image(
            sample,
            os.path.join(args.sample_save_path,
                         f"predicted_{sample_id}_{target}_{predicted}.png")
        )


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
