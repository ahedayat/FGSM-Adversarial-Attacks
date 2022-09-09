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
import adverserial as adverserial
from adverserial.iterative_fgsm import IterativeFGSM


def get_output(model, sample, device):
    model.eval()
    sample = sample.to(device)
    sample = sample.unsqueeze(0)

    output = model(sample)
    init_pred = output.max(1, keepdim=True)[1]

    return init_pred.item()


def get_sample(model, dataset, index, device):
    sample, target = dataset[index]

    counter = 0

    pred = get_output(model, sample, device)
    while (counter < len(dataset)) and (pred != target):
        index = (index + 1) % len(dataset)
        sample, target = dataset[index]
        pred = get_output(model, sample, device)
        counter += 1

    return sample, target


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

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
    )

    dl.eval(
        model=model,
        eval_data_loader=test_data_loader,
        criterion=torch.nn.CrossEntropyLoss().cuda(),
        epoch=None,
        Tensor=Tensor,
        report_path=".",
        report_name=f"checking_attack_test"
    )

    if cuda:
        model = model.cuda()

    # Attack
    assert args.attack_mode in [
        "data", "test"], "'attack_mode' must be one of this elements: ['data', 'test']."

    if args.attack_mode == "data":

        train_sample_id = args.train_data_index
        test_sample_id = args.test_data_index

        assert train_sample_id < len(
            train_dataset), "Smaple index for training dataset is grater that number of data in training dataset."
        assert test_sample_id < len(
            test_dataset), "Smaple index for testing dataset is grater that number of data in testing dataset."

        fgsm = adverserial.IterativeFGSM(
            num_iterations=args.fgsm_iteration,
            epsilon=args.fgsm_epsilon,
            # loss=F.nll_loss,
            loss=nn.CrossEntropyLoss(),
            verbose=True
        )

        train_sample_data, train_sample_label = get_sample(
            model, train_dataset, train_sample_id, device)
        test_sample_data, test_sample_label = get_sample(
            model, test_dataset, test_sample_id, device)

        modes = ("train", "test")
        samples = (train_sample_data, test_sample_data)
        labels = (train_sample_label, test_sample_label)

        for (mode, sample, label) in zip(modes, samples, labels):

            # Saving Original Image
            # save_image(
            #     saving_path=args.fgsm_save_path,
            #     saving_name=f"original_{mode}_{label}.png",
            #     tensor=sample
            # )
            save_image(
                sample,
                os.path.join(args.sample_save_path,
                             f"original_{mode}_{label}.png")
            )

            perturbed = fgsm.attack(
                model=model,
                data=sample,
                target=label,
                device=device
            )

            perturbed_label = model(perturbed)

            perturbed = perturbed.squeeze()
            perturbed_label = perturbed_label.max(1, keepdim=True)[1]
            perturbed_label = perturbed_label.item()

            # Saving Perturbed Image
            save_image(
                perturbed,
                os.path.join(args.sample_save_path,
                             f"perturbed_{mode}_{perturbed_label}.png")
            )


            # Saving Noise
            noise = (perturbed.cpu() - sample.cpu()) / fgsm.alpha
            save_image(
              noise,
              os.path.join(args.sample_save_path, f"noise_{mode}_{label}_to_{perturbed_label}.png")
            )

    elif args.attack_mode == "test":

        modes = ("train", "test")
        datasets = (train_dataset, test_dataset)

        for epsilon in np.arange(0, args.fgsm_epsilon, 0.05):

            fgsm = adverserial.IterativeFGSM(
                num_iterations=args.fgsm_iteration,
                epsilon=epsilon,
                # loss=F.nll_loss,
                loss=nn.CrossEntropyLoss(),
                verbose=False
            )

            for (mode, dataset) in zip(modes, datasets):
                print(f"+-------------------------+")
                print(f"{mode} + Epsilon = {fgsm.alpha}")
                print(f"+-------------------------+")
                report, top1 = adverserial.IterativeFGSM.test(
                    iterative_fgsm=fgsm,
                    model=model,
                    dataset=dataset,
                    device=device
                )

                print("Accuracy: {:.3f}".format(top1.avg))

                report.to_csv(
                    os.path.join(args.sample_save_path, f"fgsm_{mode}_{fgsm.alpha*100}.csv")
                )

    else:
        exit()


if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
