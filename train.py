import os
import utils as utils
from datetime import datetime, date


import torch

import nets as nets
import deeplearning as dl
import dataloaders as data


def save_report(df, backbone_name, saving_path):
    """
        Saving Output Report Dataframe that is returned in Training
    """
    _time = datetime.now()
    hour, minute, second = _time.hour, _time.minute, _time.second

    _date = date.today()
    year, month, day = _date.year, _date.month, _date.day

    report_name = "{}_{}_{}_{}_{}_{}_{}.csv".format(
        backbone_name, year, month, day, hour, minute, second)

    print("Saving Report at '{}'".format(
        os.path.join(saving_path, report_name)))

    df.to_csv(os.path.join(saving_path, report_name))


def _main(args):

    # Hardware
    cuda = True if args.gpu and torch.cuda.is_available() else False
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

    # Optimizer
    assert args.optimizer in [
        "sgd", "adam"], "Optimizer must be one of this items: ['sgd', 'adam']"

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning Rate Schedular
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='min',
                                                              patience=3,
                                                              threshold=0.9,
                                                              min_lr=1e-10,
                                                              verbose=False,
                                                              )

    # Loading Model
    if cuda:
        model = model.cuda()
    print("args.ckpt_load_path: ", args.ckpt_load_path)
    if args.ckpt_load_path is not None:
        print("**** Loading Model...")
        model, optimizer = nets.load(
            ckpt_path=args.ckpt_load_path, model=model, optimizer=optimizer)

    # Loss Function
    criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()

    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Teacher Network
    teacher = None
    if str(args.teacher_type).lower() != "none":
        teacher = nets.MyVGG(
            vgg_type=args.teacher_type,
            num_classes=10,
            pretrained=True,
            dropout=args.teacher_dropout,   # must be added to the args.
            freeze_backbone=False
        )

        if cuda:
            teacher = teacher.cuda()

        # Loading pre-trained 'MyVGG' for teacher network
        if args.teacher_ckpt_load_path is not None:
            print("**** Loading Teacher Model...")
            teacher, _ = nets.load(
                ckpt_path=args.teacher_ckpt_load_path, model=teacher, optimizer=None)

        teacher.freeze_backbone()
        # teacher.freeze_classifier()
        teacher.eval()

        if cuda:
            teacher = teacher.cuda()

    # Checkpoint Address
    saving_path, saving_prefix = args.ckpt_save_path, args.ckpt_prefix
    saving_checkpoint_freq = args.ckpt_save_freq

    # Training
    model, optimizer = dl.train(
        model=model,
        teacher=teacher,
        train_dataset=train_dataset,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        report_path=args.report_path,
        Tensor=Tensor,
        saving_checkpoint_path=saving_path,
        saving_prefix=f"{saving_prefix}{args.vgg_type}_{args.teacher_type}",
        saving_checkpoint_freq=saving_checkpoint_freq
    )

    file_name = f"{saving_prefix}{args.vgg_type}_{args.teacher_type}_final.ckpt"

    nets.save(
        file_path=saving_path,
        file_name=file_name,
        model=model,
        optimizer=optimizer
    )

if __name__ == "__main__":
    args = utils.get_args()
    _main(args)
