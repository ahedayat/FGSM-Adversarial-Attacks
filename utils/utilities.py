"""
Utilities of Project
"""

import os
import argparse
from yaml import parse
from PIL import Image

import torchvision


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(
        self,
        start_val=0,
        start_count=0,
        start_avg=0,
        start_sum=0
    ):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
            Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
            Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def save_tensor_as_image(
    saving_path,
    saving_name,
    tensor
):
    transform = torchvision.transforms.ToPILImage()
    pil_image = transform(tensor)

    pil_image.save(
        os.path.join(saving_path, saving_name)
    )


def get_args():
    """
    Argunments of:
        - `train.py`
        - `test.py`
    """
    parser = argparse.ArgumentParser(
        description='Arguemnt Parser of `Train` and `Evaluation` of deep neural network.')

    # Hardware
    parser.add_argument('-g', '--gpu', action='store_true', dest='gpu',
                        default=True, help='Use GPU')
    parser.add_argument('-w', '--num-workers', dest='num_workers', default=1,
                        type=int, help='Number of workers for loading data')

    # CNN Backbone
    parser.add_argument('--vgg-type', dest='vgg_type', default="vgg19",
                        type=str, help="Backbone Network: ['vgg19', 'vgg16', 'vgg13', 'vgg11']")
    parser.add_argument("--dropout", dest="dropout", default=0.5, type=float,
                        help="dropout probability for MyVGG network.")
    parser.add_argument('--teacher-type', dest='teacher_type', default="vgg19",
                        type=str, help="Backbone Network: ['vgg19', 'vgg16', 'vgg13', 'vgg11']")
    parser.add_argument("--teacher-dropout", dest="teacher_dropout", default=0.5, type=float,
                        help="dropout probability for teacher network.")

    # Data Path
    # - Train
    parser.add_argument('--train-data-path', dest='train_data_path', default="./datasets/cifar10",
                        type=str, help='train dataset base directory')
    # - Test
    parser.add_argument('--test-data-path', dest='test_data_path', default="./datasets/cifar10",
                        type=str, help='Test dataset base directory')

    # Model Parameters
    parser.add_argument("--feature-layer-index", dest="feature_layer_index", default=1, type=int,
                        help="feature layer index that is used to extract feature for resnet architecture. valid value: [1,2,3,4].")

    # Optimizer Parameters
    parser.add_argument("--optimizer", dest="optimizer", default="adam", type=str,
                        help="Optimization Algorithm")
    parser.add_argument("--num-epochs", dest="num_epochs", default=100, type=int,
                        help="number of total epochs to run")
    parser.add_argument("--batch-size", dest="batch_size", default=64, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")
    parser.add_argument("--lr", dest="lr", default=4.8,
                        type=float, help="learning rate")
    parser.add_argument("--momentum", dest="momentum", default=0.9, type=float,
                        help="momentum of SGD algorithm")
    parser.add_argument("--weight-decay", dest="weight_decay",  default=1e-6,
                        type=float, help="weight decay")

    # Attack Parameters
    parser.add_argument("--attack-mode", dest="attack_mode", default="data", type=str,
                        help="Attack mode: ['data', 'test'].")
    parser.add_argument("--fgsm-iteration", dest="fgsm_iteration", default=10, type=int,
                        help="numberr of iteration of FGSM algorithm")
    parser.add_argument("--fgsm-epsilon", dest="fgsm_epsilon",  default=0.01,
                        type=float, help="epsilon")

    # Sample Parameters
    parser.add_argument("--test-data-index", dest="test_data_index", default=0, type=int,
                        help="index of sample from testing dataset that is used for FGSM (in attack.py) or Predicting (in predict.py).")
    parser.add_argument("--train-data-index", dest="train_data_index", default=0, type=int,
                        help="index of sample from training dataset that is used for FGSM (in attack.py) or Predicting (in predict.py).")

    parser.add_argument("--sample-save-path", dest="sample_save_path", default="./predicted", type=str,
                        help="save path for FGSM results (in attack.py) or predicted samples (in predict.py).")

    # Saving Parameters
    parser.add_argument("--ckpt-save-path", dest="ckpt_save_path", type=str, default="./checkpoints",
                        help="Checkpoints address for saving")
    parser.add_argument("--ckpt-load-path", dest="ckpt_load_path", type=str, default=None,
                        help="Checkpoints address for loading")
    parser.add_argument("--teacher-ckpt-load-path", dest="teacher_ckpt_load_path", type=str, default=None,
                        help="Checkpoints address for teacher network")
    parser.add_argument("--ckpt-prefix", dest="ckpt_prefix", type=str, default="ckpt_",
                        help="Checkpoints prefix for saving a checkpoint")
    parser.add_argument("--ckpt-save-freq", dest="ckpt_save_freq", type=int, default=20,
                        help="Saving checkpoint frequency")
    parser.add_argument("--report-path", dest="report_path", type=str, default="./reports",
                        help="Saving report path")

    options = parser.parse_args()

    return options
