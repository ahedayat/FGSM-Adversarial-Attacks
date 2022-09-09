# Architecture
vgg_type="vgg19"
dropout=0.5

# Pre-Trained Checkpoints
ckpt_load_path="./pretrained_models/pretrainedvgg19_none_final.ckpt"

# Data Path
train_data_path="/content/datasets/cifar10"
test_data_path="/content/datasets/cifar10"


train_data_index=12345
test_data_index=4321

sample_save_path="./predicted"


python -W ignore predict.py \
        --gpu \
        --vgg-type $vgg_type \
        --dropout $dropout \
        --ckpt-load-path $ckpt_load_path \
        --train-data-path $train_data_path \
        --test-data-path $test_data_path \
        --train-data-index $train_data_index \
        --test-data-index $test_data_index \
        --sample-save-path $sample_save_path
