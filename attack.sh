# Architecture
vgg_type="vgg19"
dropout=0.5

# Pre-Trained Checkpoints
ckpt_load_path="./pretrained_models/pretrainedvgg19_none_final.ckpt"

# Data Path
train_data_path="/content/datasets/cifar10"
test_data_path="/content/datasets/cifar10"

# FGSM Parameters
attack_mode="data"
fgsm_iteration=1
fgsm_epsilon=0.05
# fgsm_epsilon=0.21

train_data_index=114
test_data_index=114

sample_save_path="./attacks"


python -W ignore attack.py \
        --gpu \
        --vgg-type $vgg_type \
        --dropout $dropout \
        --ckpt-load-path $ckpt_load_path \
        --train-data-path $train_data_path \
        --test-data-path $test_data_path \
        --attack-mode $attack_mode \
        --fgsm-iteration $fgsm_iteration \
        --fgsm-epsilon $fgsm_epsilon \
        --train-data-index $train_data_index \
        --test-data-index $test_data_index \
        --sample-save-path $sample_save_path
