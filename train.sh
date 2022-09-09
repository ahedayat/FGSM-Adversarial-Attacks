
# Hardware
num_workers=2

# Architecture
vgg_type="vgg11"
dropout=0.2
teacher_type="vgg19"
teacher_dropout=0.2

# Data Path
train_data_path="/content/datasets/cifar10"
test_data_path="/content/datasets/cifar10"

# Optimizer Parameters
optimizer="adam"
num_epochs=20
batch_size=64

# Learning Rate 
lr=1e-3
momentum=0.9
weight_decay=5e-3

# Checkpoints Parameters
ckpt_save_path="./checkpoints"
ckpt_load_path="./pretrained_models/pretrainedvgg11_none_final.ckpt"
ckpt_prefix="pretrained"
ckpt_save_freq=25

teacher_ckpt_load_path="./pretrained_models/pretrainedvgg19_none_final.ckpt"

# Report
report_path="./reports_vgg11_vgg19"

python -W ignore train.py \
        --gpu \
        --num-workers $num_workers \
        --vgg-type $vgg_type \
        --dropout $dropout \
        --teacher-type $teacher_type \
        --teacher-dropout $teacher_dropout \
        --train-data-path $train_data_path \
        --test-data-path $test_data_path \
        --optimizer $optimizer \
        --num-epochs $num_epochs \
        --batch-size $batch_size \
        --lr $lr \
        --momentum $momentum \
        --weight-decay $weight_decay \
        --ckpt-save-path $ckpt_save_path \
        --ckpt-save-freq $ckpt_save_freq \
        --ckpt-prefix $ckpt_prefix \
        --report-path $report_path \
        --ckpt-load-path $ckpt_load_path \
        --teacher-ckpt-load-path $teacher_ckpt_load_path


