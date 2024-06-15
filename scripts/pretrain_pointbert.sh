python  main.py \
    --wandb \
    --model ULIP_PointBERT \
    --lr 3e-3 \
    --pretrain_dataset_name s3dis \
    --output-dir ./outputs/reproduce_pointbert_1kpts \
    # --resume ./outputs/reproduce_pointbert_1kpts/only_scene_epoch10_checkpoint.pt