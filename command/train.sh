# MSCOCO14 ResNet101 448
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-R101-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output './output/ResNet_448_MSCOCO/bs128work1' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9997 \
--seed 1 \
--gpus 0,1,2,3


# MSCOCO14 ResNet101 448 bs 64
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-R101-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone resnet101 --dataname coco14 --batch-size 64 --print-freq 400 \
--output './output/ResNet_448_MSCOCO14/' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 5e-5 --optim AdamW --pretrained \
--num_class 80 --img_size 448 --weight-decay 5e-3 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9998 \
--seed 1 \
--gpus 0,1,2,3

# MSCOCO14 ResNet101 512
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-R101-512' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone resnet101 --dataname coco14 --batch-size 128 --print-freq 400 \
--output './output/ResNet_512_MSCOCO/bs128work1' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 80 --img_size 512 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9997 \
--gpus 0,1,2,3

# MSCOCO14 ResNet101 576
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-R101-576' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone resnet101 --dataname coco14 --batch-size 64 --print-freq 400 \
--output './output/ResNet_576_MSCOCO/bs64work1' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 2 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 80 --img_size 576 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9997 \
--gpus 0,1,2,3

# MSCOCO14 ResNet101 576
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-R101-576' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone resnet101 --dataname coco14 --batch-size 64 --print-freq 400 \
--output './output/ResNet_576_MSCOCO/bs64work2' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 2 --dtgfl --loss_clip 0 \
--epochs 80 --lr 5e-5 --optim AdamW --pretrained \
--num_class 80 --img_size 576 --weight-decay 5e-3 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9998 \
--gpus 0,1,2,3

# MS-COCO TResL_V1 448
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-TResL-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone tresnetl --dataname coco14 --batch-size 128 --print-freq 400 \
--output './output/TResNetL_448_COCO/bs128work2' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 --orid_norm \
--hidden_dim 2432 --dim_feedforward 2432 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9997 \
--gpus 0,1,2,3


# MS-COCO TResL_V1 448 with mixup for image
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-TResL-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone tresnetl --dataname coco14 --batch-size 128 --print-freq 400 \
--output './output/TResNetL_448_COCO/bs128_mixup_work2' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 --orid_norm \
--hidden_dim 2432 --dim_feedforward 2432 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9997 \
--seed 634 --mix_up \
--gpus 0,1,2,3

# MS-COCO TResL_22k 448
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-TResL_22k-448' \
--backbone tresnetl_v2 --dataname coco14 --batch-size 112 --print-freq 400 \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--output './output/TResNetV2_448_COCO/best' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 80 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 --orid_norm \
--hidden_dim 2048 --dim_feedforward 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp \
--ema-decay 0.9997 \
--seed 1 \
--gpus 0,1,2,3



# NUSWIDE TResL_V1 448
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-TResL-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone tresnetl --dataname nus_wide --batch-size 128 --print-freq 400 \
--output './output/TResNetL_448_WIDE/bs128work1' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 1 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 81 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 --orid_norm \
--hidden_dim 2432 --dim_feedforward 2432 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--keep_other_self_attn_dec \
--keep_first_self_attn_dec \
--ema-decay 0.9997 \
--gpus 0,1,2,3

# NUSWIDE ResNet101 448
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-R101-448' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone resnet101 --dataname nus_wide --batch-size 128 --print-freq 400 \
--output './output/ResNet_448_NUSWIDE/bs128work1' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 81 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9997 \
--gpus 0,1,2,3

# NUSWIDE TResL_22k 448
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-TResL_22k-448' \
--backbone tresnetl_v2 --dataname nus_wide --batch-size 128 --print-freq 400 \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--output './output/TResNetV2_448_NUSWIDE/best' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 1 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 81 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 --orid_norm \
--hidden_dim 2048 --dim_feedforward 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp \
--ema-decay 0.9997 \
--keep_other_self_attn_dec \
--keep_first_self_attn_dec \
--seed 1 \
--gpus 0,1,2,3

# NUSWIDE_ASL TResL_22k 448
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-TResL_22k-448' \
--backbone tresnetl_v2 --dataname nus_wide_asl --batch-size 128 --print-freq 400 \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--output './output/TResNetV2_448_NUSWIDE/best' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 1 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim Adam_twd --pretrained \
--num_class 81 --img_size 448 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 --orid_norm \
--hidden_dim 2048 --dim_feedforward 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp \
--ema-decay 0.9997 \
--keep_other_self_attn_dec \
--keep_first_self_attn_dec \
--seed 1 \
--gpus 0,1,2,3

# VG500 TResL_22k 576
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-TResL_22k-576' \
--backbone tresnetl_v2 --dataname vg500 --batch-size 64 --print-freq 400 \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--output './output/TResNetV2_576_VG500/bs64_best' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 0 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 500 --img_size 576 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length -1 --orid_norm \
--hidden_dim 2048 --dim_feedforward 2048 \
--enc_layers 1 --dec_layers 2 --nheads 4 \
--early-stop --amp \
--ema-decay 0.9997 \
--gpus 0,1,2,3


# NUSWIDE ResNet101 576
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d \
main_mlc.py -a 'Q2L-R101-576' \
--dataset_dir '/media/mlldiskSSD/MLICdataset' \
--backbone resnet101 --dataname vg500 --batch-size 64 --print-freq 400 \
--output './output/ResNet_576_VG500/bs64work1' \
--world-size 1 --rank 0 --dist-url tcp://127.0.0.1:3716 \
--gamma_pos 0 --gamma_neg 4 --dtgfl --loss_clip 0 \
--epochs 80 --lr 1e-4 --optim AdamW --pretrained \
--num_class 500 --img_size 576 --weight-decay 1e-2 \
--cutout --n_holes 1 --cut_fact 0.5 --length 224 \
--hidden_dim 2048 --dim_feedforward 8192 \
--enc_layers 1 --dec_layers 2 --nheads 4 --position_embedding v2 \
--early-stop --amp \
--ema-decay 0.9997 \
--gpus 0,1,2,3
