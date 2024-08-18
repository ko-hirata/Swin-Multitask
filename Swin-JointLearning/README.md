# SwinJointLearning
This code performs joint learning of segmentation and classification tasks. It utilizes parameters learned from the segmentation task for the classification task. Prior semantic segmentation with Swin-Unet is required.

![Joint-SwinTransformer](https://github.com/user-attachments/assets/44ac8487-cbae-4d2d-91e4-b7c3511d0a55)


## Pre-trained models
Create the following directory and put the pre-trained SwinTransformer model in it. You can download the pre-trained model [here](https://drive.google.com/file/d/1Qn2gXIRsUC6_XZq0oN6FnTjnGYV0bK6T/view?usp=sharing).
```python
mkdir pretrain_ckpt
```

## Train
```python
python train.py --root_dir DATA_DIR --output_dir OUT_DIR --ckpt_path_rep CKPT_PATH_LEARNED_BY_SEG --max_epochs 600 --batch_size 128 --base_lr 0.004 --img_size 224 --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml --cfg_rep ./configs/swin_tiny_patch4_window7_224_lite.yaml
```

## Test
```python
python test.py --root_dir DATA_DIR --output_dir OUT_DIR --ckpt_path CKPT_PATH --ckpt_path_rep CKPT_FILE_LEARNED_BY_SEG --img_size 224 --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml --cfg_rep ./configs/swin_tiny_patch4_window7_224_lite.yaml
```