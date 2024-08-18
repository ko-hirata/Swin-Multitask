# SwinMultitask
The code performs multi-task learning of classification, semantic segmentation and image reconstruction using a SwinTransformer-based network.

![MTL-Swin-Unet](https://github.com/user-attachments/assets/f477b424-1d69-48ab-a01f-3e15b33ae0ce)

## Pre-trained models
Create the following directory and put the pre-trained SwinTransformer model in it. You can download the pre-trained model [here](https://drive.google.com/file/d/1Qn2gXIRsUC6_XZq0oN6FnTjnGYV0bK6T/view?usp=sharing).
```python
mkdir pretrain_ckpt
```

## Train
```python
python train.py --root_dir DATA_DIR --output_dir OUTPUT_DIR --max_epochs 600 --batch_size 64 --base_lr 0.02 --img_size 224 --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml
```

## Test
The evaluation is performed on classification only. The result and GradCAM outputs are saved in "OUTPUT_DIR/inference_gradcam".

```python
python test.py --root_dir DATA_DIR --output_dir OUTPUT_DIR --ckpt_path CKPT_PATH --img_size 224 --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml
```

## Test 3tasks
The evaluation is performed on classification and segmentation. The predicted results of segmentation and image reconstruction are saved in "OUTPUT_DIR/inference_3tasks".

```python
python test_3tasks.py --root_dir DATA_DIR --output_dir OUTPUT_DIR --ckpt_path CKPT_PATH --img_size 224 --cfg ./configs/swin_tiny_patch4_window7_224_lite.yaml
```