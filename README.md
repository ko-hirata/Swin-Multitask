# Swin-Multitask
The codes for the work "Brain Hematoma Marker Recognition Using Multitask Learning: SwinTransformer and Swin-Unet"

## Abstract
This paper proposes a method MTL-Swin-Unet which is multi-task learning using transformers for classification and semantic segmentation. For sprious-correlation problems,
this method allows us to enhance the image representation with
two other image representations: representation obtained by
semantic segmentation and representation obtained by image reconstruction.
In our experiments, the proposed method outperformed in F-value measure than other classifiers when the test data included slices from the same patient (no covariance shift). Similarly, when the test data did not include slices
from the same patient (covariance shift setting), the proposed method
outperformed in AUC measure.


## Installation
Run the follwing code to set up Swin-Multitask and Swin-JointLearning.
```python
conda create -n SwinMTL python=3.8
conda activate SwinMTL
cd Swin-Multitask

pip install -r requirements.txt
```