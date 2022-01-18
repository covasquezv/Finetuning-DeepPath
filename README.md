# Fine tuning classification

## Organization     

``` bash
├── src                                <- Source code
    ├── scripts                        <- Main scripts
    │   └── fine_tuning                <- Fine tuning scripts
    │       └── fine_tuning.py
    │   
    └── lib                            <- Util scripts
        └── training                   <- Training utils (+ fine tuning)
            └── utils.py               <- Updated script

``` 

## Usage

``` bash
python src.scripts.fine_tuning.fine_tuning (...)
```
Performs fine tuning of the input model

``` bash

Parameters
-------------
patches_df_path: str
    DataFrame with training patches.
images_dir: str
    Patches folder.
min_threshold: float 
    Tissue proportion threshold.
base_model_name: str
    Selected model. Choices: resnet50, inception-v3, vgg-16
x_col: str
    Input column. Default: Filepath
y_col: str 
    Label with ground truth label. It changes according to model e.g Label_normal_vs_tumour. Default: Label
epochs: int
    Number of training epochs.
batch_size: int 
    Number of samples in the training batch. Default: 32
history_save_dir: str
    Logs folder 
model_save_dir: str 
    Fine tuned model folder.
random_seed: int
    Seed for reproducibility. Default: 0
rotation_range: int
    Range of rotation angle for data augmentation. Default: 10
width_shift_range: float
    Range of width shift for data augmentation. Default: 0.1
height_shift_range: float 
    Range of height shift for data augmentation. Default: 0.1 	
zoom_range: float 
    Zoom range for for data augmentation. Default: 0.1
horizontal_flip: bool
    True for horizontal flip in data augmentation. Default: True
vertical_flip:bool 
    True for vertical flip in data augmentation.
balanced_weights: bool 
    True for balance weight according to amount of data by class. Default: True
previous_checkpoint: str 
    Path to previous weights. E.g ./weights/inception-v3_test_1.h5
df_patches_val: str 
    DataFrame with validation patches.

Returns
-------------
None.

Outputs
-------------
Saved fine tuned weights.

```
