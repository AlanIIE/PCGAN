# PCGAN
PCGAN: Partition-Controlled Human Image Generation (AAAI 2019)

[Check out our paper](https://arxiv.org/)


### Requirment
* python3
* Numpy
* Scipy
* Skimage
* Pandas
* Tensorflow
* Keras
* Keras-contrib
* tqdm 

### Clone repository
```git clone https://github.com/AlanIIE/PCGAN.git```

### Training
In order to train a model:

1. You can download all the pre-processed data from https://drive.google.com/drive/folders/1Bm8vd1xHFg6vCf78S7RchNRsYPzvx60x?usp=sharing, extract and move the folders to ```./data```.

Then you can download the data annotations from https://drive.google.com/drive/folders/1AeABOFJzp8L7HTm9r78_wdMj2FraGIML?usp=sharing and move the annotation files to ```./data```.
Go to step 4.


1.1 Download market dataset https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view. Put it in data folder. Rename this folder to data/market-dataset. Rename bounding_box_test and bounding_box_train with test and train.


1.2 Download [deep fasion dataset in-shop clothes retrival benchmark](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html). Move img/ to data folder and rename it fasion/. Download key-point estimations from (https://yadi.sk/d/suymftBy3S7oKD) for fasion. Run script ```split_fasion_data.py``` in data/ folder. Go to the step 3.

1.3 Download [COCO2017](http://cocodataset.org/#download).

1.4 Download [LIP](https://github.com/lemondan/HumanParsing-Dataset).

1.5 Pre-procession:

Deploy the [Mask-rcnn](https://github.com/matterport/Mask_RCNN) in ```./```.

1.5.1 For COCO2017 and LIP:

Copy ```preprocess/cut_with_annotations.ipynb``` to ```./Mask_RCNN/samples/coco/```.
Copy the COCO2017 dataset to ```./COCO2017/```.
Run ```cut_with_annotations.ipynb``` to cut the human body images to 128x64 and save the ground-truth masks.
The LIP can be done in the same way.

1.5.2 For market and DeepFasion:

Copy ```preprocess/maskrcnn.ipynb``` to ```./Mask_RCNN/samples/```.
Copy the market dataset to ```./market-dataset/```.
Run ```maskrcnn.ipynb``` to compute the human body mask.

2. Download pose estimator (conversion of this https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) [pose_estimator.h5](https://yadi.sk/d/blgmGpDi3PjXvK). Launch ```python compute_cordinates.py.``` It will compute human keypoints. Alternativlly you can download key points estimations from (https://yadi.sk/d/suymftBy3S7oKD).
3. Create pairs dataset with ```python create_pairs_dataset.py```. It define pairs for training.
4. Run ```python train.py``` (see list of parameters in cmd.py)
For example:
#### Market
##### R-mask
```
python3 train.py --output_dir output/Rmask --checkpoints_dir output/Rmask --dataset market --l1_penalty_weight 0.01 --batch_size 24 --number_of_epochs 90 --number_of_batches 500 --checkpoint_ratio 30 --use_mask 1
```

##### Full
```
python3 train.py --output_dir output/full --checkpoints_dir output/full --dataset market --l1_penalty_weight 0.01 --batch_size 24 --number_of_epochs 90 --number_of_batches 500 --checkpoint_ratio 30 --use_mask 1
```

#### Fashion
##### R-mask
```
python3 train.py --output_dir output/Rmask --checkpoints_dir output/Rmask --dataset fasion --l1_penalty_weight 0.01 --batch_size 4 --number_of_epochs 90 --number_of_batches 500 --checkpoint_ratio 30 --use_mask 1
```

##### Full
```
python3 train.py --output_dir output/full --checkpoints_dir output/full --dataset fasion --l1_penalty_weight 0.01 --batch_size 4 --number_of_epochs 90 --number_of_batches 500 --checkpoint_ratio 30 --use_mask 1
```


### Testing
0. Download checkpoints (https://drive.google.com/drive/folders/1xAzxIjLE8mWqOiueuKXK0A-fY0miKYyl?usp=sharing).
1. Run ```python test.py --generator_checkpoint path/to/generator/checkpoint``` (and same parameters as in train.py. You can test on COCO2017 and LIP using the model trained on market). It generate images and compute inception score, SSIM score and their masked versions.
2. To compute FID score, ```cd gan``` and run ```python3 fid.py ../output/fasion_maskrcnn_result_imgs/ ../output/fasion_maskrcnn_train_imgs/ -i /tmp/imagenet/ --gpu 0```

Citation:

```
@InProceedings{Dong_2019_AAAI,
author = {Dong Liang, Rui Wang, Xiaowei Tian, Cong Zou},
title = {PCGAN: Partition-Controlled Human Image Generation},
booktitle = {AAAI},
year = {2019}
}
```
