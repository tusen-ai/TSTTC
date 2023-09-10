# TSTTC for Time-To-Contact estimation.

## Introduction
The TSTTC dataset is a large scale dataset for TTC estimation via monocular images. For more details, please refer to our paper on [Arxiv](https://arxiv.org/abs/2309.01539). To download the dataset and evaluate result on test set, please refer to the [TSTTC](https://open-dataset.tusen.ai/) webpage.

## Dataset Structure

The TSTTC dataset is organized into four main folders: `train`, `val`, `test`, and `nerf`, following the data structure outlined below:


<pre>
TSTTC Dataset
├── train
│ ├── {bag_stamp1}.pkl
│ ├── {bag_stamp1}
│ │ ├── cam1
│ │ │ ├── 16xxxx.jpg
│ │ │ ├── 16xxxx.jpg
│ │ │ └── ...
│ │ ├── cam3
│ │ │ ├── 16xxxx.jpg
│ │ │ ├── 16xxxx.jpg
│ │ │ └── ...
│ │ └── ...
│ └── ...
├── val
│ └── ...
├── test
│ └── ...
└── nerf
  └── ...
</pre>

Within each set, there are multiple bags identified by a unique bag stamp. Each bag consists of up to five camera subfolders: `cam1`, `cam3`, `cam4`, `cam8`, and `cam9`, except for the `nerf` part, which only contains `cam1`. The `cam1`, `cam3`, and `cam4` subfolders correspond to frontal view cameras with different focal lengths, while `cam8` and `cam9` represent backward view cameras. The images within each camera subfolder are named based on the timestamp when they were captured. Additionally, each bag is associated with an annotation file named `bag_stamp.pkl`.

## Annotation Format

The annotation files in the TSTTC dataset are formatted as a list of dictionaries. Each dictionary within the file represents the annotations for a specific camera, with the timestamp serving as the key and the frame annotations as the corresponding value. The frame annotation for each timestamp is represented as a list of objects. Each object is a dictionary containing the following properties:


<pre>
Annotation File (List of Dictionaries)
├── Dict 1
│ ├── Timestamp 1
│ │ ├── Object 1
│ │ │ ├── bag_stamp
│ │ │ ├── box2d
│ │ │ ├── cam_id
│ │ │ ├── id
│ │ │ ├── occ_ratio
│ │ │ ├── same_lane
│ │ │ ├── ts
│ │ │ └── ttc_imu (Not available in test set)
│ │ ├── Object 2
│ │ │ ├── ...
│ │ ├── ...
│ ├── Timestamp 2
│ │ ├── ...
│ ├── ...
├── Dict 2
│ ├── ...
├── ...
</pre>

Here is a detailed description of each property:

| Property        | Description                                                                                                                |
|-----------------|----------------------------------------------------------------------------------------------------------------------------|
| bag_stamp       | A string representing the bag stamp of the object, indicating its association with a particular bag within the dataset.    |
| box2d           | A list of floats representing the 2D bounding box of the object in the image. In (x1,y1,x2,y2) format and coordinates are normalized to 0~1 |
| cam_id          | An identifier for the camera to which the object belongs.                                                                 |
| id              | An integer representing the tracking ID of the object across all cameras.                                                |
| occ_ratio       | A float value denoting the occlusion ratio of the object in the 2D images.                                               |
| same_lane       | A boolean flag indicating whether the object is in the same lane as the ego vehicle.                                     |
| ts              | An integer representing the timestamp of the object.                                                                     |
| ttc_imu         | A float value representing the ground truth TTC of the object. Not available in the test set.         |

Please note that the property `ttc_imu` is not available in the test set, as mentioned in the dataset description. 


## Dataset API
For the usage of the dataset API, please refer to [TSTTCDatasetDemo](./data/TSTTCDatasetDemo.ipynb).

## Quick Start for Baseline Methods
<details>
<summary>Installation</summary>

Install TSTTC from source.
```shell
git clone https://github.com/tusen-ai/TSTTC
cd TSTTC
```

Create conda env.
```shell
conda create -n TSTTC python=3.7

conda activate TSTTC

pip install -r requirements.txt

cd cuda_ops

pip3 install -v -e .

cd ..
```
</details>


<details> 
<summary>Evaluation</summary>

### Evaluate the Pixel MSE.
```shell
python tools/eval_baseline.py --path [path_to_your_val_set]
```

### Evaluate the Deep Scale

Step1. Replace the valset_dir and valAnnoPath of the [exp_file](./exp/Deep_TTC.py) to the path of your own validation set 


Step2. Run the evaluation code
```shell
python tools/eval.py -f ./exp/Deep_TTC.py -c [path_to_your_weights] --path [path_to_your_val_set] -d 1 -b 8 --fp16 --box_level
```

</details>

<details>
<summary>Training</summary>
Step1. Replace the trainset_dir and trainAnnoPath of the [exp_file](./exp/Deep_TTC.py) to the path of your own training set

Step2. Run the training code
```shell
python tools/train.py -f ./exp/Deep_TTC.py -d 1 -b 8 --fp16 
```
</details>

## Acknowledgements
<details><summary> <b>Expand</b> </summary>

* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
</details>

## Reference
If you use TSTTC in your research, please cite our work:
```latex
@article{shi2023tsttc,
      title={TSTTC: A Large-Scale Dataset for Time-to-Contact Estimation in Driving Scenarios}, 
      author={Yuheng Shi and Zehao Huang and Yan Yan and Naiyan Wang and Xiaojie Guo},
      year={2023},
      journal={2309.01539},
}
```
