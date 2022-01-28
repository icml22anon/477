## Neighborhood-aware Geometric Encoding Network for Point Cloud Registraion

This is an anonymous repository under review by the International Conference on Machine Learning (ICML) 2022. Do not distribute.

## Environments

All experiments were run on a RTX 3090 GPU with an  Intel 8255C CPU at 2.50GHz CPU.  Dependencies can be installed using 

```
pip install -r requirements.txt
```

## Compile python bindings and Reconfigure

```
# Compile

cd NgeNet/cpp_wrappers
sh compile_wrappers.sh
```

## 1. 3DMatch and 3DLoMatch

### dataset

We adopted the 3DMatch and 3DLoMatch provided from PREDATOR, and download it from [here](https://share.phys.ethz.ch/~gsg/pairwise_reg/3dmatch.zip) [**5.17G**].
Unzip it, then we should get the following directories structure:

``` 
| -- indoor
    | -- train (#82, cats: #54)
        | -- 7-scenes-chess
        | -- 7-scenes-fire
        | -- ...
        | -- sun3d-mit_w20_athena-sc_athena_oct_29_2012_scan1_erika_4
    | -- test (#8, cats: #8)
        | -- 7-scenes-redkitchen
        | -- sun3d-home_md-home_md_scan9_2012_sep_30
        | -- ...
        | -- sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
```

### train

```
# Reconfigure configs/threedmatch.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the 3dMatch dataset.

cd NgeNet
python train.py configs/threedmatch.yaml

# note: The code `torch.cuda.empty_cache()` in `train.py` has some impact on the training speed.
# You can remove it or change its postion according to your GPU memory. 
```

### evaluate and visualize

```
# Reconfigure configs/threedmatch.yaml by updating the following values.
# checkpoint: your_ckpt_path
# saved_path: your_saved_path

cd NgeNet
python eval_3dmatch.py --benchmark 3DMatch
python eval_3dmatch.py --benchmark 3DMatch --vis
python eval_3dmatch.py --benchmark 3DLoMatch
python eval_3dmatch.py --benchmark 3DLoMatch --vis
```

## 2. Odometry KITTI

### dataset

Download odometry kitti [here](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) with `[velodyne laser data, 80 GB]` and `[ground truth poses (4 MB)]`, then unzip and organize in the following format.

```
| -- kitti
    | -- dataset
        | -- poses (#11 txt)
        | -- sequences (#11 / #22)
    | -- icp (generated automatically when training and testing)
        | -- 0_0_11.npy
        | -- ...
        | -- 9_992_1004.npy
```

### train

```
# Reconfigure configs/kitti.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the Odometry KITTI.

cd NgeNet
python train.py configs/kitti.yaml
```

### evaluate and visualize

```
# Reconfigure configs/kitti.yaml by updating the following values.
# checkpoint: your_ckpt_path

cd NgeNet
python eval_kitti.py
python eval_kitti.py --vis
```

## 3. MVP-RG

### dataset

Download MVP-RG dataset [here](https://mvp-dataset.github.io/MVP/Registration.html), then organize in the following format.

```
| -- mvp_rg
    | -- MVP_Train_RG.h5
    | -- MVP_Test_RG.h5
```

### train

```
# Reconfigure configs/mvp_rg.yaml by updating the following values based on your dataset.

# exp_dir: your_saved_path for checkpoints and summary.
# root: your_data_path for the MVP-RG.

python train.py configs/mvp_rg.yaml
```

### evaluate and visualize

```
# Reconfigure configs/mvp_rg.yaml by updating the following values.
# checkpoint: your_ckpt_path

python eval_mvp_rg.py
python eval_mvp_rg.py --vis
```

## Acknowledgements

Thanks for the open source code. We don't list them here due to anonymous problems.