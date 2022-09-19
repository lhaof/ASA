# Attentive Symmetric Autoencoder for Brain MRI Segmentation-MICCAI 2022

---

## Getting Started
This repo contains the supported code of Attentive Symmetric Autoencoder. It is based on [nnFormer](https://github.com/282857341/nnFormer).
### Installatioin
We test our code in `CUDA 10.2` and `pytorch 1.8.1`

```bash
git clone 
cd ASA_Pretrain
pip install -r requirements.txt
cd ../ASA_Segmentation
pip install -e .
```



### Pre-training ASA
```bash
python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 20003 tools/train.py --data_path DATA_PATH --output_dir OUTPUT_DIR
```

### Segmentation
#### Prepare Data
First Create Folder for raw data, preprocessed data and result folder

```bash
mkdir RAW_DATA_PATH
mkdir PREPROCESSED_DATA_PATH
mkdir RESULT_FOLDER_PATH
 
export nnFormer_raw_data_base=RAW_DATA_PATH
export nnFormer_preprocessed=PREPROCESSED_DATA_PATH
export RESULTS_FOLDER_nnFormer=RESULT_FOLDER_PATH
```

Download the BraTS Dataset from the [Challenge](http://braintumorsegmentation.org/).

Then change the dataset path in `dataset_conversion/Task999_BraTS_2021.py` and run it to convert the dataset. 

```bash
python dataset_conversion\Task999_BraTS_2021.py
```

After that, you can preprocess the above data using following commands:

```bash
nnFormer_plan_and_preprocess -t 999 --verify_dataset_integrity
```

### Training and Testing
Download the [ASA_PRETRAIN_MODEL](https://drive.google.com/file/d/1oX6HYhxyVmltutjAmhyzTUl5aT526CQy/view?usp=sharing) and change the `PRETRAIN_PATH` in `training\network_training\nnFormerTrainerV2_MEDIUMVIT_MAE.py`



Then Finetuning the model

```bash
nnFormer_train --network 3d_fullres --network_trainer nnFormerTrainerV2_MEDIUMVIT --task 999 --fold 0 --tag DEFAULT
```

Testing

```bash
nnFormer_predict -i "DATA_RAW_PATH/nnFormer_raw_data/Task999_BraTS2021/imagesTs/" -o "OUTPUT_PATH" -t 999 --tag "DEBUG" -tr nnFormerTrainerV2_MEDIUMVIT_MAE
```

### Pretrain Model & Segmentation Model
|   Model   |                            config                            |                            Params                            |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ASA_PRETRAIN  | [config](https://drive.google.com/file/d/1JTlocHD02UrqedwOFgdkdbySis3xUI0T/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1oX6HYhxyVmltutjAmhyzTUl5aT526CQy/view?usp=sharing) |
| ASA_SEGMENTATION | [config](https://drive.google.com/file/d/133aL_-jpNndLHKvgmC9hwskuvMzz9RXX/view?usp=sharing) | [google drive](https://drive.google.com/file/d/12_0PahqUCZFtLMZLHnftv7FZkoDxYBOI/view?usp=sharing) |
