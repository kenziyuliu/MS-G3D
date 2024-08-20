# SCP-GCN

PyTorch implementation of "Skeleton Compression-Based Multi-Detail Part-Wise Graph Convolutional Network for Robust Human Action Recognition".

## Dependencies

- Python >= 3.6
- PyTorch >= 1.12.0
- tqdm, ftfy,regex,transformers,clip

### Download Datasets

Two datasets are available for download:

- NTU-RGB+D 60 Skeleton
- NTU-RGB+D 120 Skeleton

#### NTU RGB+D 60 and 120

1. Request the dataset here: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp

2. Download the skeleton-only datasets:
    - `nturgbd_skeletons_s001_to_s017.zip`  (NTU RGB+D 60)
    - `nturgbd_skeletons_s018_to_s032.zip`  (NTU RGB+D 120, on top of NTU RGB+D 60)
    - Total size should be 5.8GB + 4.5GB.

### Data Preprocessing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - ntu60/
  - ntu60/     # from `nturgbd_skeletons_s001_to_s017.zip`
    ...
  - ntu120/    # from `nturgbd_skeletons_s018_to_s032.zip`
    ...
```

#### Generating Data

1. NTU RGB+D
    - `cd data_gen`
    - `python ntu_gendata.py`
    - `python ntu120_gendata.py`

## Training & Testing

- The general training template command:
```
python main_limbs.py
```

## Notes
- The best joint-bone fusion result may not come from the best single stream models; for example, we provided 3 pretrained models for NTU RGB+D 60 XSub joint stream where the best fusion performance comes from the best performing model (~94.6%).


## Acknowledgements

This repo is based on
  - [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
  - [ST-GCN](https://github.com/yysijie/st-gcn)

Thanks to the original authors for their work!


## Citation

Please cite this work if you find it useful:

## Contact
Please email `19848056457@163.com` for further questions
