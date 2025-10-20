# VDRP
[NeurIPS 2025] Official code for Visual Diversity and Region-aware Prompt learning for Zero-shot HOI detection

### Dataset 
Follow the process of [UPT](https://github.com/fredzzhang/upt).

The downloaded files should be placed as follows. Otherwise, please replace the default path to your custom locations.
```
|- VDRP
|   |- hicodet
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |- vcoco
|   |   |- mscoco2014
|   |       |- train2014
|   |       |-val2014
:   :      
```

### Dependencies
1. Follow the environment setup in [UPT](https://github.com/fredzzhang/upt).

2. Our code is built upon [CLIP](https://github.com/openai/CLIP). Install the local package of CLIP:
```
cd CLIP && python setup.py develop && cd ..
```

3. Download the CLIP weights to `checkpoints/pretrained_clip`.
```
|- VDRP
|   |- checkpoints
|   |   |- pretrained_clip
|   |       |- ViT-B-16.pt
|   |       |- ViT-L-14-336px.pt
:   :      
```

4. Download the weights of DETR and put them in `checkpoints/`.


| Dataset | DETR weights |
| --- | --- |
| HICO-DET | [weights](https://drive.google.com/file/d/1BQ-0tbSH7UC6QMIMMgdbNpRw2NcO8yAD/view?usp=sharing)  |
| V-COCO | [weights](https://drive.google.com/file/d/1AIqc2LBkucBAAb_ebK9RjyNS5WmnA4HV/view?usp=sharing) |


```
|- VDRP
|   |- checkpoints
|   |   |- detr-r50-hicodet.pth
|   |   |- detr-r50-vcoco.pth
:   :   :
```

### Pre-extracted Features
Download the pre-extracted features from [HERE](https://drive.google.com/file/d/1lUnUQD3XcWyQdwDHMi74oXBcivibGIWN/view?usp=sharing) and the pre-extracted bboxes from [HERE](https://drive.google.com/file/d/19Mo1d4J6xX9jDNvDJHEWDpaiPKxQHQsT/view?usp=sharing). The downloaded files have to be placed as follows.

```
|- VDRP
|   |- hicodet_pkl_files
|   |   |- union_embeddings_cachemodel_crop_padding_zeros_vitb16.p
|   |   |- hicodet_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
|   |- vcoco_pkl_files
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit16.p
|   |   |- vcoco_union_embeddings_cachemodel_crop_padding_zeros_vit336.p
:   :      
```

### Train/Test

Please follow the commands in ```./scripts```.



### Model Zoo

| Method          | Type  | Unseen↑ | Seen↑ | Full↑ | HM↑   |
|-----------------|-------|---------|-------|-------|-------|
| VDRP (Ours)     | NF-UC | 36.45   | 31.60 | 32.57 | 33.85 |
| VDRP (Ours)     | RF-UC | 31.29   | 34.41 | 33.78 | 32.77 |
| VDRP (Ours)     | UO    | 36.13   | 32.84 | 33.39 | 34.41 |
| VDRP (Ours)     | UV    | 26.69   | 33.72 | 32.73 | 29.80 |

### Model Weights

You can download the VDRP weights from huggingface:
```

```

## Citation
If you find our paper and/or code helpful, please consider citing:
```
}
```

## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt), [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main) and [CMMP](https://github.com/ltttpku/CMMP) for open-sourcing their code.
