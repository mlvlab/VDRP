CUDA_VISIBLE_DEVICES=4,5 python main_tip_finetune.py --world-size 2 \
    --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/camera_UO_R \
    --num_classes 117 --port 3399 \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --use_insadapter \
    --zs_type unseen_object \
    --logits_type S+T \
    --prompt_learning \
    --use_verb_cue_selector \
    --use_gaussian_sampler \
    --use_cov_net \

wait
