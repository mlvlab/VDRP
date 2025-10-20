CUDA_VISIBLE_DEVICES=4,5 python main_tip_finetune.py --world-size 2 \
    --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/camera_NF \
    --num_classes 117 --port 3397 \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --use_insadapter \
    --zs_type non_rare_first \
    --logits_type S+T \
    --prompt_learning \
    --use_verb_cue_selector \
    --use_gaussian_sampler \
    --use_cov_net \
    
wait
