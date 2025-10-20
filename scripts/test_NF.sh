# [ViT-L] 
# --output-dir checkpoints/camera_NF_L
# --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt

# Training 
CUDA_VISIBLE_DEVICES=0,1 python main_tip_finetune.py --world-size 2 \
    --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/camera_NF \
    --num_classes 117 --port 3397 \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --use_insadapter \
    --zs_type non_rare_first \
    --logits_type S+T \
    --prompt_learning \
    --use_verb_cue_selector \
    --use_gaussian_sampler \
    --use_cov_net \
    
# Evaluation
CUDA_VISIBLE_DEVICES=0 python main_tip_finetune.py --world-size 1 \
    --pretrained checkpoints/detr-r50-hicodet.pth --output-dir checkpoints/camera_NF \
    --num_classes 117 --port 3397 \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt --zs --use_insadapter \
    --zs_type non_rare_first \
    --logits_type S+T \
    --prompt_learning \
    --use_verb_cue_selector \
    --use_gaussian_sampler \
    --use_cov_net \
    --resume checkpoints/camera_NF/{File} --eval
