#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --partition=small-g
#SBATCH --time=24:00:00
#SBATCH --account=project_465002853
#SBATCH --output=logs/suim6/output_%j.txt

# Use node-local scratch for MIOpen DB (avoid Lustre/NFS locking issues)
MIOPEN_LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/${USER}/miopen-${SLURM_JOB_ID}"
export MIOPEN_USER_DB_PATH="$MIOPEN_LOCAL"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_LOCAL"
mkdir -p "$MIOPEN_LOCAL"
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_MODE=1

# Activate conda in non-interactive shells and activate the env
source /scratch/project_465002853/miniconda3/etc/profile.d/conda.sh
conda activate clip

# Hugging Face cache in a directory you own
export HF_HOME="/scratch/project_465002853/hf_cache_${USER}"
mkdir -p "$HF_HOME"
export HF_HUB_DISABLE_TELEMETRY=1


cd /flash/project_465002853/projects/mlmp

mkdir -p logs/suim6

# # ========== SOURCE ==========
# python main.py \
#     --ovss_type naclip \
#     --ovss_backbone ViT-L/14 \
#     --save_dir .save/SUIM6Dataset/No_Adaptation/ \
#     --data_dir /scratch/project_465002853/datasets/suim/SUIM/ \
#     --dataset SUIM6Dataset \
#     --workers 4 \
#     --init_resize 320 256 \
#     --patch_size 224 224 \
#     --patch_stride 112 \
#     --corruptions_list gaussian_noise impulse_noise shot_noise defocus_blur motion_blur brightness contrast pixelate jpeg_compression \
#     --steps 1 \
#     --batch-size 8 \
#     --trials 1 \
#     --seed 0 \
#     --reset_mode continual \
#     --domain_gen False \
#     --domain_gen_num 5 \
#     --lifelong None \
#     --lifelong_rnds 3 \

# # ========== TENT ==========
# python main.py \
#     --adapt \
#     --method tent \
#     --ovss_type naclip \
#     --ovss_backbone ViT-L/14 \
#     --save_dir .save/SUIM6Dataset/tent/ \
#     --data_dir /scratch/project_465002853/datasets/suim/SUIM/ \
#     --dataset SUIM6Dataset \
#     --workers 4 \
#     --init_resize 320 256 \
#     --patch_size 224 224 \
#     --patch_stride 112 \
#     --corruptions_list gaussian_noise impulse_noise shot_noise defocus_blur motion_blur brightness contrast pixelate jpeg_compression \
#     --lr 1e-3 \
#     --optimizer sgd  \
#     --steps 1 \
#     --batch-size 8 \
#     --trials 1 \
#     --seed 0 \
#     --plot_loss \
#     --reset_mode continual \
#     --domain_gen False \
#     --domain_gen_num 5 \
#     --lifelong None \
#     --lifelong_rnds 3 \

# # ========== WATT ==========
# python main.py \
#     --adapt \
#     --method watt \
#     --prompt_dir prompts.yaml \
#     --watt_l 2 \
#     --watt_m 5 \
#     --ovss_type naclip \
#     --ovss_backbone ViT-L/14 \
#     --save_dir .save/SUIM6Dataset/watt/ \
#     --data_dir /scratch/project_465002853/datasets/suim/SUIM/ \
#     --dataset SUIM6Dataset \
#     --workers 4 \
#     --init_resize 320 256 \
#     --patch_size 224 224 \
#     --patch_stride 112 \
#     --corruptions_list gaussian_noise impulse_noise shot_noise defocus_blur motion_blur brightness contrast pixelate jpeg_compression \
#     --lr 1e-3 \
#     --optimizer sgd  \
#     --steps 1 \
#     --batch-size 8 \
#     --trials 1 \
#     --seed 0 \
#     --plot_loss \
#     --reset_mode continual \
#     --domain_gen False \
#     --domain_gen_num 5 \
#     --lifelong None \
#     --lifelong_rnds 3 \

# # ========== CLIPArTT ==========
# python main.py \
#     --adapt \
#     --method clipartt \
#     --clipartt_k 3 \
#     --ovss_type naclip \
#     --ovss_backbone ViT-L/14 \
#     --save_dir .save/SUIM6Dataset/clipartt/ \
#     --data_dir /scratch/project_465002853/datasets/suim/SUIM/ \
#     --dataset SUIM6Dataset \
#     --workers 4 \
#     --init_resize 320 256 \
#     --patch_size 224 224 \
#     --patch_stride 112 \
#     --corruptions_list gaussian_noise impulse_noise shot_noise defocus_blur motion_blur brightness contrast pixelate jpeg_compression \
#     --lr 1e-3 \
#     --optimizer sgd  \
#     --steps 1 \
#     --batch-size 8 \
#     --trials 1 \
#     --seed 0 \
#     --plot_loss \
#     --reset_mode continual \
#     --domain_gen False \
#     --domain_gen_num 5 \
#     --lifelong None \
#     --lifelong_rnds 3 \

# # ========== MLMP ==========
# python main.py \
#     --adapt \
#     --method mlmp \
#     --prompt_dir prompts.yaml \
#     --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 \
#     --alpha_cls 1.0 \
#     --ovss_type naclip \
#     --ovss_backbone ViT-L/14 \
#     --save_dir .save/SUIM6Dataset/mlmp/ \
#     --data_dir /scratch/project_465002853/datasets/suim/SUIM/ \
#     --dataset SUIM6Dataset \
#     --workers 4 \
#     --init_resize 320 256 \
#     --patch_size 224 224 \
#     --patch_stride 112 \
#     --corruptions_list gaussian_noise impulse_noise shot_noise defocus_blur motion_blur brightness contrast pixelate jpeg_compression \
#     --lr 1e-3 \
#     --optimizer sgd  \
#     --steps 1 \
#     --batch-size 8 \
#     --trials 1 \
#     --seed 0 \
#     --plot_loss \
#     --reset_mode continual \
#     --domain_gen False \
#     --domain_gen_num 5 \
#     --lifelong None \
#     --lifelong_rnds 3 \

# # ========== METHOD ==========
# python main.py \
#     --adapt \
#     --method method \
#     --train_imag_norm True \
#     --last_imag_k_norm 6 \
#     --train_imag_attn False \
#     --last_imag_k_attn 1 \
#     --train_text_norm False \
#     --last_text_k_norm 0 \
#     --loss_ent True --lamb_ent 1.0 \
#     --loss_div True --lamb_div 1.0 \
#     --loss_aug_cons True --lamb_aug_cons 1.0 \
#     --loss_src_cons False --lamb_src_cons 1.0 \
#     --updownsample 1.0 \
#     --prompt_average False \
#     --ovss_type naclip \
#     --ovss_backbone ViT-L/14 \
#     --save_dir .save/SUIM6Dataset/tent/ \
#     --data_dir /scratch/project_465002853/datasets/suim/SUIM/ \
#     --dataset SUIM6Dataset \
#     --workers 4 \
#     --init_resize 320 256 \
#     --patch_size 224 224 \
#     --patch_stride 112 \
#     --corruptions_list gaussian_noise impulse_noise shot_noise defocus_blur motion_blur brightness contrast pixelate jpeg_compression \
#     --lr 1e-3 \
#     --optimizer sgd  \
#     --steps 1 \
#     --batch-size 8 \
#     --trials 1 \
#     --seed 0 \
#     --plot_loss \
#     --reset_mode continual \
#     --domain_gen False \
#     --domain_gen_num 5 \
#     --lifelong None \
#     --lifelong_rnds 3 \