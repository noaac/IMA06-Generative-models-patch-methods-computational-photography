#!/bin/bash
# filepath: /Users/mlg/Library/CloudStorage/SynologyDrive-mlg/Telecom/Cours2AAAAAAAAAAA/Cours/IMA206/Projet/job_script.sh
#SBATCH --job-name=simclr_train_job
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"
echo "Working directory: $(pwd)"

# Activate the environment
source ~/im06-ssl/venv-im06/bin/activate
cd ~/im06-ssl

# ===============================
#  *** MODIFICATION CLÉ 1: ISOLATION DES RÉSULTATS ***
# Crée un dossier de sortie unique pour ce job en utilisant son ID SLURM.
# Tous les résultats (runs wandb, checkpoints) seront contenus ici.
# ===============================
JOB_OUTPUT_DIR="job_outputs_${SLURM_JOB_ID}"
mkdir -p "$JOB_OUTPUT_DIR"
echo "Job outputs will be saved to: $JOB_OUTPUT_DIR"

# ===============================
# WANDB Configuration
# ===============================
export WANDB_MODE=offline
# Fait pointer WANDB vers un sous-dossier à l'intérieur de notre dossier de sortie unique.
export WANDB_DIR="$JOB_OUTPUT_DIR/wandb"
mkdir -p "$WANDB_DIR"

echo "[WANDB] Mode: offline"
echo "[WANDB] Data dir: $WANDB_DIR"

# ===============================
# Define parameters here
# ===============================

# General
TRAIN=true
EVALUATE=true

# Dataset
DATASET="pathmnist"
IMAGE_SIZE=128
NUM_CLASSES=9
DOWNLOAD=true

EXP_NAME="proj_head_linear_deep"

# Encoder & projection head
ENCODER="resnet50"
PRETRAINED=true
SYNC_BN=true
PROJ_INPUT_DIM=2048
PROJ_HIDDEN_DIM=512
PROJ_OUTPUT_DIM=128

# Training
EPOCHS=2
BATCH_SIZE=256
WEIGHT_DECAY=1e-6
TEMPERATURE=0.7
WARMUP=true
WARMUP_EPOCHS=10
COSINE_DECAY=true
USE_LARS=true
VECTORIZED_LOSS=true
GPU_TRANSFORMS=true
NUM_WORKERS=4
PIN_MEMORY=true
LOG_INTERVAL=40
SAVE_EVERY=10

# Evaluation
EVAL_EPOCHS=1
EVAL_BATCH_SIZE=256
EVAL_LR=1e-3
EVAL_WEIGHT_DECAY=1e-6
CUT_RATIO=0.2

#Transformations
export TRANSFORMS_CROP=true
export TRANSFORMS_COLOR=true
export TRANSFORMS_BLUR=true
export TRANSFORMS_ROTATION=true

# ===============================
# Build the command
# ===============================

CMD="main.py \
    --experiment_name $EXP_NAME \
    --train $TRAIN \
    --evaluate $EVALUATE \
    --dataset $DATASET \
    --image_size $IMAGE_SIZE \
    --num_classes $NUM_CLASSES \
    --download $DOWNLOAD \
    --encoder $ENCODER \
    --pretrained $PRETRAINED \
    --sync_batch_norm $SYNC_BN \
    --proj_input_dim $PROJ_INPUT_DIM \
    --proj_hidden_dim $PROJ_HIDDEN_DIM \
    --proj_output_dim $PROJ_OUTPUT_DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --weight_decay $WEIGHT_DECAY \
    --temperature $TEMPERATURE \
    --warmup $WARMUP \
    --warmup_epochs $WARMUP_EPOCHS \
    --cosine_decay $COSINE_DECAY \
    --use_lars $USE_LARS \
    --vectorized_loss $VECTORIZED_LOSS \
    --gpu_transforms $GPU_TRANSFORMS \
    --num_workers $NUM_WORKERS \
    --pin_memory $PIN_MEMORY \
    --log_interval $LOG_INTERVAL \
    --save_every $SAVE_EVERY \
    --eval_epochs $EVAL_EPOCHS \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --eval_lr $EVAL_LR \
    --eval_weight_decay $EVAL_WEIGHT_DECAY \
    --cut_ratio $CUT_RATIO \
    --wandb_mode offline \
    --wandb_dir $WANDB_DIR \
    --transform_crop $TRANSFORMS_CROP \
    --transform_color $TRANSFORMS_COLOR \
    --transform_blur $TRANSFORMS_BLUR \
    --transform_rotation $TRANSFORMS_ROTATION \
    --device cuda"

# ===============================
# Execute the command
# ===============================
echo "[INFO] Running command:"
echo "$CMD"
python -m torch.distributed.run --nproc_per_node=2 --master_port=29510 $CMD

echo ""
echo "=========================================="
echo "Run effectuée avec succès !"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "End time: $(date)"

# ===============================
#  *** MODIFICATION CLÉ 2: ARCHIVAGE PROPRE ***
# Crée une archive avec un nom clair et n'archive QUE le dossier de sortie
# unique de ce job, ainsi que ses fichiers de log.
# ===============================
ARCHIVE_NAME="clean_results_job_${SLURM_JOB_ID}.tar.gz"
echo ""
echo "Creating clean results archive..."
tar -czf "$ARCHIVE_NAME" "$JOB_OUTPUT_DIR" *.out *.err

if [ -f "$ARCHIVE_NAME" ]; then
    ARCHIVE_SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
    echo "Archive created: $ARCHIVE_NAME ($ARCHIVE_SIZE)"
else
    echo "Failed to create archive"
fi

echo ""
echo "WANDB SYNC INSTRUCTIONS:"
echo "=========================================="
echo "1. Download results to your local machine:"
echo "   scp $(whoami)@$(hostname):$(pwd)/$ARCHIVE_NAME ."
echo ""
echo "2. Use the local sync script:"
echo "   ./synchro_focus.sh $ARCHIVE_NAME YourProjectName"
echo "=========================================="

echo "Job finished at: $(date)"
