REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT
NUM_GPUS = 2
DATA_DIR = /raid/home/guevel/data/Single_cells/matek
OUTPUT_DIR = logs/memory_probing
ARCH = vit_base
BATCH_SIZE = 16

# Activate conda environment
conda activate dinov2

torchrun --standalone --nproc_per_node = $NUM_GPUS dinov2/experiments/memory_probing.py \
    --data_dir = $DATA_DIR \
    --output_dir = $OUTPUT_DIR \
    --arch = $ARCH \
    --batch_size = $BATCH_SIZE
