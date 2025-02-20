# This script is to test the memory taken for different config of training
# The path indicated here are for the DGX clustet, you need to change them to your own path

REPO_ROOT=$(git rev-parse --show-toplevel)
cd $REPO_ROOT

source activate /home/guevel/.conda/envs/cell_sim

DATA_DIR=/raid/home/guevel/data/Single_cells/matek
OUTPUT_DIR=logs/memory_probing

archs=("vit_base" "vit_large")
batch_sizes=(1 2 4 8 16 32)
num_gpus=(2 4 6)

for arch in "${archs[@]}"; do
    for b in "${batch_sizes[@]}"; do
        for g in "${num_gpus[@]}"; do
            config="arch=$arch,batch_size=$b,num_gpus=$g"
            echo "Processing $config"
            torchrun --nproc_per_node=$g dinov2/experiments/memory_probing.py \
                --dataset=$DATA_DIR \
                --output_dir=$OUTPUT_DIR \
                --arch=$arch \
                --batch_size=$b \
                --num_gpus=$g
        done
    done
done
