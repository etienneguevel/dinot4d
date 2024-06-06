# DINOv2 for cell classification

This project is a fork of the DINOv2 repo published by meta, aiming to use the methodology presented in their papers to train a series of
model on blood white cells images.
Developping a foundation model for blood white cells is interesting for several reasons:

- The categories of blood white cells are not unanimous, and hematologists / datasets make different classes.
- Some blood white cells present mutations that are visible on images, and those distinguishible features could be embedded by the model

## Installing

To install the project and the required packages that are necessary for this project use `conda env create -f conda.yaml`,
then `conda activate cell_sim` and `pip install -e .`.

The dinov2 packages is then available from the conda env cell_sim for execution.

## Data

The dataset implementation is in the file `dinov2/data/datasets/custom_image_datasets.py`, and retrieves all the images in all the folders indicated in the dataset paths. There are options to ignore a fraction of the images in each dataset given, and to check if the images are not corrupted before using them.

## Training

**Disclaimer:**

Most of the code used for the training was used directly from the [dinov2 repo](https://github.com/facebookresearch/dinov2/tree/main),
and not every function has been checked. Though it should be functional and do as intended, some aspects might not work as expected.

### Config file

config files control every elements of the model, and of its training. The implementation made on your config files will merge with the one by default located at `dinov2/configs/ssl_default_config.yaml`.
In our case the minimal requirements for the train config files should be:

- `dataset_path` (List[str] or str) that indicates the path where the training data is located
- `output_dir` (str) that indicates the path where the logs, ckpts and models will be saved

### Submit training

The script used to submit the training process is located at `dinov2/run/train/train.py`.
An example of command would be `python dinov2/run/train/train.py --config-file dinov2/configs/train/vitl_cellsim_register.yaml --output-dir /your/output/dir --partition hard --ngpus 2`

that would launch the training on 1 node with 2 GPUs on the partition named `hard`.
> This command makes a sh script with the required slurm constraints that is then executed.

If your setup is a machine with several GPUs available on it, then the best way to launch the training is by using `torchrun` ([here](https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)), like with this command:

`torchrun --standalone --nproc_per_node=4 dinov2/train/train.py --config-file dinov2/configs/train/vitl_cellsim_register.yaml --output-dir /your/output/dir`

**The conda env previously created should be activated before launching this command.**

## Results

The [Barcelona dataset](https://www.sciencedirect.com/science/article/pii/S2352340920303681) was used to see the quality of the embeddings obtained after training.

The model used was trained following the procedure given by the autors of the dinov2 article, using 4 register tokens and a vitl architecture.

### Barcelona dataset

![umap](/umap_barcelona.png)

*UMAP in 2 dimensions plot of the images contained in the Barcelona dataset*

### Classifiers results

Nearest Neighbors and Linear Probing were applied on the embeddings created by the model, the models were performed with a 5-fold cross validation.
Results are shown in the format *mean (+/- std)*

|                | f1_score       | recall         | precision      | balanced_accuracy   |
|:---------------|:---------------|:---------------|:---------------|:--------------------|
| 1-NN           | 86.6 (+/- 0.7) | 86.7 (+/- 0.7) | 86.6 (+/- 0.7) | 85.2 (+/- 0.5)      |
| 20-NN          | 89.1 (+/- 0.7) | 89.2 (+/- 0.7) | 89.2 (+/- 0.6) | 88.1 (+/- 0.8)      |
| Linear Probing | 91.6 (+/- 0.2) | 91.6 (+/- 0.2) | 91.6 (+/- 0.2) | 90.4 (+/- 0.3)      |

### RGB images of cells

![RGB](/cells_rgb.png)
