# DINOv2 for cell classification

This project is a fork of the DINOv2 repo published by meta, aiming to use the methodology presented in their papers to train a series of
model on blood white cells images.
Developping a foundation model for blood white cells is interesting for several reasons:

- The categories of blood white cells are not unanimous, and hematologists / datasets make different classes.
- Some blood white cells present mutations that are visible on images, and those distinguishible features could be embedded by the model

## Installing

You should check that your original python has development headers, if not you
can install it with:

- `sudo apt install python3-dev`
- Installing directly from the `conda.yaml` file with `conda env create -f conda.yaml`

Then install the library package with `pip install -e .`

> Please be aware that the version of torch and similar libraries might need extra
index depending of your nvidia drivers,
reinstall those packages if needed.

## Contributing

**/!\ Please read those instructions if you want to contribute /!\\**

To contribute to the package, first install the required libraries with
`pip install -r requirements-dev.txt`.

Then setup pre-commit by running within your terminal `pre-commit install`. This
will make pre-commit auto launch after each of your commits. If you want to check
before commiting if your file are correctly formatted run `pre-commit run --all-files`.

When pushing, the submitted repo will pass through github actions checking style,
installation and tests. Do not make a Pull request before being sure that these
hooks are validated !

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

> The code used for these results is in `notebooks/dinov2_embedding_visualisation.ipynb`

### Classifiers results

Nearest Neighbors and Linear Probing were applied on the embeddings created by the model, the models were performed with a 5-fold cross validation.
Results are shown in the format *mean (+/- std)*

|                | f1_score       | recall         | precision      | balanced_accuracy   |
|:---------------|:---------------|:---------------|:---------------|:--------------------|
| 1-NN           | 86.6 (+/- 0.7) | 86.7 (+/- 0.7) | 86.6 (+/- 0.7) | 85.2 (+/- 0.5)      |
| 20-NN          | 89.1 (+/- 0.7) | 89.2 (+/- 0.7) | 89.2 (+/- 0.6) | 88.1 (+/- 0.8)      |
| Linear Probing | 91.6 (+/- 0.2) | 91.6 (+/- 0.2) | 91.6 (+/- 0.2) | 90.4 (+/- 0.3)      |

> The code used for these metrics is in `notebookd/embedding_testing.ipynb`

### RGB images of cells

![RGB](/cells_rgb.png)

> The code used for these results is in `notebooks/dinov2_embedding_visualisation.ipynb`

### Attention maps

Below are displayed the attention map between the cls or register tokens and the different patches of images :

![cls_attention_map](/notebooks/attention_map_cls.png)

![register_attention_map](/notebooks/attention_map_registers.png)

> The code used for those figures is in `notebooks/attention_maps.ipynb`
