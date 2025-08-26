# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import math
import os
import time
from functools import partial
from typing import Tuple, NoReturn

import numpy as np
import torch
from fvcore.common.checkpoint import PeriodicCheckpointer
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from torch.nn import Module

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_custom_dataset, make_labelled_dataset, make_eval_dataset
from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler, write_list
from dinov2.train.ssl_meta_arch import SSLMetaArch

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def calculate_embedding(
    dataloader: torch.utils.data.DataLoader, model: Module, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes feature embeddings for all images in a dataloader using the given model.
    Args:
        dataloader: torch.utils.data.DataLoader, A PyTorch dataloader yielding (image, label) batches.
        model: Module, The model used to extract embeddings.
        device: torch.device, The device on which to run the model ("cuda" or "cpu").
    Returns:
        embeddings: np.ndarray, The concatenated embeddings of all images, shape (N, D).
        labels: np.ndarray, The concatenated labels for all images, shape (N,).
    """
    embeddings = []
    label_list = []

    model.eval()
    model.to(device)

    for images, labels in dataloader:

        images = images.to(device)
        # Passage dans le modèle
        with torch.no_grad():
            output = model(images)

        embeddings.append(output.cpu())
        label_list.append(labels.cpu())

        del images, output
        torch.cuda.empty_cache()

    return torch.cat(embeddings, dim=0).numpy(), torch.cat(label_list, dim=0).numpy()


def k_nearest_neighbor_eval(
    train_array: np.ndarray,
    train_labels: np.ndarray,
    test_array: np.ndarray,
    test_labels: np.ndarray,
    target_names: list,
    k: int,
) -> dict[str, dict[str, dict]]:
    """
    Evaluates a k-Nearest Neighbors (k-NN) classifier on precomputed image embeddings.
    Args:
        train_array: np.ndarray, Embeddings for the training images.
        train_labels: np.ndarray, Corresponding labels for the training images.
        test_array: np.ndarray, Embeddings for the test images.
        test_labels: np.ndarray, Corresponding labels for the test images.
        target_names: list, List of class names for report formatting.
        k: int, Number of neighbors to use for KNN.
    Returns:
        report: dict, A classification report with precision, recall, and F1-score for each class.
    """
    # Initialize the classifier
    cls = KNeighborsClassifier(n_neighbors=k)

    # Fit the model
    cls.fit(train_array, train_labels)

    # Make the predictions
    preds = cls.predict(test_array)
    report = classification_report(test_labels, preds, target_names=target_names, output_dict=True, zero_division=0)

    return report


def linear_probing_eval(
    train_array: np.ndarray,
    train_labels: np.ndarray,
    test_array: np.ndarray,
    test_labels: np.ndarray,
    target_names: list,
) -> dict[str, dict[str, dict]]:
    """
    Evaluates a logistic regression classifier (linear probing) on image embeddings.
    Args:
        train_array: np.ndarray, Embeddings for the training images.
        train_labels: np.ndarray, Corresponding labels for the training images.
        test_array: np.ndarray, Embeddings for the test images.
        test_labels: np.ndarray, Corresponding labels for the test images.
        target_names: list, List of class names for report formatting.
    Returns:
        report: dict, A classification report with precision, recall, and F1-score for each class.
    """
    # Initialize the classifier
    cls = LogisticRegression()

    # Fit the model
    cls.fit(train_array, train_labels)

    # Make the predictions
    preds = cls.predict(test_array)
    report = classification_report(test_labels, preds, target_names=target_names, output_dict=True, zero_division=0)

    return report


def do_test(
    cfg: DictConfig,
    model: Module,
    iteration: int,
    dataloader_fit: torch.utils.data.DataLoader,
    dataloader_eval: torch.utils.data.DataLoader,
    device: torch.device,
    all_dict: dict[str, dict[str, dict]],
    target_names: list,
) -> NoReturn:
    """
    Performs model evaluation using 1-NN, 20-NN, and linear probing on the given dataloaders. Saves evaluation metrics and a checkpoint of the teacher model at the current training iteration.
    Args:
        cfg: DictConfig, Hydra config object containing training options and output paths.
        model: Module, The model at the current iteration.
        iteration: int, Training iteration number at which evaluation is performed.
        dataloader_fit: torch.utils.data.DataLoader, Dataloader for the training split (for embedding extraction).
        dataloader_eval: torch.utils.data.DataLoader, Dataloader for the evaluation split.
        device: torch.device, Device used for inference.
        all_dict: dict, Nested dictionary storing previous evaluation metrics, updated in-place.
        target_names: list, Names of the target classes.
    Returns:
        all_dict: dict, The updated dictionary containing metrics for this iteration, structured for saving to JSON.
    """
    if distributed.is_main_process():
        print("Starting evaluation at iteration {}".format(iteration))
        start_time = time.time()
        # Compute the embeddings
        train_array, train_labels = calculate_embedding(
            dataloader=dataloader_fit, model=model.teacher.backbone, device=device
        )
        test_array, test_labels = calculate_embedding(
            dataloader=dataloader_eval, model=model.teacher.backbone, device=device
        )

        # Eval on 1-NN
        report_1NN = k_nearest_neighbor_eval(train_array, train_labels, test_array, test_labels, target_names, k=1)

        # Eval on 20-NN
        report_20NN = k_nearest_neighbor_eval(train_array, train_labels, test_array, test_labels, target_names, k=20)

        # Eval on linear probing
        report_linear = linear_probing_eval(train_array, train_labels, test_array, test_labels, target_names)

        # Append the results to the dataset
        all_dict[str(iteration)] = {
            "1-NN": report_1NN,
            "20-NN": report_20NN,
            "linear probing": report_linear,
        }

        # metric to log
        end_time = time.time() - start_time

        acc_1nn = report_1NN["accuracy"]
        f1_macro_1nn = report_1NN["macro avg"]["f1-score"]

        acc_20nn = report_20NN["accuracy"]
        f1_macro_20nn = report_20NN["macro avg"]["f1-score"]

        acc_linear = report_linear["accuracy"]
        f1_macro_linear = report_linear["macro avg"]["f1-score"]

        logger.info(
            f"[Evaluation @ iter {iteration}] "
            f"1-NN: acc={acc_1nn:.4f}, f1-macro={f1_macro_1nn:.4f} | "
            f"20-NN: acc={acc_20nn:.4f}, f1-macro={f1_macro_20nn:.4f} | "
            f"Linear: acc={acc_linear:.4f}, f1-macro={f1_macro_linear:.4f} | "
            f"Time: {end_time:.6f}"
        )

        # Save the results
        eval_path = os.path.join(cfg.train.output_dir, "eval_metrics.json")
        with open(eval_path, "w") as f:
            json.dump(all_dict, f, indent=4)

        # Save the teacher at eval iteration
        new_state_dict = model.teacher.state_dict()

        # Create folder for the current iteration
        iterstring = f"training_{iteration}"
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    do_daino = cfg.daino.loss_weight > 0

    # Setup optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule,) = build_schedulers(
        cfg
    )  # create training parameters that depends on the epochs

    # Checkpointer
    checkpointer = FSDPCheckpointer(
        model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True
    )  # make a checkpointer that saves periodically in fsdp fashion in order to retake training if some workers fail

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=10 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=2,
    )  # wrap the checkpointer to tune the frequency to save and the number of versions to keep

    # Setup data preprocessing
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )  # make DINOV2 data transformation (global and local crops)

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # Make datasets
    dataset = make_custom_dataset(
        cfg.train.dataset_path,
        transform=data_transform,
        path_preserved=cfg.train.path_preserved,
        frac=cfg.train.frac,
    )

    if do_daino:
        labelled_dataset = make_labelled_dataset(
            cfg.daino.labelled_dataset_path,
            cfg.train.dataset_path,
        )
        print(f"{len(labelled_dataset)} elements were found for the labelled dataset")

    if cfg.evaluation.eval_period_iterations > 0:
        dataset_fit = make_eval_dataset(cfg.evaluation.fit_dataset_path, img_size)
        dataset_eval = make_eval_dataset(cfg.evaluation.eval_dataset_path, img_size)

        all_labels = sorted(set(dataset_fit.labels) | set(dataset_eval.labels))
        translate_dict = {label: i for i, label in enumerate(all_labels)}
        dataset_fit.translate_dict = translate_dict
        dataset_eval.translate_dict = translate_dict

        target_names = list(dataset_fit.translate_dict.keys())  # for the classification report

    # Save the preserved images, if necessary
    if dataset.preserved_images:
        write_list(
            os.path.join(cfg.train.output_dir, "preserved_images.pkl"),
            dataset.preserved_images,
        )

    # Setup the unlabelled data loader
    sampler_type = SamplerType.SHARDED_INFINITE  # define the sampler to use for fsdp
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Setup the labelled data generator
    if do_daino:
        labelled_dataloader = make_data_loader(
            dataset=labelled_dataset,
            batch_size=cfg.daino.labelled_batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=True,
            sampler_type=sampler_type,
            drop_last=True,
        )
        labelled_iterator = iter(labelled_dataloader)

    # Setup the eval data loader
    if cfg.evaluation.eval_period_iterations > 0:
        data_loader_fit = torch.utils.data.DataLoader(
            dataset_fit,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=True,
            drop_last=True,
        )

        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=cfg.train.batch_size_per_gpu,
            num_workers=cfg.train.num_workers,
            shuffle=False,
            drop_last=True,
        )

        # Dictionary of the eval metrics
        all_eval_metrics = {}

    # A bit of verbose for information sake
    logger.info("There are {} images in the unlabelled dataset used".format(len(dataset)))
    if do_daino:
        logger.info("There are {} images in the labelled dataset used".format(len(labelled_dataset)))

    # Training loop
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):

        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # Apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # Compute losses
        optimizer.zero_grad(set_to_none=True)
        if do_daino:
            images, labels = next(labelled_iterator)
            labels = torch.tensor(labels, device="cuda")
            loss_dict = model.forward_backward(data, teacher_temp=teacher_temp, labelled_data=(images, labels))

        else:
            loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # Clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # Perform teacher EMA update
        model.update_teacher(mom)

        # Logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)  # synchronize the gradients calculated on the different shards and gpus
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(memory=torch.cuda.max_memory_allocated() / 1024**2)

        # Checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (
            (iteration + 1) % cfg.evaluation.eval_period_iterations == 0 or iteration == 0
        ):
            do_test(
                cfg,
                model,
                iteration,
                data_loader_fit,
                data_loader_eval,
                torch.device("cuda"),
                all_eval_metrics,
                target_names,
            )
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
