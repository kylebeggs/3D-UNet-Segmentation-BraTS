# helper functions

import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
from torchvision.utils import make_grid
import torch
import logging

def visualize_raw_data(data):
    data_example = data[1]
    print(f"image shape: {data_example['image'].shape}")
    plt.figure("image", (23, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(data_example["image"][i, :, :, 59].detach().cpu(), cmap="gray")
    plt.show()
    # also visualize the 2 channels label corresponding to this image
    print(f"label shape: {data_example['label'].shape}")
    plt.figure("label", (17, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(data_example["label"][i, :, :, 59].detach().cpu())
    plt.show()


def calculate_metric(Metric, MetricBatch, metrics_dict, type):

    if type != "Dice" or "Hausdorff":
        ValueError("type argument must be Dice or Hausdorff (cap sensitive)")

    metric = Metric.aggregate().item()
    metrics_dict[f"All ({type})"].append(metric)
    metric_batch = MetricBatch.aggregate()
    metric_tc = metric_batch[0].item()
    metrics_dict[f"Tumor Core ({type})"].append(metric_tc)
    metric_wt = metric_batch[1].item()
    metrics_dict[f"Whole Tumor ({type})"].append(metric_wt)
    metric_et = metric_batch[2].item()
    metrics_dict[f"Enhancing Tumor ({type})"].append(metric_et)
    Metric.reset()
    MetricBatch.reset()


def log_metrics(writer, epoch, metrics, lr):
    for key, value in metrics.items():
        writer.add_scalar(key, value[-1], epoch + 1)
    writer.add_scalar("learning rate", lr, epoch + 1)


def log_fold(writer, metrics, fold):
    logging.info(f"Validation results for fold {fold+1:1d}:")
    for key, value in metrics.items():
        writer.add_scalar(key, value[-1], fold)
        logging.info(f"{key}: {value[-1]:.3f}")

