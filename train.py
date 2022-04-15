#%%
# training script for BRATS segmentation
from torch.utils.tensorboard import SummaryWriter
import os, glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, Dataset

# MONAI
from monai.data import decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
)

# my own functions
from helper import log_metrics, log_fold, visualize_raw_data, calculate_metric

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help='Name the model you are training.')
    parser.add_argument('--epochs', default=100, type=int , help='Set max number of epochs.')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Set the initial learning rate to be used with the reduced step scheduler.')

    args = parser.parse_args()

    return args


#%% ---------------------------------------------------------------------------
# user input parameters

args = parse_args()

run_name = args.name

if torch.cuda.is_available():
    os.environ["CUDA_VISIBILE_DEVICES"] = "0,1"
    device = torch.device("cuda")
    num_gpus = torch.cuda.device_count()
    print(f"\nUsing {num_gpus} gpus.\n\n")
    workers = 2*num_gpus  # im not sure why, but it yelled at me when trying to use more with cuda
else:
    device = torch.device("cpu")
    num_gpus = 0
    print("\nUsing ", device)
    workers = 1

# model

# dataloader
batch_size = 8

# training
num_folds = 5
decay_learning_rate = True
learning_rate = args.lr
max_epochs = args.epochs

# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True
USE_AMP = False


logging.basicConfig(filename="logs/"+run_name+".log",
                            filemode='a',
                            format='%(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

# print info to screen
logging.info("Running "+run_name)
logging.info(f"  learning rate: {learning_rate}")
logging.info(f"  batch size: {batch_size}")

#%% ---------------------------------------------------------------------------
# load images from data directories

root_dir = os.getcwd()

#!! Load training images and labels into a dictionary
images = sorted(glob.glob(os.path.join(root_dir,"data/imagesTr/BRATS_*.nii.gz")))
labels = sorted(glob.glob(os.path.join(root_dir,"data/labelsTr/BRATS_*.nii.gz")))

data = [{"image": image, "label": label} for image, label in zip(images,labels)]

kfold = KFold(n_splits=num_folds, shuffle=True)

class BRATSImageDataset(Dataset):
    def __init__(self, img_label_list, transform=None):
        self.img_label_list = img_label_list
        self.transform = transform

    def __len__(self):
        return len(self.img_label_list)

    def __getitem__(self, idx):
        if self.transform:
            img_label = self.transform(self.img_label_list[idx])
        else:
            img_label = self.img_label_list[idx]
        return img_label


#%% ---------------------------------------------------------------------------
# define transforms

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        EnsureTyped(keys=["image", "label"]),
    ]
)

val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
    ]
)

inference_transform = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
)


#%% ---------------------------------------------------------------------------
# create dataloaders

dataset = BRATSImageDataset(data, transform=train_transform)
val_dataset = BRATSImageDataset(data, transform=val_transform)

#%% ---------------------------------------------------------------------------
# define model and parameters

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

# Create UNet, DiceLoss and Adam optimizer
def CreateUnet():
    model = UNet(spatial_dims=3,
                 in_channels=4,
                 out_channels=3,
                 channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2),
                 num_res_units=2)

    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    return model

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

# define inference method
def inference(input, model):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
    if USE_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


#%% ---------------------------------------------------------------------------
# train the model

for fold, (train_ids, val_ids) in enumerate(kfold.split(data)):

    logging.info(f"FOLD {fold+1}")

    # create TensorBoard
    writer_train = SummaryWriter('runs/' + run_name + f'/training-fold{fold+1}')
    writer_val = SummaryWriter('runs/' + run_name + f'/val-fold{fold+1}')

    metrics_val =   {"All (Dice)":[], "Tumor Core (Dice)": [], "Whole Tumor (Dice)": [], "Enhancing Tumor (Dice)": [], "All (Hausdorff)":[], "Tumor Core (Hausdorff)": [], "Whole Tumor (Hausdorff)": [], "Enhancing Tumor (Hausdorff)": []}

    # use amp to accelerate training
    if USE_AMP:
        scaler = torch.cuda.amp.GradScaler()

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(), sampler=train_subsampler
    )
    val_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available(), sampler=val_subsampler
    )

    model = CreateUnet()
    model.apply(reset_weights)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", percentile=95)
    hausdorff_metric_batch = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", percentile=95)

    for epoch in range(max_epochs):

        model.train()
        current_learning_rate = optimizer.param_groups[0]["lr"]

        # perform epoch training step
        train_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            train_images, train_segs = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    train_outputs = model(train_images)
                    loss = loss_function(train_outputs, train_segs)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                train_outputs = model(train_images)
                loss = loss_function(train_outputs, train_segs)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        # calculate metrics
        train_loss /= step

        writer_train.add_scalar("training loss", train_loss, epoch + 1)
        writer_train.add_scalar("learning rate", current_learning_rate, epoch + 1)

        if decay_learning_rate:
            lr_scheduler.step()
        
        logging.info(f"[{epoch+1:3d}/{max_epochs:3d}] train loss: {train_loss:.3e}")

    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_images, val_segs = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            val_outputs = inference(val_images, model)
            val_outputs = [inference_transform(i) for i in decollate_batch(val_outputs)]
            dice_metric(y_pred=val_outputs, y=val_segs)
            dice_metric_batch(y_pred=val_outputs, y=val_segs)
            hausdorff_metric(y_pred=val_outputs, y=val_segs)
            hausdorff_metric_batch(y_pred=val_outputs, y=val_segs)

        calculate_metric(dice_metric, dice_metric_batch, metrics_val, "Dice")
        calculate_metric(hausdorff_metric, hausdorff_metric_batch, metrics_val, "Hausdorff")

    log_fold(writer_val, metrics_val, fold)

    current_val_dice = metrics_val["All (Dice)"][-1]
    current_val_hd = metrics_val["All (Hausdorff)"][-1]
    logging.info(f"\nValidation results for fold {fold+1:1d}/{num_folds:1d} Dice: {current_val_dice:.3f}, Hausdorff: {current_val_hd:.3f}\n")

    save_path = "trained_models/" + run_name + f"-fold{fold+1}" + ".pth"
    torch.save(model.state_dict(), save_path)

    writer_train.flush()
    writer_val.close()
    writer_train.flush()
    writer_val.close()


#%% ---------------------------------------------------------------------------
# Post Processing

def visualize_segmentation(data, weights_file):
    seg = segment(data, weights_file)
    print(seg.shape)
    
    plt.figure("Ground Truth", (17, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(data["label"][i, :, :, 59].detach().cpu())
    plt.show()
 
    plt.figure("3D Unet", (17, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(seg[i, :, :, 59].detach().cpu())
    plt.show()


def segment(image, weights_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CreateUnet()
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("trained_models/"+weights_file))
    else:
        model.load_state_dict(torch.load("trained_models/"+weights_file,
                                         map_location=torch.device("cpu")))
    model.eval()
    with torch.no_grad():
        input = image["image"].unsqueeze(0).to(device)
        seg = inference(input, model)
    
    return inference_transform(seg[0,:,:,:,:])

#%% ---------------------------------------------------------------------------