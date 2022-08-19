import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags

# custom modules
from src.trainer import Trainer, TrainerArguments
from src.utils import set_seed
config_flags.DEFINE_config_file(
    "config",
    default=None,
    help_string="Training Configuration from `configs` directory",
)
flags.DEFINE_bool("wandb_enabled", default=True, help="enable Weights & Biases logging")

FLAGS = flags.FLAGS
FLAGS(sys.argv)   # need to explicitly to tell flags library to parse argv before you can access FLAGS.xxx
cfg = FLAGS.config
wandb_enabled = FLAGS.wandb_enabled

# Create Fully connected network
class NN(nn.Module):
    def _init_(self, input_size, num_classes):
        super(NN, self)._init_()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, image, label=None):
        x = image.view(-1)
        out = F.relu(self.fc1(x))
        out = self.fc2(x)
        if label is not None:
            loss = nn.CrossEntropyLoss()(out, label)
            return out, loss
        return out


class MNISTDataset:
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        return {
            "image": img,
            "label": label
        }

def main():
    set_seed(42)

    # init 🤗 accelerator
    accelerator = Accelerator(
            device_placement=True,
            step_scheduler_with_optimizer=False,
            mixed_precision=cfg.trainer_args['mixed_precision'],
            gradient_accumulation_steps=cfg.trainer_args['gradient_accumulation_steps'],
            log_with="wandb" if wandb_enabled else None,
        )
    if wandb_enabled:
        # init wandb
        accelerator.init_trackers(project_name="Custom_Accelerate_Trainer_Tests", config=cfg.to_dict())
    
    # Load Data
    train_dataset = datasets.MNIST(root = "dataset/MNIST", train=True, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset = train_dataset, batch_size=cfg.batch_size, shuffle=True)
    accelerator.print(f"Train Loader = {train_loader}")

    val_dataset = datasets.MNIST(root = "dataset/MNIST", train=False, transform=transforms.ToTensor(), download=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size=cfg.batch_size, shuffle=False)


    # Initialize the network
    model = NN(input_size=cfg.input_size, num_classes=cfg.num_classes)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # implement custom metrics function
    # these metrics will be calculated and logged after every evaluation phase
    # do all the post-processing on logits in this function
    def compute_metrics(logits, labels):
        preds = np.argmax(logits, axis=1)
        acc_score = accuracy_score(labels, preds)
        # implement any other metrics you want and return them in the dict format
        return {"accuracy": acc_score}
    # trainer args
    args = TrainerArguments(**cfg.trainer_args)

    # initialize my custom trainer
    trainer = Trainer(
        model=model,
        args=args,
        optimizer=optimizer,
        accelerator=accelerator,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        compute_metrics=compute_metrics,
    )

    # call .fit to perform training + validation
    trainer.fit()
    trainer = Trainer(model, args)

    # predict on test set if you have one
    # preds = trainer.predict("./outputs/best_model.bin", test_dataloader)
    
    # here predicting on val dataset to demonstrate the `.predict()` method
    val_preds = trainer.predict("./outputs/best_model.bin", val_loader)
    if wandb_enabled:
        # end trackers
        accelerator.end_training()


if __name__ == "_main_":
    main()
