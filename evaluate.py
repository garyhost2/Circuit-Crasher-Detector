import os
import torch
import pickle
import torch.nn as nn
from torch import optim
import argparse

from config import Config
from utils.data import LoadDataset
from models.model import ImageRegressor
from models.tasks import train_model, evaluate_model


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate Image Regressor on a Test Set")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Path to the saved model weights (.pth file)",
)
args = parser.parse_args()

print("-----Evaluating the network on Test Set--------")
testset = LoadDataset(
    Config.test_csv,
    Config.images_root,
    img_size=Config.image_size,
    transforms=Config.get_test_transforms(),
    cache_strategy=Config.test_cache,
)

print("Testset Size: ", len(testset), "images")

testloader = torch.utils.data.DataLoader(
    testset, batch_size=Config.batch_size, shuffle=False
)

model_path = args.model  # Get the model path from command-line arguments
model = ImageRegressor()
model.load_state_dict(torch.load(model_path, weights_only=True, map_location=Config.device))
model = model.to(Config.device)

criterion = nn.MSELoss()
eval_loss, eval_mae, eval_r2 = evaluate_model(model, criterion, testloader, Config.device)

print(f"Evaluation Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f} \n R2 Score: {eval_r2:.4f}")
