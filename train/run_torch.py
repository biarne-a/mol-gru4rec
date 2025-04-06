import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import recall_score

from config.config import Config
from train.datasets_torch import get_data, Data
from train.save_results import save_history, save_predictions

def set_seed():
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)
    os.environ["PYTHONHASHSEED"] = str(Config.SEED)

def _get_model_local_save_filepath(config: Config) -> str:
    local_savedir = f"data/results/{config.dataset_name}"
    os.makedirs(local_savedir, exist_ok=True)
    return f"{local_savedir}/model.pth"

def _build_optimizer(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    return optimizer

def _compile_model(model, config):
    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion

def run_training(config: Config):
    os.makedirs(config.results_dir, exist_ok=True)
    set_seed()

    data = get_data(config)
    model = config.model_config.build_model(data)
    optimizer, criterion = _compile_model(model, config)
    local_save_filepath = _get_model_local_save_filepath(config)

    for epoch in range(2):  # Change to desired number of epochs
        model.train()
        for step, batch in enumerate(data.train_ds):
            loss = model.train_step(batch, optimizer, criterion)
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss}")

        model.eval()
        val_losses = []
        for batch in data.val_ds:
            val_loss = model.test_step(batch, criterion)
            val_losses.append(val_loss)
        avg_val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

        torch.save(model.state_dict(), local_save_filepath)

    save_history({"loss": loss, "val_loss": avg_val_loss}, config)
    run_evaluation(config, data)


def run_evaluation(config: Config, data: Data):
    data = get_data(config)
    local_save_filepath = _get_model_local_save_filepath(config)
    model = config.model_config.build_model(data)
    model.load_state_dict(torch.load(local_save_filepath))
    model.eval()

    test_losses = []
    all_preds = []
    all_labels = []
    for batch in data.test_ds:
        test_loss = model.test_step(batch, criterion)
        test_losses.append(test_loss)
        preds = model(batch).argmax(dim=1).cpu().numpy()
        labels = batch["label"].cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_test_loss = np.mean(test_losses)
    test_recall = recall_score(all_labels, all_preds, average="macro")
    print(f"Test Loss: {avg_test_loss}, Test Recall: {test_recall}")

    save_predictions(config, data, model)
