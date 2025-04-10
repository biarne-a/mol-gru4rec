import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import recall_score

from config.config import Config
from gru4rec import Gru4RecModel
from gru4rec.similarity.cosine_similarity import CosineSimilarity
from gru4rec.similarity.dot_product_similarity import DotProductSimilarity
from gru4rec.similarity.similarity_utils import get_similarity_module
from train.data import get_data, Data
from train.early_stopping_callback import EarlyStoppingCallback
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


def build_model(config: Config, data: Data, device: torch.device) -> Gru4RecModel:
    gru4rec_config = config.model_config.to_dict()
    similarity_module = get_similarity_module(config, data)
    gru4rec_config.pop("similarity_config")
    return Gru4RecModel(data, device, similarity_module, **gru4rec_config).to(device)


def _build_optimizer(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    return optimizer


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def send_batch_to_device(batch: dict[str, torch.tensor], device: torch.device):
    return {k: v.to(device) for k, v in batch.items()}


def _main_loop(config):
    device = get_device()
    data = get_data(config)
    model = build_model(config, data, device)
    optimizer = _build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    local_save_filepath = _get_model_local_save_filepath(config)
    all_epoch_metrics = []
    train_losses = []
    early_stopping = EarlyStoppingCallback(config.early_stopping_patience, config.early_stopping_metric)
    for epoch in range(config.epochs):
        model.train()
        for step, batch in enumerate(data.train_dataloader):
            batch = send_batch_to_device(batch, device)
            loss = model.train_step(batch, optimizer, criterion)
            train_losses.append(loss)
            if step > 0 and step % 100 == 0:
                print(
                    f"Epoch {epoch}, Step {step}, Step Loss: {loss:.3f}, Avg. Loss: {float(np.mean(train_losses)):.3f}"
                )
            if step > 0 and step % config.val_every_n_steps == 0:
                epoch_metrics = _evaluate(criterion, data, device, model)
                all_epoch_metrics.append(epoch_metrics)
                if early_stopping.update(epoch_metrics):
                    torch.save(model.state_dict(), local_save_filepath)
                    return all_epoch_metrics
        torch.save(model.state_dict(), local_save_filepath)
    return all_epoch_metrics


def run_training(config: Config):
    os.makedirs(config.results_dir, exist_ok=True)
    set_seed()

    all_epoch_metrics = _main_loop(config)

    save_history(all_epoch_metrics, config)
    # run_evaluation(config, data, criterion)


def _evaluate(criterion, data, device, model) -> dict[str, float]:
    print("Evaluating model...")
    model.eval()
    val_losses = []
    for i, batch in enumerate(data.val_dataloader):
        batch = send_batch_to_device(batch, device)
        val_loss = model.test_step(batch, criterion)
        val_losses.append(val_loss)
    avg_val_loss = float(np.mean(val_losses))
    val_metrics = model.metrics.compute()
    all_metrics = {"val_loss": avg_val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
    _print_metrics(all_metrics)
    return all_metrics


def _print_metrics(all_metrics: dict[str, float]):
    metrics_str = {k: f'{v:.3f}' for k, v in all_metrics.items()}
    print(metrics_str)


def run_evaluation(config: Config, data: Data, criterion):
    data = get_data(config)
    local_save_filepath = _get_model_local_save_filepath(config)
    model = build_model(config.model_config, data)
    model.load_state_dict(torch.load(local_save_filepath))
    model.eval()

    test_losses = []
    all_preds = []
    all_labels = []
    for batch in data.test_dataloader:
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
