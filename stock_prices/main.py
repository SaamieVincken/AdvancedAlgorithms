import wandb
from config import get_wandb_config
from data import prepare_data, create_dataset
from model import StockPredictor
from train import train_model, evaluate_model

epochs = 1000
config = get_wandb_config(epochs)
wandb.init(project="Stock forecaster", config=config, name='v0.1 base')

if __name__ == "__main__":
    train_data, test_data = prepare_data()
    X_train, y_train = create_dataset(train_data)
    X_test, y_test = create_dataset(test_data)

    model = StockPredictor()
    train_model(model, X_train, y_train, epochs)

    predictions = evaluate_model(model, X_test, y_test)

