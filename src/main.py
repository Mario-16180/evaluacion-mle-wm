import preprocess_data as ppd
import data_splitting as ds
import train as tr
import pandas as pd
import wandb
import argparse

if __name__ == '__main__':
    # argparse of the arguments of the XGBoost model
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0)
    parser.add_argument('--max_depth', type=int, default=6)
    # This parameter is used to control the balance of positive and negative weights, useful for unbalanced classes.
    # The value used is sum(negative instances) / sum(positive instances), in this case, 110316 / 31877 = 3.46
    parser.add_argument('--scale_pos_weight', type=float, default=0.1)
    args = parser.parse_args()
    gamma = args.gamma
    max_depth = args.max_depth
    scale_pos_weight = args.scale_pos_weight

    # Data reading
    data = pd.read_csv(r'data\raw\weatherAUS.csv')

    # Data preprocessing
    data_preprocessed = ppd.preprocess_data(data)

    # Saving the data
    data_preprocessed.to_csv(r'data\processed\data_preprocessed.csv', index=False)

    # Data splitting
    X_train, X_test, y_train, y_test = ds.split_data(data_preprocessed, test_size=0.2, random_state=42)

    # Model training
    model, acc, f1 = tr.train_model(X_train, X_test, y_train, y_test, gamma, max_depth, scale_pos_weight)

    print(f'Accuracy: {acc}')
    print(f'F1: {f1}')