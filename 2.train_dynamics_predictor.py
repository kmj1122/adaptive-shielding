from FunctionEncoder.Model.FunctionEncoder import FunctionEncoder
from FunctionEncoder.Dataset.TransitionDataset import TransitionDataset
import torch
import argparse
import os
from FunctionEncoder.Callbacks.LoggerCallback import LoggerCallback
from shield.util import load_data, save_config, load_model
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # fix the argparse
    parser.add_argument("--load_model", action="store_true", default=False)
    parser.add_argument("--env_id", type=str, default="SafetyPointGoal1-v1")
    parser.add_argument("--n_basis", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--use_dataloader", action="store_true", default=True, help="Use DataLoader for memory efficiency")
    args = parser.parse_args()

    env_name = args.env_id
    env_info = env_name.split('-')[0]
    train_transitions = load_data(env_info, 'train')
    eval_transitions = load_data(env_info, 'eval')
    train_X, train_Y = train_transitions['X'], train_transitions['Y']
    eval_X, eval_Y = eval_transitions['X'], eval_transitions['Y']
    n_functions = len(list(train_X.keys()))

    keys = list(train_X.keys())
    one_train_X = train_X[keys[0]]
    one_train_Y = train_Y[keys[0]]
    
    arch = "MLP"
    train_method = "least_squares"
    residuals = True
    n_basis = int(args.n_basis)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_folder = f"saved_files/dynamics_predictor/{env_name}"
    save_path = f"{save_folder}/model.pth"
    os.makedirs(save_folder, exist_ok=True)
    dataset = TransitionDataset(train_transitions, eval_transitions, n_functions=n_functions, n_examples=100, n_queries=100, dtype=torch.float32, device=device)
    example_xs, example_ys, query_xs, query_ys, info = dataset.sample(phase='eval')

    if not args.load_model:
        config = {
            "input_size": dataset.input_size,
            "output_size": dataset.output_size,
            "data_type": dataset.data_type,
            "n_basis": n_basis,
            "model_type": arch,
            "method": train_method,
            "use_residuals_method": residuals,
            "model_kwargs": {"dt": dataset.dt},
            "device": device
        }

        save_config(config, f"{save_folder}/config.yaml")
        
        model = FunctionEncoder(**config).to(device)
        model.train_model(dataset, epochs=int(args.epochs), callback=LoggerCallback(model, dataset, logdir=f"logs/dynamics_predictor/{env_name}_{arch}_{n_basis}"), save_folder=save_folder)
        model.save(save_path)
    else:
        print(f"Loading model from {save_path}")
        model = load_model(save_folder, 'dynamics_predictor')
        
        example_xs, example_ys, query_xs, query_ys, info = dataset.sample(phase='eval')

        with torch.no_grad():
            coefs, _ = model.compute_representation(example_xs, example_ys, method=train_method)
            y_hats = model.predict(query_xs, coefs)
        
        print('Prediction success')