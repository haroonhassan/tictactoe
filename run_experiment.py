#!/usr/bin/env python3
"""
Example script for running experiments with wandb tracking.
This demonstrates how to configure and run training experiments.
"""

import torch
from config import Config
from models import create_model
from trainer import Trainer


def run_baseline_experiment():
    """Run baseline experiment with standard architecture"""
    # Create config
    config = Config()

    # Configure experiment
    config.logging.wandb_name = "baseline-standard-net"
    config.logging.wandb_tags = ["baseline", "standard"]
    config.logging.wandb_notes = "Baseline experiment with standard architecture"

    # Model settings
    config.model.num_hidden_layers = 4
    config.model.hidden_size = 128
    config.model.activation = "relu"

    # Training settings
    config.training.num_games = 1000
    config.training.learning_rate = 0.001
    config.training.batch_size = 64
    config.training.temperature_decay = "cosine"
    config.training.use_augmentation = True

    # Create model and trainer
    model = create_model(config, model_type='standard')
    trainer = Trainer(model, config)

    # Run training
    trainer.train()


def run_resnet_experiment():
    """Run experiment with ResNet architecture"""
    config = Config()

    # Configure experiment
    config.logging.wandb_name = "resnet-4layer-experiment"
    config.logging.wandb_tags = ["resnet", "deep"]
    config.logging.wandb_notes = "Testing ResNet architecture with residual connections"

    # Model settings
    config.model.num_hidden_layers = 4
    config.model.hidden_size = 128
    config.model.activation = "relu"

    # Training settings
    config.training.num_games = 1000
    config.training.learning_rate = 0.001
    config.training.batch_size = 64

    # Create ResNet model
    model = create_model(config, model_type='resnet')
    trainer = Trainer(model, config)

    # Run training
    trainer.train()


def run_hyperparameter_sweep():
    """Example of running multiple experiments with different hyperparameters"""
    learning_rates = [0.0001, 0.001, 0.01]
    hidden_sizes = [64, 128, 256]

    for lr in learning_rates:
        for hidden_size in hidden_sizes:
            config = Config()

            # Configure experiment with descriptive name
            config.logging.wandb_name = f"sweep-lr{lr}-h{hidden_size}"
            config.logging.wandb_tags = ["sweep", "hyperparameter-search"]
            config.logging.wandb_notes = f"Testing lr={lr}, hidden_size={hidden_size}"

            # Set hyperparameters
            config.training.learning_rate = lr
            config.model.hidden_size = hidden_size
            config.training.num_games = 500  # Shorter for sweep

            # Run experiment
            model = create_model(config, model_type='standard')
            trainer = Trainer(model, config)
            trainer.train()


def run_quick_test():
    """Quick test run to verify everything works"""
    config = Config()

    # Configure for quick test
    config.logging.wandb_name = "quick-test"
    config.logging.wandb_tags = ["test"]
    config.logging.wandb_mode = "online"  # Use "disabled" to skip wandb

    config.training.num_games = 100
    config.training.batch_size = 32
    config.evaluation.eval_games = 20
    config.evaluation.eval_interval = 25

    # Create and run
    model = create_model(config, model_type='standard')
    trainer = Trainer(model, config)
    trainer.train()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        experiment = sys.argv[1]

        if experiment == "baseline":
            run_baseline_experiment()
        elif experiment == "resnet":
            run_resnet_experiment()
        elif experiment == "sweep":
            run_hyperparameter_sweep()
        elif experiment == "test":
            run_quick_test()
        else:
            print(f"Unknown experiment: {experiment}")
            print("Available experiments: baseline, resnet, sweep, test")
    else:
        print("Running quick test...")
        print("Usage: python run_experiment.py [baseline|resnet|sweep|test]")
        run_quick_test()
