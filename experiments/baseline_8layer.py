#!/usr/bin/env python3
"""
Baseline 8-Layer Network Training

Train an 8-layer network with standard self-play to compare
against the multi-teacher approach.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import create_model
from trainer import Trainer


def run_baseline_8layer():
    """Train 8-layer network with baseline self-play"""
    print("="*70)
    print("BASELINE 8-LAYER SELF-PLAY TRAINING")
    print("="*70)

    config = Config()
    config.logging.wandb_name = "baseline-8layer-selfplay"
    config.logging.wandb_tags = ["baseline", "8-layers", "self-play", "comparison"]
    config.logging.wandb_notes = """
    Baseline 8-layer network trained with standard self-play.
    For comparison with multi-teacher training approach.
    Same hyperparameters as multi-teacher experiment.
    """

    # Model settings - 8 layers to match multi-teacher
    config.model.num_hidden_layers = 8
    config.model.hidden_size = 128

    # Training settings - match multi-teacher experiment
    config.training.num_games = 1500
    config.training.learning_rate = 0.001
    config.training.batch_size = 64
    config.training.temperature_decay = "cosine"

    # Evaluation settings
    config.evaluation.eval_games = 100
    config.evaluation.eval_interval = 100

    print("\n✓ Created 8-layer model")
    print("Training method: Pure self-play")
    print("="*70)

    # Create model and train
    model = create_model(config, model_type='standard')
    trainer = Trainer(model, config)
    trainer.train()


if __name__ == "__main__":
    run_baseline_8layer()
