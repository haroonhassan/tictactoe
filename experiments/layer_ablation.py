#!/usr/bin/env python3
"""
Ablation Study: Does adding more hidden layers improve learning?

Hypothesis: For a simple game like Tic-Tac-Toe, deeper networks (more layers)
won't significantly improve performance and may actually slow down learning.

Experiment Design:
- Test layer counts: [1, 2, 4, 8]
- Keep everything else constant (hidden_size, lr, etc.)
- Run multiple seeds for statistical validity
- Track: final win rate, time to convergence, training time

Expected Result: Shallow networks (1-2 layers) should perform similarly to
deep networks (8 layers) for this simple task, but train faster.
"""

import sys
import os
# Add parent directory to path so we can import from root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import Config
from models import create_model
from trainer import Trainer
import time


def run_layer_experiment(num_layers: int, seed: int = 42, run_name_suffix: str = ""):
    """
    Run a single experiment with specified number of layers.

    Args:
        num_layers: Number of hidden layers to use
        seed: Random seed for reproducibility
        run_name_suffix: Optional suffix for run name

    Returns:
        dict: Results including final win rate and training time
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create config
    config = Config()
    config.seed = seed

    # Configure wandb
    config.logging.wandb_name = f"layers-{num_layers}-seed{seed}{run_name_suffix}"
    config.logging.wandb_tags = [
        "layer-ablation",           # Main experiment tag
        f"{num_layers}-layers",      # Layer count
        f"seed-{seed}",              # Seed for grouping
        "ablation-study"             # General category
    ]
    config.logging.wandb_notes = f"""
    Hypothesis: More layers don't help for Tic-Tac-Toe.
    Testing {num_layers} hidden layers.
    Seed: {seed}
    """

    # Model configuration - ONLY vary num_layers
    config.model.num_hidden_layers = num_layers
    config.model.hidden_size = 128        # Fixed
    config.model.activation = "relu"      # Fixed
    config.model.use_batch_norm = False   # Fixed
    config.model.dropout_rate = 0.0       # Fixed

    # Training configuration - ALL FIXED
    config.training.num_games = 1000
    config.training.learning_rate = 0.001
    config.training.batch_size = 64
    config.training.memory_size = 10000
    config.training.temperature_decay = "cosine"
    config.training.use_augmentation = True
    config.training.games_per_update = 10
    config.training.updates_per_training = 10

    # Evaluation configuration
    config.evaluation.eval_games = 100
    config.evaluation.eval_interval = 100

    # Create model and trainer
    print(f"\n{'='*60}")
    print(f"Running experiment: {num_layers} layers, seed {seed}")
    print(f"{'='*60}")

    model = create_model(config, model_type='standard')
    trainer = Trainer(model, config)

    # Track training time
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Return results
    return {
        'num_layers': num_layers,
        'seed': seed,
        'final_win_rate': trainer.best_eval_score,
        'training_time_minutes': training_time / 60,
        'total_parameters': model.get_num_parameters()
    }


def run_ablation_study(layer_counts=[1, 2, 4, 8], num_seeds=3):
    """
    Run full ablation study across multiple layer counts and seeds.

    Args:
        layer_counts: List of layer counts to test
        num_seeds: Number of random seeds to run per configuration
    """
    print("="*70)
    print("LAYER ABLATION STUDY")
    print("="*70)
    print(f"Testing layer counts: {layer_counts}")
    print(f"Random seeds per config: {num_seeds}")
    print(f"Total experiments: {len(layer_counts) * num_seeds}")
    print("="*70)

    results = []

    for num_layers in layer_counts:
        for seed in range(42, 42 + num_seeds):  # Seeds: 42, 43, 44, ...
            result = run_layer_experiment(num_layers, seed)
            results.append(result)

            print(f"\nCompleted: {num_layers} layers, seed {seed}")
            print(f"  Win rate: {result['final_win_rate']:.1%}")
            print(f"  Time: {result['training_time_minutes']:.1f} min")
            print(f"  Parameters: {result['total_parameters']:,}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for num_layers in layer_counts:
        layer_results = [r for r in results if r['num_layers'] == num_layers]
        win_rates = [r['final_win_rate'] for r in layer_results]
        times = [r['training_time_minutes'] for r in layer_results]
        params = layer_results[0]['total_parameters']

        print(f"\n{num_layers} layers ({params:,} parameters):")
        print(f"  Win rate: {np.mean(win_rates):.1%} ± {np.std(win_rates):.1%}")
        print(f"  Time: {np.mean(times):.1f} ± {np.std(times):.1f} min")

    print("\n" + "="*70)
    print("View all runs in wandb:")
    print("https://wandb.ai/haroon-hassan-personal/tictactoe-rl")
    print("\nFilter by tag: 'layer-ablation'")
    print("="*70)


def run_quick_ablation():
    """Quick version with fewer games for testing"""
    print("Running QUICK ablation study (for testing)")

    layer_counts = [1, 2, 4, 8]
    seed = 42

    for num_layers in layer_counts:
        config = Config()
        config.seed = seed

        # Quick test config
        config.training.num_games = 200  # Fewer games
        config.evaluation.eval_games = 50
        config.evaluation.eval_interval = 50

        config.logging.wandb_name = f"quick-ablation-{num_layers}layers"
        config.logging.wandb_tags = ["quick-test", "layer-ablation", f"{num_layers}-layers"]

        config.model.num_hidden_layers = num_layers
        config.model.hidden_size = 128

        model = create_model(config, model_type='standard')
        trainer = Trainer(model, config)
        trainer.train()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test: python experiments/layer_ablation.py quick
        run_quick_ablation()

    elif len(sys.argv) > 1 and sys.argv[1] == "single":
        # Single run: python experiments/layer_ablation.py single 4
        num_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        run_layer_experiment(num_layers, seed=42)

    else:
        # Full ablation study
        print("\n" + "="*70)
        print("This will run 12 experiments (4 layer counts × 3 seeds)")
        print("Estimated time: ~60-90 minutes")
        print("="*70)

        response = input("\nContinue? (y/n): ")
        if response.lower() == 'y':
            run_ablation_study(
                layer_counts=[1, 2, 4, 8],
                num_seeds=3
            )
        else:
            print("Cancelled. Run 'python experiments/layer_ablation.py quick' for a quick test.")
