# Experiments

This directory contains experiment scripts for testing hypotheses and comparing models.

## Available Experiments

### 1. `run_experiment.py` - General Experiment Runner

Quick experiments for testing different configurations.

**Usage:**
```bash
# Quick test (100 games)
python experiments/run_experiment.py test

# Baseline experiment (1000 games)
python experiments/run_experiment.py baseline

# ResNet architecture
python experiments/run_experiment.py resnet

# Hyperparameter sweep
python experiments/run_experiment.py sweep
```

---

### 2. `layer_ablation.py` - Layer Count Ablation Study

**Hypothesis:** For Tic-Tac-Toe, adding more hidden layers doesn't improve performance.

Tests: 1, 2, 4, and 8 layer networks with multiple random seeds.

**Usage:**
```bash
# Quick test (200 games per config)
python experiments/layer_ablation.py quick

# Single experiment (e.g., 4 layers)
python experiments/layer_ablation.py single 4

# Full ablation (12 experiments, ~60-90 min)
python experiments/layer_ablation.py
```

**What gets tested:**
- Layer counts: [1, 2, 4, 8]
- Seeds: 3 per configuration
- Fixed: hidden_size=128, lr=0.001, all other hyperparameters

**Metrics tracked:**
- Final win rate vs random player
- Training time
- Number of parameters
- Loss curves

**View results:**
Go to wandb and filter by tag: `layer-ablation`

**Expected outcome:**
Shallow networks (1-2 layers) should perform similarly to deep networks (8 layers)
but train faster, since Tic-Tac-Toe is a simple game.

---

## Creating New Experiments

Template for a new experiment:

```python
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import create_model
from trainer import Trainer

def run_my_experiment():
    config = Config()

    # Configure wandb
    config.logging.wandb_name = "my-experiment"
    config.logging.wandb_tags = ["my-tag", "experiment-type"]
    config.logging.wandb_notes = "What I'm testing..."

    # Set hyperparameters
    # ...

    # Run
    model = create_model(config)
    trainer = Trainer(model, config)
    trainer.train()

if __name__ == "__main__":
    run_my_experiment()
```

---

## Tips for Good Experiments

1. **One variable at a time** - Only change what you're testing
2. **Multiple seeds** - Run 3+ seeds for statistical validity
3. **Use tags** - Makes filtering in wandb easy
4. **Document hypothesis** - Write what you expect to find
5. **Track time** - Faster training is valuable too
6. **Compare fairly** - Keep all other settings constant

---

## Analyzing Results in wandb

1. Go to: https://wandb.ai/your-username/tictactoe-rl
2. Filter by tag (e.g., `layer-ablation`)
3. Select multiple runs
4. Click "Compare" → See parallel coordinates plot
5. Group by hyperparameter to see trends

**Key charts to check:**
- `eval/random/win_rate` - Final performance
- `loss/total` - Learning progress
- Training time - Efficiency
- System metrics - Resource usage
