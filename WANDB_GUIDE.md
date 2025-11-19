# Weights & Biases Integration Guide

This project now includes full Weights & Biases (wandb) integration for experiment tracking!

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Login to Weights & Biases

First time only:
```bash
wandb login
```

This will open your browser to get your API key. Paste it when prompted.

### 3. Run Your First Experiment

```bash
python run_experiment.py test
```

This runs a quick test experiment. When it starts, you'll see:
```
Weights & Biases initialized successfully
Run URL: https://wandb.ai/your-username/tictactoe-rl/runs/xxxxx
```

Click the URL to see your live dashboard! 📊

---

## 📊 What Gets Tracked

### Hyperparameters
All config settings are automatically logged:
- Model architecture (layers, activation, etc.)
- Training settings (learning rate, batch size, etc.)
- Temperature decay schedule
- Git commit hash (for reproducibility!)

### Training Metrics (logged every training update)
- `loss/total` - Combined loss
- `loss/policy` - Policy head loss
- `loss/value` - Value head loss
- `buffer/size` - Experience buffer size
- `buffer/win_rate` - Win rate in buffer

### Game Metrics (logged every 10 games)
- `game/moves` - Number of moves per game
- `game/temperature` - Current exploration temperature
- `game/winner` - Who won (+1=X, -1=O, 0=draw)

### Evaluation Metrics (logged every eval_interval)
- `eval/random/win_rate` - Win rate vs random player
- `eval/random/loss_rate` - Loss rate vs random
- `eval/random/draw_rate` - Draw rate vs random
- `eval/self/win_rate` - Win rate vs self (should be ~50%)

### Hyperparameters
- `hyperparameters/learning_rate` - Current LR (if using scheduler)

### Model Artifacts
- Best model checkpoints saved as wandb Artifacts

---

## 🧪 Running Experiments

### Baseline Experiment
```bash
python run_experiment.py baseline
```
Standard architecture, 1000 games.

### ResNet Experiment
```bash
python run_experiment.py resnet
```
Residual network architecture.

### Hyperparameter Sweep
```bash
python run_experiment.py sweep
```
Tests multiple learning rates and hidden sizes.

---

## ⚙️ Configuring Experiments

### In Code

```python
from config import Config
from models import create_model
from trainer import Trainer

# Create config
config = Config()

# Configure wandb settings
config.logging.wandb_name = "my-experiment"
config.logging.wandb_tags = ["resnet", "deep-learning"]
config.logging.wandb_notes = "Testing larger network"

# Set hyperparameters
config.model.hidden_size = 256
config.training.learning_rate = 0.0001
config.training.num_games = 5000

# Run
model = create_model(config, model_type='standard')
trainer = Trainer(model, config)
trainer.train()
```

### Disable wandb (for debugging)

```python
config.logging.use_wandb = False
# or
config.logging.wandb_mode = "disabled"
```

### Offline mode (no internet)

```python
config.logging.wandb_mode = "offline"
```

Then sync later:
```bash
wandb sync wandb/offline-run-xxxxx
```

---

## 📈 Analyzing Results

### Compare Experiments

1. Go to your project: https://wandb.ai/your-username/tictactoe-rl
2. Select multiple runs
3. Click "Compare" to see side-by-side metrics

### Parallel Coordinates Plot

Great for hyperparameter tuning:
- Shows how each hyperparameter affects win rate
- Identify best configurations visually

### Create Reports

Document your findings:
1. Click "Reports" in wandb
2. Drag in charts, add markdown notes
3. Share with collaborators

---

## 🎯 Best Practices

### 1. Use Descriptive Names
```python
config.logging.wandb_name = "resnet-4layer-cosine-decay-lr0.001"
```
Better than: "experiment-1"

### 2. Tag Your Runs
```python
config.logging.wandb_tags = ["resnet", "baseline", "production"]
```
Makes filtering easier.

### 3. Add Notes
```python
config.logging.wandb_notes = """
Hypothesis: Larger network should learn faster.
Testing 256 hidden units vs 128.
"""
```

### 4. Track Git Status
The integration automatically logs:
- Current commit hash
- Current branch
- Whether working directory is dirty

Always commit changes before important experiments!

### 5. Use Artifacts for Models
Best models are automatically saved as wandb Artifacts.
Download with:
```python
import wandb

run = wandb.init()
artifact = run.use_artifact('model-xxxxx:latest')
artifact_dir = artifact.download()
```

---

## 🔍 Example: Comparing Architectures

```python
# Experiment 1: Standard network
config1 = Config()
config1.logging.wandb_name = "standard-4layer"
config1.logging.wandb_tags = ["architecture-comparison", "standard"]
# ... train ...

# Experiment 2: ResNet
config2 = Config()
config2.logging.wandb_name = "resnet-4layer"
config2.logging.wandb_tags = ["architecture-comparison", "resnet"]
# ... train ...

# Then compare in wandb dashboard using the "architecture-comparison" tag
```

---

## 🐛 Troubleshooting

### "wandb not installed"
```bash
pip install wandb
```

### "Login required"
```bash
wandb login
```

### "No internet connection"
Use offline mode:
```python
config.logging.wandb_mode = "offline"
```

### Want to disable completely?
```python
config.logging.use_wandb = False
```

---

## 📚 Resources

- [Wandb Documentation](https://docs.wandb.ai/)
- [Best Practices for ML Experiments](https://wandb.ai/site/articles/ml-experiment-tracking-best-practices)
- [Wandb Artifacts Guide](https://docs.wandb.ai/guides/artifacts)

---

Happy experimenting! 🎉
