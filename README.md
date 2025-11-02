# Tic-Tac-Toe Neural Network (AlphaZero-style)

A clean, modular implementation of an AlphaZero-style neural network for playing Tic-Tac-Toe, with comprehensive logging, flexible configuration, and multiple player types including MCTS.

## Features

### Core Features
- ✅ **Two-headed neural network** (policy + value heads)
- ✅ **Self-play training** with experience replay
- ✅ **Temperature-based exploration** with multiple decay strategies
- ✅ **Canonical board representation** (current player always as 1)
- ✅ **Comprehensive logging** with Tensorboard support
- ✅ **Configurable architecture** with batch norm and dropout options
- ✅ **Data augmentation** through board symmetries (8x more data)
- ✅ **MCTS integration** for stronger play
- ✅ **Learning rate scheduling** (step, exponential, cosine)
- ✅ **Multiple optimizers** (Adam, AdamW, SGD)
- ✅ **Gradient clipping** for stability
- ✅ **Checkpoint management** with auto-cleanup
- ✅ **Best model tracking** based on evaluation scores

### Player Types
- **Human**: Interactive console player
- **Random**: Baseline random player
- **Neural**: Neural network with configurable temperature
- **MCTS**: Neural network enhanced with Monte Carlo Tree Search

## Project Structure

```
├── config.py           # Configuration management (all hyperparameters)
├── game.py            # Game logic (Board, Player, Game classes)
├── models.py          # Neural network architectures
├── neural_players.py  # Neural and MCTS players
├── trainer.py         # Training system with experience replay
├── logger.py          # Logging and metrics tracking
└── main.py           # Command-line interface
```

## Quick Start

### Installation
```bash
pip install torch numpy tensorboard
```

### Training

Quick test (20 games):
```bash
python main.py train --config-preset quick
```

Production training (10,000 games):
```bash
python main.py train --config-preset production
```

Debug mode (verbose logging):
```bash
python main.py train --config-preset debug
```

Custom configuration:
```bash
python main.py train --num-games 1000 --batch-size 64 --learning-rate 0.0001
```

### Evaluation

Evaluate against random player:
```bash
python main.py evaluate checkpoints/best_model.pt --num-games 100
```

Self-play evaluation:
```bash
python main.py evaluate checkpoints/best_model.pt --player2-type neural
```

MCTS vs MCTS:
```bash
python main.py evaluate checkpoints/best_model.pt \
    --player1-type mcts --player2-type mcts \
    --mcts-simulations 200
```

### Playing

Play against the AI:
```bash
python main.py play checkpoints/best_model.pt --human-first
```

Play against MCTS-enhanced AI:
```bash
python main.py play checkpoints/best_model.pt --ai-type mcts
```

### Analysis

Analyze model behavior:
```bash
python main.py analyze checkpoints/best_model.pt --test-perfect
```

## Configuration

All hyperparameters are centralized in `config.py`. Key settings:

### Model Architecture
```python
config.model.hidden_size = 128        # Hidden layer size
config.model.num_hidden_layers = 2    # Number of hidden layers
config.model.activation = "relu"      # Activation function
config.model.use_batch_norm = False   # Batch normalization
config.model.dropout_rate = 0.0       # Dropout (0 = disabled)
```

### Training
```python
config.training.learning_rate = 0.001
config.training.batch_size = 32
config.training.memory_size = 10000
config.training.use_augmentation = True    # 8x data through symmetries
config.training.gradient_clip = 1.0        # Gradient clipping
config.training.use_lr_scheduler = True    # Learning rate decay
```

### Temperature (Exploration)
```python
config.training.initial_temperature = 1.0
config.training.final_temperature = 0.1
config.training.temperature_decay = "cosine"  # linear/exponential/cosine
```

## Addressing Overfitting

The refactored code includes several features to help with the overfitting issue:

1. **Data Augmentation**: Enable with `use_augmentation=True` (8x more training data)
2. **Dropout**: Add regularization with `dropout_rate=0.2`
3. **Weight Decay**: L2 regularization with `weight_decay=1e-4`
4. **Learning Rate Scheduling**: Decay learning rate over time
5. **Smaller Network**: Reduce `hidden_size` to 64 or 32
6. **Batch Normalization**: Can help with training stability

### Recommended Configuration for Overfitting
```python
config.model.hidden_size = 64
config.model.dropout_rate = 0.2
config.training.weight_decay = 1e-4
config.training.use_augmentation = True
config.training.use_lr_scheduler = True
config.training.lr_scheduler_type = "cosine"
```

## Monitoring Training

### Console Output
- Game results and win rates
- Loss metrics (policy, value, total)
- Evaluation scores
- Learning rate updates

### Tensorboard
```bash
tensorboard --logdir logs/
```
Tracks:
- Loss curves
- Win rates over time
- Model parameters and gradients
- Evaluation metrics

### Log Files
- `training.log`: Detailed training log
- `training_summary.json`: Final statistics
- `config.json`: Configuration used

## Advanced Usage

### Resume Training
```bash
python main.py train --resume checkpoints/checkpoint_game_500.pt
```

### Custom Model Architecture
Create a ResNet-style model:
```bash
python main.py train --model-type resnet
```

### Save/Load Configurations
```python
# Save configuration
config.save("my_config.json")

# Load configuration
config = Config.load("my_config.json")
```

## Tips for Better Performance

1. **Start Simple**: Use fewer games and smaller networks initially
2. **Monitor Evaluation**: Watch performance against random baseline
3. **Check for Draws**: Perfect play should lead to 100% draws in self-play
4. **Temperature Tuning**: Higher early, lower late in training
5. **MCTS for Playing**: Use MCTS for strongest play after training

## Troubleshooting

### CUDA/GPU Issues
Force CPU usage:
```bash
python main.py train --cpu
```

### Memory Issues
- Reduce `memory_size` in config
- Reduce `batch_size`
- Use smaller `hidden_size`

### Slow Training
- Reduce `updates_per_training`
- Increase `games_per_update`
- Use GPU if available

## Future Improvements

Potential enhancements:
- [ ] Parallel self-play for faster data generation
- [ ] Tournament system for model selection
- [ ] Opening book generation
- [ ] Connect to larger games (Connect-4, Gomoku)
- [ ] Web interface for playing
- [ ] Model distillation for smaller inference models
- [ ] Curriculum learning (train against progressively stronger opponents)

## License

MIT License - feel free to use and modify!
