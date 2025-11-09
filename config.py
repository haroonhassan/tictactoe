"""
Configuration file for Tic-Tac-Toe Neural Network Training
All hyperparameters and settings are centralized here for easy modification.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class ModelConfig:
    """Neural network architecture configuration"""
    input_size: int = 9
    hidden_size: int = 128
    num_hidden_layers: int = 2
    output_policy_size: int = 9
    output_value_size: int = 1
    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: float = 0.0  # 0.0 means no dropout


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Learning settings
    learning_rate: float = 0.001
    batch_size: int = 32
    memory_size: int = 10000

    # Training schedule
    num_games: int = 100
    games_per_update: int = 10
    updates_per_training: int = 10

    # Temperature settings for exploration
    initial_temperature: float = 1.0
    final_temperature: float = 0.1
    temperature_decay: str = "linear"  # "linear", "exponential", or "cosine"

    # Loss weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Optimizer settings
    optimizer: str = "adam"
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None

    # Data augmentation
    use_augmentation: bool = True  # Use board symmetries for 8x more data

    # Learning rate scheduling
    use_lr_scheduler: bool = False
    lr_scheduler_type: str = "step"  # "step", "exponential", "cosine"
    lr_decay_factor: float = 0.1  # For step scheduler
    lr_decay_steps: int = 1000  # For step scheduler
    lr_min: float = 1e-6  # Minimum learning rate


@dataclass
class GameConfig:
    """Game-specific settings"""
    board_size: int = 3
    player_tokens: tuple = ('X', 'O')
    empty_token: str = '-'


@dataclass
class EvaluationConfig:
    """Evaluation and testing settings"""
    eval_games: int = 100
    eval_interval: int = 50  # Evaluate every N training games
    opponents: list = field(default_factory=lambda: ["random", "self"])
    temperature_during_eval: float = 0.0  # Deterministic during evaluation


@dataclass
class LoggingConfig:
    """Logging and monitoring settings"""
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_file: str = "training.log"

    # Tensorboard settings
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"

    # Checkpoint settings
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 100
    keep_last_n_checkpoints: int = 5

    # Metrics logging
    log_metrics_interval: int = 10
    verbose: bool = True


@dataclass
class Config:
    """Master configuration combining all settings"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    game: GameConfig = field(default_factory=GameConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 42

    def save(self, path: str):
        """Save configuration to file"""
        import json
        import dataclasses

        def dataclass_to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        with open(path, 'w') as f:
            json.dump(dataclass_to_dict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        # Recursively create dataclass instances
        def dict_to_dataclass(data, cls):
            if not isinstance(data, dict):
                return data

            kwargs = {}
            for field_name, field_value in data.items():
                if field_name in cls.__dataclass_fields__:
                    field_type = cls.__dataclass_fields__[field_name].type
                    if hasattr(field_type, '__dataclass_fields__'):
                        kwargs[field_name] = dict_to_dataclass(
                            field_value, field_type)
                    else:
                        kwargs[field_name] = field_value
            return cls(**kwargs)

        return dict_to_dataclass(data, cls)


# Default configuration instance
default_config = Config()

# Preset configurations for different scenarios


def get_quick_test_config():
    """Quick testing configuration with minimal training"""
    config = Config()
    config.training.num_games = 20
    config.training.batch_size = 16
    config.training.memory_size = 1000
    config.evaluation.eval_games = 10
    return config


def get_production_config():
    """Production configuration with extensive training"""
    config = Config()
    config.model.hidden_size = 256
    config.training.num_games = 10000
    config.training.batch_size = 128
    config.training.memory_size = 50000
    config.training.learning_rate = 0.0001
    config.evaluation.eval_games = 500
    return config


def get_debug_config():
    """Debug configuration with verbose logging"""
    config = Config()
    config.logging.log_level = "DEBUG"
    config.logging.verbose = True
    config.training.num_games = 10
    config.evaluation.eval_games = 5
    return config
