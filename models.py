"""
Refactored neural network models with improved architecture and flexibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import numpy as np


class TicTacToeNet(nn.Module):
    """
    Enhanced neural network for Tic-Tac-Toe with configurable architecture.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Build the network layers
        layers = []
        input_size = config.model.input_size

        # Hidden layers
        for i in range(config.model.num_hidden_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, config.model.hidden_size))
            else:
                layers.append(nn.Linear(config.model.hidden_size,
                              config.model.hidden_size))

            # Add batch normalization if configured
            if hasattr(config.model, 'use_batch_norm') and config.model.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.model.hidden_size))

            # Add activation
            if config.model.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.model.activation == 'tanh':
                layers.append(nn.Tanh())
            elif config.model.activation == 'gelu':
                layers.append(nn.GELU())

            # Add dropout if configured
            if hasattr(config.model, 'dropout_rate') and config.model.dropout_rate > 0:
                layers.append(nn.Dropout(config.model.dropout_rate))

        self.shared_layers = nn.Sequential(*layers)

        # Output heads
        self.policy_head = self._build_policy_head(config)
        self.value_head = self._build_value_head(config)

        # Initialize weights
        self._initialize_weights()

    def _build_policy_head(self, config) -> nn.Module:
        """Build the policy output head"""
        return nn.Sequential(
            nn.Linear(config.model.hidden_size,
                      config.model.output_policy_size)
        )

    def _build_value_head(self, config) -> nn.Module:
        """Build the value output head"""
        return nn.Sequential(
            nn.Linear(config.model.hidden_size,
                      config.model.output_value_size),
            nn.Tanh()  # Value should be in [-1, 1]
        )

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            value: Tensor of shape (batch_size, 1) with value estimates
            policy: Tensor of shape (batch_size, output_policy_size) with policy logits
        """
        # Shared layers
        features = self.shared_layers(x)

        # Output heads
        value = self.value_head(features)
        policy = self.policy_head(features)

        return value, policy

    def predict(self, board_state: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        Make a prediction for a single board state.

        Returns:
            value: Scalar value estimate
            policy: Numpy array of move probabilities
        """
        self.eval()
        with torch.no_grad():
            if board_state.dim() == 1:
                board_state = board_state.unsqueeze(0)

            value, policy_logits = self(board_state)
            policy_probs = F.softmax(policy_logits, dim=1)

            return value.item(), policy_probs.squeeze().cpu().numpy()

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_summary(self) -> str:
        """Get a summary of the network architecture"""
        summary = []
        summary.append(f"TicTacToeNet Architecture:")
        summary.append(f"  Input size: {self.config.model.input_size}")
        summary.append(
            f"  Hidden layers: {self.config.model.num_hidden_layers}")
        summary.append(f"  Hidden size: {self.config.model.hidden_size}")
        summary.append(f"  Activation: {self.config.model.activation}")
        summary.append(
            f"  Output policy size: {self.config.model.output_policy_size}")
        summary.append(
            f"  Output value size: {self.config.model.output_value_size}")
        summary.append(f"  Total parameters: {self.get_num_parameters():,}")
        return "\n".join(summary)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""

    def __init__(self, hidden_size: int, activation: str = 'relu'):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = x + residual
        x = self.activation(x)
        return x


class ResNetTicTacToe(TicTacToeNet):
    """
    Residual network variant for Tic-Tac-Toe.
    Useful for experimenting with deeper architectures.
    """

    def __init__(self, config):
        # Temporarily modify config to skip parent's layer building
        original_num_layers = config.model.num_hidden_layers
        config.model.num_hidden_layers = 0
        super().__init__(config)
        config.model.num_hidden_layers = original_num_layers

        # Build residual layers
        layers = []

        # Initial projection
        layers.append(nn.Linear(config.model.input_size,
                      config.model.hidden_size))
        layers.append(nn.BatchNorm1d(config.model.hidden_size))
        layers.append(nn.ReLU())

        # Residual blocks
        for _ in range(config.model.num_hidden_layers):
            layers.append(ResidualBlock(
                config.model.hidden_size, config.model.activation))

        self.shared_layers = nn.Sequential(*layers)

        # Reinitialize weights
        self._initialize_weights()


def create_model(config, model_type: str = 'standard') -> TicTacToeNet:
    """
    Factory function to create different model types.

    Args:
        config: Configuration object
        model_type: Type of model ('standard' or 'resnet')
    """
    if model_type == 'standard':
        return TicTacToeNet(config)
    elif model_type == 'resnet':
        return ResNetTicTacToe(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_checkpoint(checkpoint_path: str, config=None) -> Tuple[TicTacToeNet, Dict[str, Any]]:
    """
    Load a model from a checkpoint.

    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Use saved config if not provided
    if config is None:
        config = checkpoint['config']

    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint
