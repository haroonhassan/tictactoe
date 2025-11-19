#!/usr/bin/env python3
"""
Multi-Opponent Training Experiment

Goal: Train a network that:
- Has 100% draw rate vs self (optimal play)
- Has 0% loss rate vs random (perfect defense)
- Maximizes win rate vs random (exploits weaknesses)

Method: Train against multiple opponent types instead of pure self-play
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from config import Config
from models import create_model
from game import Board, Game
from neural_players import NeuralPlayer, create_player
from logger import Logger


class MultiOpponentTrainer:
    """
    Trainer that plays against multiple opponent types to learn
    both offensive and defensive strategies.
    """

    def __init__(self, model, config, logger: Optional[Logger] = None):
        self.model = model
        self.config = config
        self.logger = logger or Logger(config)

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Experience buffer
        from trainer import ExperienceBuffer
        self.experience_buffer = ExperienceBuffer(config.training.memory_size)

        # Training statistics
        self.game_count = 0
        self.training_steps = 0
        self.best_eval_score = 0

        # Opponent mix configuration
        self.opponent_mix = config.training.opponent_mix if hasattr(config.training, 'opponent_mix') else {
            'self': 0.5,      # 50% self-play
            'random': 0.3,    # 30% random
            'deterministic': 0.2  # 20% weak deterministic bot
        }

        # Initialize wandb
        self.use_wandb = config.logging.use_wandb
        if self.use_wandb:
            self._init_wandb()

        self.logger.logger.info(f"Multi-opponent training mix: {self.opponent_mix}")
        self.logger.logger.info(self.model.get_architecture_summary())

    def _create_optimizer(self):
        """Create optimizer"""
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay

        if self.config.training.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _init_wandb(self):
        """Initialize wandb"""
        try:
            import wandb
            from trainer import get_git_info
            self.wandb = wandb

            git_info = get_git_info()

            wandb_config = {
                'experiment_type': 'multi_opponent',
                'opponent_mix': self.opponent_mix,
                'model_type': 'standard',
                'num_hidden_layers': self.config.model.num_hidden_layers,
                'hidden_size': self.config.model.hidden_size,
                'learning_rate': self.config.training.learning_rate,
                'num_games': self.config.training.num_games,
                'git_commit': git_info['commit'],
            }

            wandb.init(
                project=self.config.logging.wandb_project,
                name=self.config.logging.wandb_name,
                tags=self.config.logging.wandb_tags,
                config=wandb_config,
                mode=self.config.logging.wandb_mode
            )

            wandb.watch(self.model, log='all', log_freq=100)
            self.logger.logger.info("Weights & Biases initialized")

        except ImportError:
            self.logger.logger.warning("wandb not installed")
            self.use_wandb = False

    def _select_opponent_type(self) -> str:
        """Randomly select opponent type based on configured mix"""
        rand = random.random()
        cumulative = 0.0

        for opponent_type, probability in self.opponent_mix.items():
            cumulative += probability
            if rand < cumulative:
                return opponent_type

        return 'self'  # Fallback

    def _create_opponent(self, opponent_type: str, token: str):
        """Create an opponent of the specified type"""
        if opponent_type == 'self':
            # Self-play with exploration temperature
            temp = self._calculate_temperature(self.game_count)
            return NeuralPlayer(token, self.model, self.config, temperature=temp, device=self.device)

        elif opponent_type == 'random':
            # Pure random player
            return create_player('random', token)

        elif opponent_type == 'deterministic':
            # Weak deterministic bot (always plays top-left available)
            from game import Player
            class DeterministicBot(Player):
                def get_move(self, board):
                    """Always pick first available square (top-left to bottom-right)"""
                    for row in range(board.size):
                        for col in range(board.size):
                            if board.grid[row][col] == '-':
                                return (row, col)
                    return (0, 0)  # Should never happen

            return DeterministicBot(token, "Deterministic")

        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

    def _calculate_temperature(self, game_num: int) -> float:
        """Calculate exploration temperature"""
        progress = game_num / self.config.training.num_games
        initial = self.config.training.initial_temperature
        final = self.config.training.final_temperature

        if self.config.training.temperature_decay == 'linear':
            return initial - (initial - final) * progress
        elif self.config.training.temperature_decay == 'exponential':
            return initial * (final / initial) ** progress
        elif self.config.training.temperature_decay == 'cosine':
            return final + (initial - final) * 0.5 * (1 + np.cos(np.pi * progress))
        else:
            return initial

    def _play_training_game(self) -> List[Dict[str, Any]]:
        """Play a training game against a random opponent type"""
        board = Board(self.config.game.board_size)
        game_data = []

        # Select opponent type for this game
        opponent_type = self._select_opponent_type()

        # Randomly choose who goes first
        if random.random() < 0.5:
            # Neural player goes first as X
            player_x = NeuralPlayer('X', self.model, self.config,
                                   temperature=self._calculate_temperature(self.game_count),
                                   device=self.device)
            player_o = self._create_opponent(opponent_type, 'O')
            neural_is_x = True
        else:
            # Opponent goes first as X
            player_x = self._create_opponent(opponent_type, 'X')
            player_o = NeuralPlayer('O', self.model, self.config,
                                   temperature=self._calculate_temperature(self.game_count),
                                   device=self.device)
            neural_is_x = False

        players = {'X': player_x, 'O': player_o}
        current_token = 'X'

        while True:
            current_player = players[current_token]

            # Only record data for neural player's moves
            is_neural_turn = (current_token == 'X' and neural_is_x) or (current_token == 'O' and not neural_is_x)

            if is_neural_turn:
                # Get board state before move
                canonical = board.to_flat_canonical(current_token)
                state_tensor = torch.tensor(canonical, dtype=torch.float32)

                # Get move from player
                row, col = current_player.get_move(board)

                # Create policy target
                policy = torch.zeros(self.config.model.output_policy_size)
                move_idx = row * board.size + col
                policy[move_idx] = 1.0

                # Store experience
                game_data.append({
                    'state': state_tensor,
                    'policy': policy,
                    'player': current_token,
                    'value': None,  # Will be filled after game ends
                    'opponent_type': opponent_type
                })
            else:
                # Just execute opponent's move
                row, col = current_player.get_move(board)

            # Make move
            board.make_move(row, col, current_token)

            # Check game end
            result = board.get_game_result()
            if result:
                # Assign values based on game result
                neural_token = 'X' if neural_is_x else 'O'
                for experience in game_data:
                    if result == 'D':
                        experience['value'] = torch.tensor([0.0], dtype=torch.float32)
                    elif result == neural_token:
                        experience['value'] = torch.tensor([1.0], dtype=torch.float32)
                    else:
                        experience['value'] = torch.tensor([-1.0], dtype=torch.float32)

                    experience['winner'] = result

                break

            # Switch player
            current_token = 'O' if current_token == 'X' else 'X'

        return game_data

    def train(self):
        """Main training loop"""
        self.logger.logger.info("Starting multi-opponent training...")

        for game_num in range(1, self.config.training.num_games + 1):
            self.game_count = game_num

            # Play training game
            game_data = self._play_training_game()
            if game_data:  # Only add if neural player made moves
                self.experience_buffer.add_batch(game_data)

                # Log game result
                winner = game_data[-1]['winner']
                opponent_type = game_data[0]['opponent_type']
                self.logger.logger.info(
                    f"Game {game_num}: vs {opponent_type}, "
                    f"Result: {winner}, Moves: {len(game_data)}"
                )

            # Training step
            if game_num % self.config.training.games_per_update == 0:
                self._training_update()

            # Evaluation
            if game_num % self.config.evaluation.eval_interval == 0:
                self._evaluate()

            # Checkpoint
            if game_num % self.config.logging.checkpoint_interval == 0:
                self._save_checkpoint()

        # Final evaluation
        self._evaluate()
        self._save_checkpoint()

        if self.use_wandb:
            self.wandb.finish()

        self.logger.close()

    def _training_update(self):
        """Perform training updates"""
        if len(self.experience_buffer) < self.config.training.batch_size:
            return

        losses = []
        for _ in range(self.config.training.updates_per_training):
            loss_dict = self._train_step()
            if loss_dict:
                losses.append(loss_dict)
                self.training_steps += 1

        if losses:
            avg_losses = {
                'total': np.mean([l['total'] for l in losses]),
                'policy': np.mean([l['policy'] for l in losses]),
                'value': np.mean([l['value'] for l in losses])
            }

            if self.use_wandb:
                self.wandb.log({
                    'game': self.game_count,
                    'loss/total': avg_losses['total'],
                    'loss/policy': avg_losses['policy'],
                    'loss/value': avg_losses['value'],
                })

    def _train_step(self):
        """Single training step"""
        batch = self.experience_buffer.sample(self.config.training.batch_size)
        if not batch:
            return None

        states = torch.stack([exp['state'] for exp in batch]).to(self.device)
        target_policies = torch.stack([exp['policy'] for exp in batch]).to(self.device)
        target_values = torch.stack([exp['value'] for exp in batch]).to(self.device)

        self.model.train()
        pred_values, pred_policies = self.model(states)

        policy_indices = target_policies.argmax(dim=1)
        policy_loss = F.cross_entropy(pred_policies, policy_indices)
        value_loss = F.mse_loss(pred_values, target_values)

        total_loss = (self.config.training.policy_loss_weight * policy_loss +
                     self.config.training.value_loss_weight * value_loss)

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.config.training.gradient_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                    self.config.training.gradient_clip)

        self.optimizer.step()

        return {
            'total': total_loss.item(),
            'policy': policy_loss.item(),
            'value': value_loss.item()
        }

    def _evaluate(self):
        """Evaluate model performance"""
        from trainer import Trainer
        # Use standard trainer's evaluation
        temp_trainer = Trainer(self.model, self.config, self.logger)
        temp_trainer.game_count = self.game_count
        temp_trainer.use_wandb = self.use_wandb
        if self.use_wandb:
            temp_trainer.wandb = self.wandb
        temp_trainer._evaluate()

        if temp_trainer.best_eval_score > self.best_eval_score:
            self.best_eval_score = temp_trainer.best_eval_score

    def _save_checkpoint(self):
        """Save checkpoint"""
        self.logger.save_checkpoint(
            self.model,
            self.optimizer,
            self.game_count,
            self.config
        )


def run_multi_opponent_experiment():
    """Run multi-opponent training experiment"""
    config = Config()

    # Configure experiment
    config.logging.wandb_name = "multi-opponent-training"
    config.logging.wandb_tags = ["multi-opponent", "perfect-defense", "experiment"]
    config.logging.wandb_notes = """
    Training against multiple opponent types:
    - 50% self-play (learn optimal strategy)
    - 30% random (learn to exploit randomness)
    - 20% deterministic bot (learn to punish patterns)

    Goal: Achieve 100% draw rate vs self AND 0% loss rate vs random
    """

    # Model settings - use 4 layers (best from ablation)
    config.model.num_hidden_layers = 4
    config.model.hidden_size = 128

    # Training settings
    config.training.num_games = 1000
    config.training.learning_rate = 0.001
    config.training.batch_size = 64
    config.training.temperature_decay = "cosine"

    # Opponent mix
    config.training.opponent_mix = {
        'self': 0.5,
        'random': 0.3,
        'deterministic': 0.2
    }

    # Evaluation
    config.evaluation.eval_games = 100
    config.evaluation.eval_interval = 100

    # Create model and trainer
    model = create_model(config, model_type='standard')
    trainer = MultiOpponentTrainer(model, config)

    # Run training
    trainer.train()


if __name__ == "__main__":
    run_multi_opponent_experiment()
