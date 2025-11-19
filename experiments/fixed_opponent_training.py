#!/usr/bin/env python3
"""
Fixed Opponent Training Experiment

Step 1: Train network 'a' to achieve 100% draws vs self (game-theory optimal)
Step 2: Train new network vs network 'a' (50%) + vs random (50%)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import numpy as np
from typing import Dict, List, Optional, Any

from config import Config
from models import create_model
from game import Board
from neural_players import NeuralPlayer, create_player
from logger import Logger
from trainer import Trainer, ExperienceBuffer, get_git_info


class FixedOpponentTrainer:
    """Train against a fixed opponent network + random"""

    def __init__(self, model, opponent_model, config, logger: Optional[Logger] = None):
        self.model = model
        self.opponent_model = opponent_model  # Fixed network 'a'
        self.config = config
        self.logger = logger or Logger(config)

        self.device = torch.device(config.device)
        self.model.to(self.device)

        if self.opponent_model:
            self.opponent_model.to(self.device)
            self.opponent_model.eval()  # Keep opponent frozen

        self.optimizer = self._create_optimizer()
        self.experience_buffer = ExperienceBuffer(config.training.memory_size)

        self.game_count = 0
        self.training_steps = 0
        self.best_eval_score = 0

        self.use_wandb = config.logging.use_wandb
        if self.use_wandb:
            self._init_wandb()

        self.logger.logger.info(self.model.get_architecture_summary())

    def _create_optimizer(self):
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        if self.config.training.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def _init_wandb(self):
        try:
            import wandb
            self.wandb = wandb
            git_info = get_git_info()

            wandb_config = {
                'experiment_type': 'fixed_opponent',
                'model_type': 'standard',
                'num_hidden_layers': self.config.model.num_hidden_layers,
                'hidden_size': self.config.model.hidden_size,
                'learning_rate': self.config.training.learning_rate,
                'num_games': self.config.training.num_games,
                'opponent_mix': 0.5,  # 50% fixed opponent, 50% random
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

    def _calculate_temperature(self, game_num: int) -> float:
        progress = game_num / self.config.training.num_games
        initial = self.config.training.initial_temperature
        final = self.config.training.final_temperature

        if self.config.training.temperature_decay == 'cosine':
            return final + (initial - final) * 0.5 * (1 + np.cos(np.pi * progress))
        return initial

    def _play_training_game(self) -> List[Dict[str, Any]]:
        board = Board(self.config.game.board_size)
        game_data = []

        # 50% vs opponent network 'a', 50% vs random
        if random.random() < 0.5:
            opponent_type = 'network_a'
        else:
            opponent_type = 'random'

        # Randomly choose who goes first
        if random.random() < 0.5:
            player_x = NeuralPlayer('X', self.model, self.config,
                                   temperature=self._calculate_temperature(self.game_count),
                                   device=self.device)
            if opponent_type == 'network_a':
                player_o = NeuralPlayer('O', self.opponent_model, self.config,
                                       temperature=0.0,  # Deterministic
                                       device=self.device)
            else:
                player_o = create_player('random', 'O')
            neural_is_x = True
        else:
            if opponent_type == 'network_a':
                player_x = NeuralPlayer('X', self.opponent_model, self.config,
                                       temperature=0.0,  # Deterministic
                                       device=self.device)
            else:
                player_x = create_player('random', 'X')
            player_o = NeuralPlayer('O', self.model, self.config,
                                   temperature=self._calculate_temperature(self.game_count),
                                   device=self.device)
            neural_is_x = False

        players = {'X': player_x, 'O': player_o}
        current_token = 'X'

        while True:
            current_player = players[current_token]
            is_neural_turn = (current_token == 'X' and neural_is_x) or (current_token == 'O' and not neural_is_x)

            if is_neural_turn:
                canonical = board.to_flat_canonical(current_token)
                state_tensor = torch.tensor(canonical, dtype=torch.float32)
                row, col = current_player.get_move(board)

                policy = torch.zeros(self.config.model.output_policy_size)
                move_idx = row * board.size + col
                policy[move_idx] = 1.0

                game_data.append({
                    'state': state_tensor,
                    'policy': policy,
                    'player': current_token,
                    'value': None,
                    'opponent_type': opponent_type
                })
            else:
                row, col = current_player.get_move(board)

            board.make_move(row, col, current_token)

            result = board.get_game_result()
            if result:
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

            current_token = 'O' if current_token == 'X' else 'X'

        return game_data

    def train(self):
        self.logger.logger.info("Starting training vs fixed opponent...")

        for game_num in range(1, self.config.training.num_games + 1):
            self.game_count = game_num

            game_data = self._play_training_game()
            if game_data:
                self.experience_buffer.add_batch(game_data)
                winner = game_data[-1]['winner']
                opponent_type = game_data[0]['opponent_type']
                self.logger.logger.info(
                    f"Game {game_num}: vs {opponent_type}, "
                    f"Result: {winner}, Moves: {len(game_data)}"
                )

            if game_num % self.config.training.games_per_update == 0:
                self._training_update()

            if game_num % self.config.evaluation.eval_interval == 0:
                self._evaluate()

            if game_num % self.config.logging.checkpoint_interval == 0:
                self._save_checkpoint()

        self._evaluate()
        self._save_checkpoint()

        if self.use_wandb:
            self.wandb.finish()

        self.logger.close()

    def _training_update(self):
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
        from trainer import Trainer
        temp_trainer = Trainer(self.model, self.config, self.logger)
        temp_trainer.game_count = self.game_count
        temp_trainer.use_wandb = self.use_wandb
        if self.use_wandb:
            temp_trainer.wandb = self.wandb
        temp_trainer._evaluate()

        if temp_trainer.best_eval_score > self.best_eval_score:
            self.best_eval_score = temp_trainer.best_eval_score

    def _save_checkpoint(self):
        self.logger.save_checkpoint(
            self.model,
            self.optimizer,
            self.game_count,
            self.config
        )


def train_baseline_network_a():
    """Train network 'a': achieve 100% draws vs self (2-layer network)"""
    print("="*70)
    print("STEP 1: Training network 'a' (2-layer, 100% draws vs self)")
    print("="*70)

    config = Config()
    config.logging.wandb_name = "network-a-2layer-baseline"
    config.logging.wandb_tags = ["network-a", "baseline", "2-layers", "self-play"]
    config.logging.wandb_notes = "Training 2-layer network 'a' to achieve 100% draws vs self"

    config.model.num_hidden_layers = 2  # 2 layers converges to 100% draws
    config.model.hidden_size = 128

    config.training.num_games = 1000  # More games for 2-layer to converge
    config.training.learning_rate = 0.001
    config.training.batch_size = 64
    config.training.temperature_decay = "cosine"

    config.evaluation.eval_games = 100
    config.evaluation.eval_interval = 100

    model = create_model(config, model_type='standard')
    trainer = Trainer(model, config)
    trainer.train()

    # Save network 'a'
    checkpoint_path = 'checkpoints/network_a_2layer.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, checkpoint_path)
    print(f"\n✓ Network 'a' (2-layer) saved to {checkpoint_path}")

    return model, config


def train_vs_network_a(network_a, config_a):
    """Train new network: 50% vs network 'a', 50% vs random"""
    print("\n" + "="*70)
    print("STEP 2: Training new network (50% vs network 'a', 50% vs random)")
    print("="*70)

    config = Config()
    config.logging.wandb_name = "vs-network-a-50-50"
    config.logging.wandb_tags = ["fixed-opponent", "network-a", "experiment"]
    config.logging.wandb_notes = """
    Training new network against fixed network 'a' (100% draws vs self) + random.
    Mix: 50% vs network 'a', 50% vs random
    """

    config.model.num_hidden_layers = 4
    config.model.hidden_size = 128

    config.training.num_games = 1000
    config.training.learning_rate = 0.001
    config.training.batch_size = 64
    config.training.temperature_decay = "cosine"

    config.evaluation.eval_games = 100
    config.evaluation.eval_interval = 100

    # Create new model
    model = create_model(config, model_type='standard')

    # Train with fixed opponent
    trainer = FixedOpponentTrainer(model, network_a, config)
    trainer.train()

    return model


if __name__ == "__main__":
    # Step 1: Train network 'a' (2-layer, 100% draws vs self)
    print("\nTraining 2-layer network 'a' for 100% draws vs self...")
    network_a, config_a = train_baseline_network_a()

    # Step 2: Automatically proceed to train new 4-layer network
    print("\n" + "="*70)
    print("Network 'a' training complete. Starting Step 2...")
    print("="*70)

    train_vs_network_a(network_a, config_a)
