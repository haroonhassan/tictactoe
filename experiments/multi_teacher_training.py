#!/usr/bin/env python3
"""
Multi-Teacher Training Experiment

Train an 8-layer network against 3 opponents:
1. Best 4-layer defender (100% draws vs self)
2. Best 4-layer attacker (best vs random)
3. Random player

Goal: Learn both defensive and offensive strategies
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
from trainer import ExperienceBuffer, get_git_info


class MultiTeacherTrainer:
    """Train against multiple teacher networks + random"""

    def __init__(self, model, defender_model, attacker_model, config, logger: Optional[Logger] = None):
        self.model = model
        self.defender_model = defender_model  # Best defender
        self.attacker_model = attacker_model  # Best attacker
        self.config = config
        self.logger = logger or Logger(config)

        self.device = torch.device(config.device)
        self.model.to(self.device)

        if self.defender_model:
            self.defender_model.to(self.device)
            self.defender_model.eval()
        if self.attacker_model:
            self.attacker_model.to(self.device)
            self.attacker_model.eval()

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
                'experiment_type': 'multi_teacher',
                'model_type': 'standard',
                'num_hidden_layers': self.config.model.num_hidden_layers,
                'hidden_size': self.config.model.hidden_size,
                'learning_rate': self.config.training.learning_rate,
                'num_games': self.config.training.num_games,
                'teacher_mix': '33% defender, 33% attacker, 33% random',
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

    def _select_opponent_type(self) -> str:
        """Select opponent: 33% defender, 33% attacker, 33% random"""
        rand = random.random()
        if rand < 0.333:
            return 'defender'
        elif rand < 0.666:
            return 'attacker'
        else:
            return 'random'

    def _create_opponent(self, opponent_type: str, token: str):
        """Create opponent based on type"""
        if opponent_type == 'defender':
            return NeuralPlayer(token, self.defender_model, self.config,
                              temperature=0.0, device=self.device)
        elif opponent_type == 'attacker':
            return NeuralPlayer(token, self.attacker_model, self.config,
                              temperature=0.0, device=self.device)
        else:  # random
            return create_player('random', token)

    def _play_training_game(self) -> List[Dict[str, Any]]:
        """Play a training game"""
        board = Board(self.config.game.board_size)
        game_data = []

        opponent_type = self._select_opponent_type()

        # Randomly choose who goes first
        if random.random() < 0.5:
            player_x = NeuralPlayer('X', self.model, self.config,
                                   temperature=self._calculate_temperature(self.game_count),
                                   device=self.device)
            player_o = self._create_opponent(opponent_type, 'O')
            neural_is_x = True
        else:
            player_x = self._create_opponent(opponent_type, 'X')
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
        self.logger.logger.info("Starting multi-teacher training...")
        self.logger.logger.info("Teachers: 33% defender, 33% attacker, 33% random")

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


def run_multi_teacher_experiment():
    """Run multi-teacher training experiment"""
    print("="*70)
    print("MULTI-TEACHER TRAINING")
    print("="*70)

    # Load the teacher models
    print("\nLoading teacher models...")

    # Defender 1: best_model from Step 2 (4-layer, 100% draws vs self)
    defender_checkpoint = torch.load('checkpoints/best_model.pt', weights_only=False)
    defender_config = defender_checkpoint['config']
    defender_model = create_model(defender_config, model_type='standard')
    defender_model.load_state_dict(defender_checkpoint['model_state_dict'])
    print("✓ Loaded 4-layer defender (100% draws vs self)")

    # Defender 2: network_a from Step 1 (2-layer, 100% draws vs self at game 300)
    attacker_checkpoint = torch.load('checkpoints/network_a_2layer.pt', weights_only=False)
    attacker_config = attacker_checkpoint['config']
    attacker_model = create_model(attacker_config, model_type='standard')
    attacker_model.load_state_dict(attacker_checkpoint['model_state_dict'])
    print("✓ Loaded 2-layer network 'a' (100% draws vs self, 68% wins vs random)")

    # Create new 8-layer student model
    config = Config()
    config.logging.wandb_name = "multi-teacher-8layer"
    config.logging.wandb_tags = ["multi-teacher", "8-layers", "3-opponents"]
    config.logging.wandb_notes = """
    Training 8-layer network against 3 teachers:
    - 33% vs 4-layer defender (100% draws vs self, 40% wins vs random)
    - 33% vs 2-layer network 'a' (100% draws vs self, 68% wins vs random)
    - 33% vs random player

    Goal: Learn optimal defense AND better offense vs random
    """

    config.model.num_hidden_layers = 8
    config.model.hidden_size = 128

    config.training.num_games = 1500
    config.training.learning_rate = 0.001
    config.training.batch_size = 64
    config.training.temperature_decay = "cosine"

    config.evaluation.eval_games = 100
    config.evaluation.eval_interval = 100

    print("\n✓ Created 8-layer student model")
    print("\nTraining mix: 33% defender, 33% attacker, 33% random")
    print("="*70)

    # Create student model and train
    student_model = create_model(config, model_type='standard')
    trainer = MultiTeacherTrainer(student_model, defender_model, attacker_model, config)
    trainer.train()


if __name__ == "__main__":
    run_multi_teacher_experiment()
