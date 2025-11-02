"""
Refactored training system with improved structure, telemetry, and evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from collections import deque
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time

from game import Board, Game
from neural_players import NeuralPlayer, create_player
from logger import Logger


class ExperienceBuffer:
    """
    Experience replay buffer for training data.
    """
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add(self, experience: Dict[str, Any]):
        """Add a single experience"""
        self.buffer.append(experience)
    
    def add_batch(self, experiences: List[Dict[str, Any]]):
        """Add multiple experiences"""
        self.buffer.extend(experiences)
    
    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample a batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if not self.buffer:
            return {'size': 0, 'occupancy': 0.0}
        
        # Sample recent experiences for value distribution
        recent = list(self.buffer)[-min(100, len(self.buffer)):]
        values = [exp['value'].item() for exp in recent]
        
        return {
            'size': len(self.buffer),
            'occupancy': len(self.buffer) / self.max_size,
            'value_mean': np.mean(values),
            'value_std': np.std(values),
            'win_rate': sum(1 for v in values if v > 0.5) / len(values),
            'loss_rate': sum(1 for v in values if v < -0.5) / len(values),
            'draw_rate': sum(1 for v in values if -0.5 <= v <= 0.5) / len(values)
        }


class Trainer:
    """
    Enhanced trainer with better organization and telemetry.
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
        
        # Setup learning rate scheduler
        self.scheduler = self._create_lr_scheduler() if config.training.use_lr_scheduler else None
        
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(config.training.memory_size)
        
        # Training statistics
        self.game_count = 0
        self.training_steps = 0
        self.best_eval_score = 0
        
        # Log model architecture
        self.logger.logger.info(self.model.get_architecture_summary())
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        params = self.model.parameters()
        lr = self.config.training.learning_rate
        weight_decay = self.config.training.weight_decay
        
        if self.config.training.optimizer == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif self.config.training.optimizer == 'sgd':
            return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif self.config.training.optimizer == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler based on configuration"""
        config = self.config.training
        
        if config.lr_scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=config.lr_decay_steps,
                gamma=config.lr_decay_factor
            )
        elif config.lr_scheduler_type == 'exponential':
            # Calculate gamma to reach lr_min after all games
            gamma = (config.lr_min / config.learning_rate) ** (1 / config.num_games)
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif config.lr_scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_games,
                eta_min=config.lr_min
            )
        else:
            raise ValueError(f"Unknown scheduler type: {config.lr_scheduler_type}")
    
    def train(self):
        """Main training loop"""
        self.logger.logger.info("Starting training...")
        self.logger.logger.info(f"Configuration: {self.config.training}")
        
        for game_num in range(1, self.config.training.num_games + 1):
            self.game_count = game_num
            
            # Calculate temperature for this game
            temperature = self._calculate_temperature(game_num)
            
            # Play self-play game
            game_data = self._play_self_play_game(temperature)
            self.experience_buffer.add_batch(game_data)
            
            # Log game result
            winner = game_data[-1]['winner'] if game_data else 'D'
            self.logger.log_game_result(winner, game_num, len(game_data), temperature)
            
            # Training step
            if game_num % self.config.training.games_per_update == 0:
                self._training_update()
                
                # Update learning rate scheduler
                if self.scheduler:
                    self.scheduler.step()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    if game_num % 100 == 0:  # Log LR every 100 games
                        self.logger.logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Evaluation
            if game_num % self.config.evaluation.eval_interval == 0:
                self._evaluate()
            
            # Checkpoint
            if game_num % self.config.logging.checkpoint_interval == 0:
                self._save_checkpoint()
        
        # Final evaluation
        self._evaluate()
        self._save_checkpoint()
        
        # Log summary
        self.logger.log_summary()
        self.logger.close()
    
    def _calculate_temperature(self, game_num: int) -> float:
        """Calculate temperature for current game"""
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
    
    def _play_self_play_game(self, temperature: float) -> List[Dict[str, Any]]:
        """Play a self-play game and collect training data"""
        board = Board(self.config.game.board_size)
        game_data = []
        
        # Create players
        player_x = NeuralPlayer('X', self.model, self.config, temperature, device=self.device)
        player_o = NeuralPlayer('O', self.model, self.config, temperature, device=self.device)
        players = {'X': player_x, 'O': player_o}
        
        current_token = 'X'
        
        while True:
            current_player = players[current_token]
            
            # Get board state before move
            canonical = board.to_flat_canonical(current_token)
            state_tensor = torch.tensor(canonical, dtype=torch.float32)
            
            # Get move from player
            row, col = current_player.get_move(board)
            
            # Create policy target (one-hot)
            policy = torch.zeros(self.config.model.output_policy_size)
            move_idx = row * board.size + col
            policy[move_idx] = 1.0
            
            # Store experience
            game_data.append({
                'state': state_tensor,
                'policy': policy,
                'player': current_token,
                'value': None  # Will be filled after game ends
            })
            
            # Make move
            board.make_move(row, col, current_token)
            
            # Check game end
            result = board.get_game_result()
            if result:
                # Assign values based on game result
                for experience in game_data:
                    if result == 'D':
                        experience['value'] = torch.tensor([0.0], dtype=torch.float32)
                    elif result == experience['player']:
                        experience['value'] = torch.tensor([1.0], dtype=torch.float32)
                    else:
                        experience['value'] = torch.tensor([-1.0], dtype=torch.float32)
                    
                    # Add winner info for logging
                    experience['winner'] = result
                
                break
            
            # Switch player
            current_token = 'O' if current_token == 'X' else 'X'
        
        # Apply data augmentation if configured
        if hasattr(self.config.training, 'use_augmentation') and self.config.training.use_augmentation:
            game_data = self._augment_data(game_data, board)
        
        return game_data
    
    def _augment_data(self, game_data: List[Dict[str, Any]], board: Board) -> List[Dict[str, Any]]:
        """Apply data augmentation through board symmetries"""
        augmented_data = []
        
        for exp in game_data:
            # Reshape state to 2D
            state_2d = exp['state'].reshape(board.size, board.size).numpy()
            policy_2d = exp['policy'].reshape(board.size, board.size).numpy()
            
            # Get all symmetries
            for rot in range(4):
                # Rotation
                rot_state = np.rot90(state_2d, rot)
                rot_policy = np.rot90(policy_2d, rot)
                
                augmented_data.append({
                    'state': torch.tensor(rot_state.flatten(), dtype=torch.float32),
                    'policy': torch.tensor(rot_policy.flatten(), dtype=torch.float32),
                    'value': exp['value'],
                    'player': exp['player'],
                    'winner': exp['winner']
                })
                
                # Reflection
                flip_state = np.fliplr(rot_state)
                flip_policy = np.fliplr(rot_policy)
                
                augmented_data.append({
                    'state': torch.tensor(flip_state.flatten(), dtype=torch.float32),
                    'policy': torch.tensor(flip_policy.flatten(), dtype=torch.float32),
                    'value': exp['value'],
                    'player': exp['player'],
                    'winner': exp['winner']
                })
        
        return augmented_data
    
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
        
        # Log average losses
        if losses:
            avg_losses = {
                'total': np.mean([l['total'] for l in losses]),
                'policy': np.mean([l['policy'] for l in losses]),
                'value': np.mean([l['value'] for l in losses])
            }
            self.logger.log_training_step(avg_losses, self.game_count, 
                                         self.config.training.batch_size)
    
    def _train_step(self) -> Optional[Dict[str, float]]:
        """Single training step"""
        # Sample batch
        batch = self.experience_buffer.sample(self.config.training.batch_size)
        
        if not batch:
            return None
        
        # Prepare batch tensors
        states = torch.stack([exp['state'] for exp in batch]).to(self.device)
        target_policies = torch.stack([exp['policy'] for exp in batch]).to(self.device)
        target_values = torch.stack([exp['value'] for exp in batch]).to(self.device)
        
        # Forward pass
        self.model.train()
        pred_values, pred_policies = self.model(states)
        
        # Calculate losses
        policy_indices = target_policies.argmax(dim=1)
        policy_loss = F.cross_entropy(pred_policies, policy_indices)
        value_loss = F.mse_loss(pred_values, target_values)
        
        # Combined loss with weights
        total_loss = (self.config.training.policy_loss_weight * policy_loss + 
                     self.config.training.value_loss_weight * value_loss)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping if configured
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
        eval_results = {}
        
        for opponent_type in self.config.evaluation.opponents:
            results = self._evaluate_against(opponent_type)
            eval_results[opponent_type] = results
        
        # Log evaluation results
        self.logger.log_evaluation(eval_results, self.game_count)
        
        # Update best model if improved
        if 'random' in eval_results:
            score = eval_results['random']['win_rate']
            if score > self.best_eval_score:
                self.best_eval_score = score
                self._save_best_model()
    
    def _evaluate_against(self, opponent_type: str) -> Dict[str, float]:
        """Evaluate against a specific opponent"""
        results = {'wins': 0, 'losses': 0, 'draws': 0, 'total_moves': 0}
        
        # Create neural player (deterministic for evaluation)
        neural_player = NeuralPlayer('X', self.model, self.config, 
                                    temperature=self.config.evaluation.temperature_during_eval,
                                    device=self.device)
        
        # Create opponent
        if opponent_type == 'random':
            opponent = create_player('random', 'O')
        elif opponent_type == 'self':
            opponent = NeuralPlayer('O', self.model, self.config,
                                   temperature=self.config.evaluation.temperature_during_eval,
                                   device=self.device)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")
        
        # Play evaluation games
        game = Game()
        
        for _ in range(self.config.evaluation.eval_games):
            # Alternate who goes first
            if _ % 2 == 0:
                result = game.play(neural_player, opponent)
                if result['winner'] == 'X':
                    results['wins'] += 1
                elif result['winner'] == 'O':
                    results['losses'] += 1
                else:
                    results['draws'] += 1
            else:
                result = game.play(opponent, neural_player)
                if result['winner'] == 'O':
                    results['wins'] += 1
                elif result['winner'] == 'X':
                    results['losses'] += 1
                else:
                    results['draws'] += 1
            
            results['total_moves'] += result['moves']
        
        # Calculate statistics
        total_games = self.config.evaluation.eval_games
        return {
            'win_rate': results['wins'] / total_games,
            'loss_rate': results['losses'] / total_games,
            'draw_rate': results['draws'] / total_games,
            'avg_moves': results['total_moves'] / total_games
        }
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        self.logger.save_checkpoint(
            self.model,
            self.optimizer,
            self.game_count,
            self.config
        )
    
    def _save_best_model(self):
        """Save best model separately"""
        best_path = Path(self.config.logging.checkpoint_dir) / 'best_model.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'eval_score': self.best_eval_score,
            'game_num': self.game_count,
            'config': self.config
        }, best_path)
        
        self.logger.logger.info(f"New best model saved with score: {self.best_eval_score:.1%}")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.game_count = checkpoint['game_num']
        
        self.logger.logger.info(f"Resumed from checkpoint at game {self.game_count}")
