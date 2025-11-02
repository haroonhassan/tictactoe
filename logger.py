"""
Advanced logging and metrics tracking for neural network training.
Provides console logging, file logging, tensorboard integration, and metrics tracking.
"""

import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
import numpy as np


class MetricsTracker:
    """Track and aggregate training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.window_size = window_size
        self.step_count = defaultdict(int)
        
    def add(self, metric_name: str, value: float, step: Optional[int] = None):
        """Add a metric value"""
        self.metrics[metric_name].append(value)
        if step is not None:
            self.step_count[metric_name] = step
            
    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """Get average of a metric over last_n values"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0
        
        values = list(self.metrics[metric_name])
        if last_n:
            values = values[-last_n:]
        return np.mean(values)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return {}
        
        values = list(self.metrics[metric_name])
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'last': values[-1],
            'count': len(values)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get stats for all tracked metrics"""
        return {name: self.get_stats(name) for name in self.metrics.keys()}
    
    def reset(self, metric_name: Optional[str] = None):
        """Reset metrics"""
        if metric_name:
            self.metrics[metric_name].clear()
        else:
            self.metrics.clear()


class Logger:
    """Enhanced logger with file, console, and tensorboard support"""
    
    def __init__(self, config):
        self.config = config
        self.metrics_tracker = MetricsTracker()
        self.start_time = time.time()
        
        # Setup logging directory
        self.log_dir = Path("logs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file and console logging
        self._setup_logging()
        
        # Setup tensorboard if requested
        self.writer = None
        if config.logging.use_tensorboard:
            self._setup_tensorboard()
            
        # Game statistics
        self.game_results = deque(maxlen=1000)
        self.evaluation_history = []
        
    def _setup_logging(self):
        """Configure Python logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level),
            format=log_format,
            handlers=[]
        )
        
        self.logger = logging.getLogger('TicTacToeAI')
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.logging.log_to_file:
            file_path = self.log_dir / self.config.logging.log_file
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(file_handler)
            
    def _setup_tensorboard(self):
        """Setup tensorboard writer"""
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = self.log_dir / self.config.logging.tensorboard_dir
            self.writer = SummaryWriter(tb_dir)
            self.logger.info(f"Tensorboard logging enabled. Run: tensorboard --logdir {tb_dir}")
        except ImportError:
            self.logger.warning("Tensorboard not available. Install with: pip install tensorboard")
            self.writer = None
            
    def log_game_result(self, winner: str, game_num: int, moves: int, temperature: float):
        """Log the result of a game"""
        result = {
            'game_num': game_num,
            'winner': winner,
            'moves': moves,
            'temperature': temperature,
            'timestamp': time.time()
        }
        self.game_results.append(result)
        
        # Calculate win rates
        if len(self.game_results) >= 10:
            recent_games = list(self.game_results)[-100:]
            x_wins = sum(1 for g in recent_games if g['winner'] == 'X')
            o_wins = sum(1 for g in recent_games if g['winner'] == 'O')
            draws = sum(1 for g in recent_games if g['winner'] == 'D')
            
            self.metrics_tracker.add('win_rate_x', x_wins / len(recent_games))
            self.metrics_tracker.add('win_rate_o', o_wins / len(recent_games))
            self.metrics_tracker.add('draw_rate', draws / len(recent_games))
            
            if game_num % 10 == 0:
                self.logger.info(
                    f"Last 100 games - X: {x_wins}%, O: {o_wins}%, Draw: {draws}%"
                )
                
    def log_training_step(self, losses: Dict[str, float], game_num: int, batch_size: int):
        """Log training step metrics"""
        # Track metrics
        for loss_name, loss_value in losses.items():
            self.metrics_tracker.add(f'loss_{loss_name}', loss_value, game_num)
            
        # Log to tensorboard
        if self.writer:
            for loss_name, loss_value in losses.items():
                self.writer.add_scalar(f'Loss/{loss_name}', loss_value, game_num)
                
        # Console logging
        if self.config.logging.verbose and game_num % self.config.logging.log_metrics_interval == 0:
            avg_total = self.metrics_tracker.get_average('loss_total', 10)
            avg_policy = self.metrics_tracker.get_average('loss_policy', 10)
            avg_value = self.metrics_tracker.get_average('loss_value', 10)
            
            self.logger.info(
                f"Game {game_num} - Loss (avg last 10): "
                f"Total={avg_total:.4f}, Policy={avg_policy:.4f}, Value={avg_value:.4f}"
            )
            
    def log_evaluation(self, eval_results: Dict[str, Any], game_num: int):
        """Log evaluation results"""
        self.evaluation_history.append({
            'game_num': game_num,
            'results': eval_results,
            'timestamp': time.time()
        })
        
        # Log to console
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Evaluation at game {game_num}:")
        for opponent, results in eval_results.items():
            win_rate = results.get('win_rate', 0)
            draw_rate = results.get('draw_rate', 0)
            avg_moves = results.get('avg_moves', 0)
            self.logger.info(
                f"  vs {opponent}: Win={win_rate:.1%}, Draw={draw_rate:.1%}, "
                f"Avg moves={avg_moves:.1f}"
            )
        self.logger.info(f"{'='*50}\n")
        
        # Log to tensorboard
        if self.writer:
            for opponent, results in eval_results.items():
                self.writer.add_scalar(
                    f'Eval/{opponent}/win_rate', 
                    results.get('win_rate', 0), 
                    game_num
                )
                self.writer.add_scalar(
                    f'Eval/{opponent}/draw_rate', 
                    results.get('draw_rate', 0), 
                    game_num
                )
                
    def log_model_parameters(self, model, game_num: int):
        """Log model parameter statistics"""
        if not self.writer:
            return
            
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, game_num)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, game_num)
                
    def save_checkpoint(self, model, optimizer, game_num: int, config):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.logging.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'game_num': game_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'metrics': self.metrics_tracker.get_all_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_game_{game_num}.pt'
        import torch
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints(checkpoint_dir)
        
    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Keep only the last N checkpoints"""
        checkpoints = sorted(checkpoint_dir.glob('checkpoint_*.pt'))
        if len(checkpoints) > self.config.logging.keep_last_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.logging.keep_last_n_checkpoints]:
                checkpoint.unlink()
                self.logger.debug(f"Removed old checkpoint: {checkpoint}")
                
    def log_summary(self):
        """Log training summary"""
        elapsed_time = time.time() - self.start_time
        
        self.logger.info("\n" + "="*60)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total time: {elapsed_time/60:.1f} minutes")
        
        # Game statistics
        if self.game_results:
            total_games = len(self.game_results)
            x_wins = sum(1 for g in self.game_results if g['winner'] == 'X')
            o_wins = sum(1 for g in self.game_results if g['winner'] == 'O')
            draws = sum(1 for g in self.game_results if g['winner'] == 'D')
            
            self.logger.info(f"\nGame Statistics ({total_games} games):")
            self.logger.info(f"  X wins: {x_wins} ({x_wins/total_games:.1%})")
            self.logger.info(f"  O wins: {o_wins} ({o_wins/total_games:.1%})")
            self.logger.info(f"  Draws: {draws} ({draws/total_games:.1%})")
            
        # Loss statistics
        loss_stats = self.metrics_tracker.get_all_stats()
        if loss_stats:
            self.logger.info("\nLoss Statistics:")
            for metric_name, stats in loss_stats.items():
                if 'loss' in metric_name:
                    self.logger.info(
                        f"  {metric_name}: mean={stats['mean']:.4f}, "
                        f"std={stats['std']:.4f}, final={stats['last']:.4f}"
                    )
                    
        # Evaluation history
        if self.evaluation_history:
            self.logger.info("\nEvaluation History:")
            for eval_point in self.evaluation_history[-3:]:  # Last 3 evaluations
                game_num = eval_point['game_num']
                self.logger.info(f"  Game {game_num}:")
                for opponent, results in eval_point['results'].items():
                    win_rate = results.get('win_rate', 0)
                    self.logger.info(f"    vs {opponent}: {win_rate:.1%} win rate")
                    
        self.logger.info("="*60 + "\n")
        
        # Save summary to file
        summary_path = self.log_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'elapsed_time_seconds': elapsed_time,
                'game_results': list(self.game_results),
                'evaluation_history': self.evaluation_history,
                'final_metrics': loss_stats
            }, f, indent=2)
            
        self.logger.info(f"Summary saved to: {summary_path}")
        
    def close(self):
        """Clean up resources"""
        if self.writer:
            self.writer.close()
