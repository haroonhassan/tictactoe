#!/usr/bin/env python3
"""
Multi-Teacher Training - 500 Games
Re-run to capture the sweet spot at game 200
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path

from config import Config
from models import create_model
from experiments.multi_teacher_training import MultiTeacherTrainer


def run_multi_teacher_500():
    """Run multi-teacher training for 500 games with careful checkpoint saving"""
    print("="*70)
    print("MULTI-TEACHER TRAINING - 500 GAMES")
    print("="*70)

    # Load the teacher models
    print("\nLoading teacher models...")

    # Defender: best_model (we'll make a backup first)
    defender_checkpoint = torch.load('checkpoints/best_model.pt', weights_only=False)
    defender_config = defender_checkpoint['config']
    defender_model = create_model(defender_config, model_type='standard')
    defender_model.load_state_dict(defender_checkpoint['model_state_dict'])
    print("✓ Loaded 4-layer defender")

    # Attacker: network_a
    attacker_checkpoint = torch.load('checkpoints/network_a_2layer.pt', weights_only=False)
    attacker_config = attacker_checkpoint['config']
    attacker_model = create_model(attacker_config, model_type='standard')
    attacker_model.load_state_dict(attacker_checkpoint['model_state_dict'])
    print("✓ Loaded 2-layer network 'a'")

    # Create new 8-layer student model
    config = Config()
    config.logging.wandb_name = "multi-teacher-500games"
    config.logging.wandb_tags = ["multi-teacher", "8-layers", "500-games", "sweet-spot"]
    config.logging.wandb_notes = """
    Re-running multi-teacher training for 500 games to capture sweet spot.
    Looking for game ~200 where we achieve 100% draws vs self + good vs random.
    """

    config.model.num_hidden_layers = 8
    config.model.hidden_size = 128

    config.training.num_games = 500  # Shorter run to capture sweet spot
    config.training.learning_rate = 0.001
    config.training.batch_size = 64
    config.training.temperature_decay = "cosine"

    config.evaluation.eval_games = 100
    config.evaluation.eval_interval = 50  # Evaluate more frequently

    # Save checkpoints to special directory
    config.logging.checkpoint_interval = 50  # Save every 50 games

    # Create dedicated checkpoint directory
    checkpoint_dir = Path('checkpoints/multi_teacher_500')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Checkpoints will be saved to: {checkpoint_dir}")

    print("\n✓ Created 8-layer student model")
    print("Training mix: 33% defender, 33% attacker, 33% random")
    print("Evaluating every 50 games")
    print("="*70)

    # Create student model and train
    student_model = create_model(config, model_type='standard')
    trainer = MultiTeacherTrainer(student_model, defender_model, attacker_model, config)

    # Manually save checkpoints to dedicated directory after each evaluation
    original_save = trainer._save_checkpoint
    def custom_save():
        # Save to main checkpoints as usual
        original_save()
        # Also save to dedicated directory with game number
        checkpoint_path = checkpoint_dir / f'checkpoint_game_{trainer.game_count}.pt'
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'game_count': trainer.game_count,
            'config': trainer.config
        }, checkpoint_path)
        print(f"✓ Saved to {checkpoint_path}")

    trainer._save_checkpoint = custom_save

    trainer.train()

    print("\n" + "="*70)
    print(f"Training complete! Checkpoints saved in: {checkpoint_dir}")
    print("="*70)


if __name__ == "__main__":
    run_multi_teacher_500()
