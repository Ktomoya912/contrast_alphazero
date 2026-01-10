"""
Tests for play-time MCTS configuration and warmup scheduler
"""

import torch
import torch.optim as optim
import pytest

from config import mcts_config, play_mcts_config, training_config
from main import WarmupScheduler
from model import ContrastDualPolicyNet
from players.alpha_zero import AlphaZeroPlayer
from contrast_game import ContrastGame


class TestPlayMCTSConfig:
    """Test that play-time MCTS config is properly separated from training config"""

    def test_play_config_exists(self):
        """Test that play_mcts_config is available"""
        assert play_mcts_config is not None

    def test_play_config_has_higher_simulations(self):
        """Play config should have more simulations than training config"""
        assert play_mcts_config.NUM_SIMULATIONS > mcts_config.NUM_SIMULATIONS
        assert play_mcts_config.NUM_SIMULATIONS == 200
        assert mcts_config.NUM_SIMULATIONS == 50

    def test_play_config_has_different_c_puct(self):
        """Play config should have higher C_PUCT for more exploration"""
        assert play_mcts_config.C_PUCT > mcts_config.C_PUCT
        assert play_mcts_config.C_PUCT == 1.5
        assert mcts_config.C_PUCT == 1.0

    def test_play_config_has_no_noise(self):
        """Play config should have no Dirichlet noise (deterministic)"""
        assert play_mcts_config.DIRICHLET_EPSILON == 0.0
        assert mcts_config.DIRICHLET_EPSILON == 0.25

    def test_alphazero_player_uses_play_config(self, tmp_path):
        """AlphaZeroPlayer should use play config by default"""
        # Create dummy model
        model = ContrastDualPolicyNet()
        model_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)

        # Create player with defaults
        player = AlphaZeroPlayer(1, str(model_path))

        # Check that it uses play config
        assert player.num_simulations == play_mcts_config.NUM_SIMULATIONS
        assert player.c_puct == play_mcts_config.C_PUCT

    def test_alphazero_player_can_override_config(self, tmp_path):
        """AlphaZeroPlayer should allow overriding config"""
        # Create dummy model
        model = ContrastDualPolicyNet()
        model_path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)

        # Create player with custom config
        custom_sims = 100
        custom_c_puct = 2.0
        player = AlphaZeroPlayer(1, str(model_path), num_simulations=custom_sims, c_puct=custom_c_puct)

        # Check that it uses custom values
        assert player.num_simulations == custom_sims
        assert player.c_puct == custom_c_puct


class TestWarmupScheduler:
    """Test the learning rate warmup scheduler"""

    def test_warmup_scheduler_initialization(self):
        """Test that warmup scheduler can be initialized"""
        model = ContrastDualPolicyNet()
        optimizer = optim.Adam(model.parameters(), lr=0.2)
        scheduler = WarmupScheduler(optimizer, warmup_steps=1000, base_lr=0.2)

        assert scheduler.warmup_steps == 1000
        assert scheduler.base_lr == 0.2
        assert scheduler.current_step == 0

    def test_warmup_scheduler_linear_increase(self):
        """Test that warmup scheduler increases LR linearly"""
        model = ContrastDualPolicyNet()
        optimizer = optim.Adam(model.parameters(), lr=0.2)
        scheduler = WarmupScheduler(optimizer, warmup_steps=100, base_lr=0.2)

        # Initial LR should be very low
        initial_lr = scheduler.get_lr()
        assert initial_lr == 0.2  # Not warmed up yet

        # After 50 steps, should be at 50% of base LR
        for _ in range(50):
            scheduler.step()
        mid_lr = scheduler.get_lr()
        assert abs(mid_lr - 0.1) < 1e-6  # Should be 0.2 * 50/100 = 0.1

        # After 100 steps, should be at 100% of base LR
        for _ in range(50):
            scheduler.step()
        final_lr = scheduler.get_lr()
        assert abs(final_lr - 0.2) < 1e-6  # Should be 0.2

    def test_warmup_scheduler_with_low_lr(self):
        """Test warmup with typical low learning rate"""
        model = ContrastDualPolicyNet()
        base_lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
        scheduler = WarmupScheduler(optimizer, warmup_steps=10, base_lr=base_lr)

        # After 5 steps, should be at 50%
        for _ in range(5):
            scheduler.step()
        assert abs(scheduler.get_lr() - base_lr * 0.5) < 1e-9

    def test_warmup_config_flags(self):
        """Test that warmup config flags are present"""
        assert hasattr(training_config, 'USE_WARMUP')
        assert hasattr(training_config, 'WARMUP_STEPS')
        assert hasattr(training_config, 'MAX_GRAD_NORM')

        # Check default values
        assert training_config.USE_WARMUP == False  # Default off for backward compatibility
        assert training_config.WARMUP_STEPS == 1000
        assert training_config.MAX_GRAD_NORM == 10.0


class TestNetworkArchitectureChanges:
    """Test that network architecture has been properly adjusted"""

    def test_reduced_residual_blocks(self):
        """Test that residual blocks have been reduced"""
        from config import network_config

        # Should be 6 blocks now (reduced from 8)
        assert network_config.NUM_RES_BLOCKS == 6

    def test_model_uses_reduced_blocks(self):
        """Test that model actually uses the reduced number of blocks"""
        model = ContrastDualPolicyNet()

        # Check that model has correct number of residual blocks
        assert len(model.res_blocks) == 6

    def test_model_parameter_count_reasonable(self):
        """Test that model has reasonable parameter count for 5x5 game"""
        model = ContrastDualPolicyNet()
        total_params = sum(p.numel() for p in model.parameters())

        # Should have around 1M parameters (less than the 8-block version)
        # This is reasonable for a 5x5 board game
        assert total_params < 1_500_000  # Less than 1.5M
        assert total_params > 500_000    # More than 0.5M

    def test_forward_pass_works(self):
        """Test that forward pass works with new architecture"""
        model = ContrastDualPolicyNet()
        dummy_input = torch.randn(4, 66, 5, 5)

        m_out, t_out, v_out = model(dummy_input)

        assert m_out.shape == (4, 625)
        assert t_out.shape == (4, 51)
        assert v_out.shape == (4, 1)
