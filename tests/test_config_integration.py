#!/usr/bin/env python3
"""config.pyの統合をテストするスクリプト"""

from config import (
    evaluation_config,
    game_config,
    mcts_config,
    network_config,
    path_config,
    training_config,
)
from contrast_game import (
    BOARD_SIZE,
    HISTORY_SIZE,
    INITIAL_BLACK_TILES,
    INITIAL_GRAY_TILES,
    NUM_TILES,
)
from logger import get_logger, setup_logger

setup_logger()
logger = get_logger(__name__)


def test_config_values():
    """config.pyの値が正しく使用されているかテスト"""

    logger.info("Testing config.py integration...")

    # GameConfig
    assert BOARD_SIZE == game_config.BOARD_SIZE, "BOARD_SIZE should come from config"
    assert HISTORY_SIZE == game_config.HISTORY_SIZE, (
        "HISTORY_SIZE should come from config"
    )
    assert INITIAL_BLACK_TILES == game_config.INITIAL_BLACK_TILES, (
        "INITIAL_BLACK_TILES should come from config"
    )
    assert INITIAL_GRAY_TILES == game_config.INITIAL_GRAY_TILES, (
        "INITIAL_GRAY_TILES should come from config"
    )
    assert NUM_TILES == game_config.NUM_TILE_ACTIONS, (
        "NUM_TILES should come from config"
    )

    logger.info(
        f"✓ GameConfig values: BOARD_SIZE={BOARD_SIZE}, HISTORY_SIZE={HISTORY_SIZE}"
    )

    # MCTSConfig
    logger.info(
        f"✓ MCTSConfig: NUM_SIMULATIONS={mcts_config.NUM_SIMULATIONS}, C_PUCT={mcts_config.C_PUCT}"
    )

    # NetworkConfig
    logger.info(
        f"✓ NetworkConfig: NUM_RES_BLOCKS={network_config.NUM_RES_BLOCKS}, NUM_FILTERS={network_config.NUM_FILTERS}"
    )

    # TrainingConfig
    logger.info(
        f"✓ TrainingConfig: BATCH_SIZE={training_config.BATCH_SIZE}, LEARNING_RATE={training_config.LEARNING_RATE}"
    )

    # EvaluationConfig
    logger.info(
        f"✓ EvaluationConfig: EVAL_INTERVAL={evaluation_config.EVAL_INTERVAL}, BASELINE_ELO={evaluation_config.BASELINE_ELO}"
    )

    # PathConfig
    logger.info(
        f"✓ PathConfig: LOGS_DIR={path_config.LOGS_DIR}, MODELS_DIR={path_config.MODELS_DIR}"
    )

    logger.info("All config values are correctly integrated! ✓")


def test_model_uses_config():
    """モデルがconfig.pyを使用していることをテスト"""
    import torch

    from model import ContrastDualPolicyNet

    logger.info("Testing model integration with config.py...")

    # デフォルト引数でモデルを作成
    model = ContrastDualPolicyNet()

    # 入力テンソルを作成
    dummy_input = torch.randn(
        1, game_config.INPUT_CHANNELS, game_config.BOARD_SIZE, game_config.BOARD_SIZE
    )

    # 推論を実行
    with torch.no_grad():
        move_out, tile_out, value_out = model(dummy_input)

    # 出力形状を確認
    expected_move_size = game_config.BOARD_SIZE**2 * game_config.BOARD_SIZE**2
    expected_tile_size = game_config.NUM_TILE_ACTIONS

    assert move_out.shape == (1, expected_move_size), (
        f"Move output shape mismatch: {move_out.shape}"
    )
    assert tile_out.shape == (1, expected_tile_size), (
        f"Tile output shape mismatch: {tile_out.shape}"
    )
    assert value_out.shape == (1, 1), f"Value output shape mismatch: {value_out.shape}"

    logger.info(
        f"✓ Model outputs: Move={move_out.shape}, Tile={tile_out.shape}, Value={value_out.shape}"
    )
    logger.info("Model correctly uses config.py values! ✓")


def test_mcts_uses_config():
    """MCTSがconfig.pyを使用していることをテスト"""
    import torch

    from mcts import MCTS
    from model import ContrastDualPolicyNet

    logger.info("Testing MCTS integration with config.py...")

    device = torch.device("cpu")
    model = ContrastDualPolicyNet().to(device)
    model.eval()

    # デフォルト引数でMCTSを作成（config.pyの値が使われるはず）
    mcts = MCTS(network=model, device=device)

    # 設定値が正しいか確認
    assert mcts.alpha == mcts_config.DIRICHLET_ALPHA, (
        "MCTS should use DIRICHLET_ALPHA from config"
    )
    assert mcts.c_puct == mcts_config.C_PUCT, "MCTS should use C_PUCT from config"
    assert mcts.eps == mcts_config.DIRICHLET_EPSILON, (
        "MCTS should use DIRICHLET_EPSILON from config"
    )

    logger.info(
        f"✓ MCTS parameters: alpha={mcts.alpha}, c_puct={mcts.c_puct}, epsilon={mcts.eps}"
    )
    logger.info("MCTS correctly uses config.py values! ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("Config.py Integration Test")
    print("=" * 60)

    try:
        test_config_values()
        print()
        test_model_uses_config()
        print()
        test_mcts_uses_config()
        print()
        print("=" * 60)
        print("✓ All integration tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        exit(1)
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
