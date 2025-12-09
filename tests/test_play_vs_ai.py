#!/usr/bin/env python
"""
play_vs_ai.pyの簡易テスト

コンポーネントが正しく動作することを確認
"""

import sys

import numpy as np

from contrast_game import P1, P2
from logger import setup_logger
from play_vs_ai import HumanVsAI


def test_initialization():
    """初期化のテスト"""
    print("初期化テスト...")

    # 未学習モデルでも動作することを確認
    game = HumanVsAI(
        model_path="nonexistent.pth", num_simulations=5, player1_type="human"
    )

    assert game.player1_type == "human"
    assert game.player2_type == "ai"
    assert game.num_simulations == 5
    assert P1 in game.players
    assert P2 in game.players

    print("✓ 初期化成功")


def test_ai_action():
    """AI行動のテスト"""
    print("\nAI行動テスト...")

    game = HumanVsAI(
        model_path="nonexistent.pth",
        num_simulations=5,
        player1_type="ai",
        player2_type="ai",
    )

    # 初期状態でAIの行動を取得（プレイヤー1の番）
    action = game.get_action_for_player(P1)

    assert action is not None
    assert isinstance(action, (int, np.integer))
    assert action >= 0

    print("✓ AI行動取得成功")


def test_multiple_ai_moves():
    """複数回のAI行動テスト"""
    print("\n複数回AI行動テスト...")

    game = HumanVsAI(
        model_path="nonexistent.pth",
        num_simulations=3,
        player1_type="ai",
        player2_type="ai",
    )

    # 数手進める
    moves_made = 0
    max_moves = 10

    while not game.game.game_over and moves_made < max_moves:
        # AIの行動を取得して実行
        action = game.get_action_for_player(game.game.current_player)
        if action is None:
            break

        done, winner = game.game.step(action)
        moves_made += 1

        if done:
            break

    print(f"✓ {moves_made}手実行成功")
    assert moves_made > 0


def test_different_player_types():
    """異なるプレイヤータイプのテスト"""
    print("\n異なるプレイヤータイプテスト...")

    # random vs rule
    game = HumanVsAI(
        model_path="nonexistent.pth",
        num_simulations=3,
        player1_type="random",
        player2_type="rule",
    )

    moves_made = 0
    max_moves = 10

    while not game.game.game_over and moves_made < max_moves:
        action = game.get_action_for_player(game.game.current_player)
        if action is None:
            break

        done, winner = game.game.step(action)
        moves_made += 1

        if done:
            break

    print(f"✓ {moves_made}手実行成功 (random vs rule)")
    assert moves_made > 0


if __name__ == "__main__":
    # ロギング設定
    setup_logger()

    print("=" * 50)
    print("play_vs_ai.py 動作テスト")
    print("=" * 50)

    try:
        test_initialization()
        test_ai_action()
        test_multiple_ai_moves()
        test_different_player_types()

        print("\n" + "=" * 50)
        print("✅ 全てのテストが成功しました")
        print("=" * 50)

    except AssertionError as e:
        print(f"\n❌ テスト失敗: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
