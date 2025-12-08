"""
RuleBasedPlayerのテスト

重要: get_action()がプレイヤーIDに応じた正しいハッシュ値を返すことを確認
"""

import unittest

from contrast_game import P1, P2, ContrastGame
from players.rule_based import RuleBasedPlayer


class TestRuleBasedPlayerInitialization(unittest.TestCase):
    """RuleBasedPlayerの初期化テスト"""

    def test_player_p1_initialization(self):
        """P1プレイヤーが正しく初期化されるか確認"""
        player = RuleBasedPlayer(P1)

        self.assertEqual(player.player_id, P1)
        self.assertEqual(player.opponent_id, P2)
        self.assertEqual(player.direction, -1)
        self.assertEqual(player.goal_row, 0)
        self.assertEqual(player.home_row, 4)

    def test_player_p2_initialization(self):
        """P2プレイヤーが正しく初期化されるか確認"""
        player = RuleBasedPlayer(P2)

        self.assertEqual(player.player_id, P2)
        self.assertEqual(player.opponent_id, P1)
        self.assertEqual(player.direction, 1)
        self.assertEqual(player.goal_row, 4)
        self.assertEqual(player.home_row, 0)


class TestRuleBasedPlayerGetAction(unittest.TestCase):
    """get_action()のテスト（重要：ハッシュ値が正しいか）"""

    def test_get_action_returns_valid_hash_p1(self):
        """P1でget_action()が有効なハッシュを返すか確認"""
        game = ContrastGame()
        player = RuleBasedPlayer(P1)

        # P1の番
        self.assertEqual(game.current_player, P1)

        action = player.get_action(game)

        # 有効なアクションが返される
        self.assertIsNotNone(action)
        # NumPy型も許容
        import numpy as np

        self.assertTrue(isinstance(action, (int, np.integer)))

        # 合法手リストに含まれている
        legal_actions = game.get_all_legal_actions()
        self.assertIn(action, legal_actions)

    def test_get_action_returns_valid_hash_p2(self):
        """P2でget_action()が有効なハッシュを返すか確認"""
        game = ContrastGame()

        # P2の番まで進める
        legal_actions = game.get_all_legal_actions()
        if legal_actions:
            game.step(legal_actions[0])

        player = RuleBasedPlayer(P2)

        # P2の番
        self.assertEqual(game.current_player, P2)

        action = player.get_action(game)

        # 有効なアクションが返される
        self.assertIsNotNone(action)
        # NumPy型も許容
        import numpy as np

        self.assertTrue(isinstance(action, (int, np.integer)))

        # 合法手リストに含まれている
        legal_actions = game.get_all_legal_actions()
        self.assertIn(action, legal_actions)

    def test_action_can_be_executed_p1(self):
        """P1のアクションがゲームで実行できるか確認"""
        game = ContrastGame()
        player = RuleBasedPlayer(P1)

        action = player.get_action(game)

        # アクションを実行
        initial_player = game.current_player
        done, winner = game.step(action)

        # 手番が変わる（またはゲーム終了）
        self.assertTrue(game.current_player != initial_player or done)

    def test_action_can_be_executed_p2(self):
        """P2のアクションがゲームで実行できるか確認"""
        game = ContrastGame()

        # P2の番まで進める
        legal_actions = game.get_all_legal_actions()
        if legal_actions:
            game.step(legal_actions[0])

        player = RuleBasedPlayer(P2)

        action = player.get_action(game)

        # アクションを実行
        initial_player = game.current_player
        done, winner = game.step(action)

        # 手番が変わる（またはゲーム終了）
        self.assertTrue(game.current_player != initial_player or done)

    def test_multiple_actions_p1(self):
        """P1で複数回アクションを取得して実行できるか確認"""
        game = ContrastGame()
        player = RuleBasedPlayer(P1)

        for _ in range(5):
            if game.game_over:
                break
            if game.current_player != P1:
                # P2の番はスキップ
                legal_actions = game.get_all_legal_actions()
                if legal_actions:
                    game.step(legal_actions[0])
                continue

            action = player.get_action(game)
            self.assertIsNotNone(action)

            legal_actions = game.get_all_legal_actions()
            self.assertIn(action, legal_actions)

            game.step(action)

    def test_multiple_actions_p2(self):
        """P2で複数回アクションを取得して実行できるか確認"""
        game = ContrastGame()
        player_p1 = RuleBasedPlayer(P1)
        player_p2 = RuleBasedPlayer(P2)

        for _ in range(10):
            if game.game_over:
                break

            if game.current_player == P1:
                action = player_p1.get_action(game)
            else:
                action = player_p2.get_action(game)

            self.assertIsNotNone(action)

            legal_actions = game.get_all_legal_actions()
            self.assertIn(action, legal_actions)

            game.step(action)


class TestRuleBasedPlayerStrategies(unittest.TestCase):
    """戦略メソッドのテスト"""

    def test_check_immediate_win_p1(self):
        """P1の即時勝利判定が正しく動作するか確認"""
        game = ContrastGame()
        player = RuleBasedPlayer(P1)

        # P1の駒を上段近くに配置
        game.pieces[1, 2] = P1
        game.pieces[4, :] = 0
        game.current_player = P1

        action = player.get_action(game)

        # アクションが返される
        self.assertIsNotNone(action)

    def test_check_immediate_win_p2(self):
        """P2の即時勝利判定が正しく動作するか確認"""
        game = ContrastGame()
        player = RuleBasedPlayer(P2)

        # P2の駒を下段近くに配置
        game.pieces[3, 2] = P2
        game.pieces[0, :] = 0
        game.current_player = P2

        action = player.get_action(game)

        # アクションが返される
        self.assertIsNotNone(action)

    def test_no_legal_actions(self):
        """合法手がない場合にNoneを返すか確認"""
        game = ContrastGame()
        player = RuleBasedPlayer(P1)

        # ゲーム終了状態に
        game.game_over = True

        action = player.get_action(game)

        # Noneが返される
        self.assertIsNone(action)


class TestRuleBasedPlayerVsRuleBasedPlayer(unittest.TestCase):
    """RuleBasedPlayer同士の対戦テスト"""

    def test_full_game_p1_vs_p2(self):
        """P1とP2のRuleBasedPlayer同士で完全なゲームができるか確認"""
        game = ContrastGame()
        player_p1 = RuleBasedPlayer(P1)
        player_p2 = RuleBasedPlayer(P2)

        max_moves = 200
        move_count = 0

        while not game.game_over and move_count < max_moves:
            if game.current_player == P1:
                action = player_p1.get_action(game)
            else:
                action = player_p2.get_action(game)

            if action is None:
                break

            legal_actions = game.get_all_legal_actions()
            self.assertIn(action, legal_actions)

            game.step(action)
            move_count += 1

        # ゲームが終了するか最大手数に達する
        self.assertTrue(game.game_over or move_count >= max_moves)


class TestRuleBasedPlayerConsistency(unittest.TestCase):
    """一貫性テスト：返されるハッシュ値が常に正しいか"""

    def test_action_hash_consistency_across_turns(self):
        """複数ターンに渡ってハッシュ値が一貫しているか確認"""
        game = ContrastGame()
        player_p1 = RuleBasedPlayer(P1)
        player_p2 = RuleBasedPlayer(P2)

        for turn in range(20):
            if game.game_over:
                break

            current_player_id = game.current_player

            if current_player_id == P1:
                player = player_p1
            else:
                player = player_p2

            action = player.get_action(game)
            self.assertIsNotNone(action)

            # プレイヤーIDが一致していることを確認
            self.assertEqual(player.player_id, current_player_id)

            # 合法手に含まれることを確認
            legal_actions = game.get_all_legal_actions()
            self.assertIn(
                action,
                legal_actions,
                f"Turn {turn}: Player {current_player_id} returned invalid action",
            )

            game.step(action)


if __name__ == "__main__":
    unittest.main(verbosity=2)
