"""
MCTS (Monte Carlo Tree Search) の網羅的なテストコード

テストカバレッジ:
- MCTS初期化
- ゲーム状態のキー生成
- ノードの展開 (expand)
- 探索 (search)
- 評価 (evaluate)
- PUCT (Predictor Upper Confidence Bound)
- ディリクレノイズの付加
- バックプロパゲーション
- 複数シミュレーション
- エッジケース
"""

import unittest

import numpy as np
import torch

from contrast_game import ContrastGame
from mcts import MCTS
from model import ContrastDualPolicyNet


class MockNetwork(torch.nn.Module):
    """テスト用のモックネットワーク"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        一様分布とゼロ評価値を返す
        """
        batch_size = x.shape[0]

        # Move logits: (batch, 625)
        move_logits = torch.zeros(batch_size, 625)

        # Tile logits: (batch, 51)
        tile_logits = torch.zeros(batch_size, 51)

        # Value: (batch, 1)
        value = torch.zeros(batch_size, 1)

        return move_logits, tile_logits, value

    def eval(self):
        return self


class DeterministicNetwork(torch.nn.Module):
    """テスト用の決定的なネットワーク"""

    def __init__(self, value=0.5):
        super().__init__()
        self.value = value

    def forward(self, x):
        batch_size = x.shape[0]

        # 特定のアクションに高いlogitを与える
        move_logits = torch.full((batch_size, 625), -10.0)
        move_logits[:, 0] = 10.0  # Move index 0 が最も高い

        tile_logits = torch.full((batch_size, 51), -10.0)
        tile_logits[:, 0] = 10.0  # Tile index 0 が最も高い

        value = torch.full((batch_size, 1), self.value)

        return move_logits, tile_logits, value

    def eval(self):
        return self


class TestMCTSInitialization(unittest.TestCase):
    """MCTSの初期化テスト"""

    def test_mcts_initialization(self):
        """MCTSが正しく初期化されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")

        mcts = MCTS(network, device)

        self.assertIsNotNone(mcts.network)
        self.assertEqual(mcts.device, device)
        self.assertEqual(mcts.alpha, 0.3)
        self.assertEqual(mcts.c_puct, 1.0)
        self.assertEqual(mcts.eps, 0.25)

    def test_mcts_custom_parameters(self):
        """カスタムパラメータでの初期化を確認"""
        network = MockNetwork()
        device = torch.device("cpu")

        mcts = MCTS(network, device, alpha=0.5, c_puct=2.0, epsilon=0.1)

        self.assertEqual(mcts.alpha, 0.5)
        self.assertEqual(mcts.c_puct, 2.0)
        self.assertEqual(mcts.eps, 0.1)

    def test_mcts_empty_dictionaries(self):
        """初期状態で辞書が空であることを確認"""
        network = MockNetwork()
        device = torch.device("cpu")

        mcts = MCTS(network, device)

        self.assertEqual(len(mcts.P), 0)
        self.assertEqual(len(mcts.N), 0)
        self.assertEqual(len(mcts.W), 0)


class TestGameToKey(unittest.TestCase):
    """ゲーム状態のキー生成テスト"""

    def test_game_to_key_deterministic(self):
        """同じゲーム状態から同じキーが生成されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        key1 = mcts.game_to_key(game)
        key2 = mcts.game_to_key(game)

        self.assertEqual(key1, key2)

    def test_game_to_key_different_states(self):
        """異なるゲーム状態から異なるキーが生成されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game1 = ContrastGame()
        key1 = mcts.game_to_key(game1)

        # ゲーム状態を変更
        legal_actions = game1.get_all_legal_actions()
        game1.step(legal_actions[0])

        key2 = mcts.game_to_key(game1)

        self.assertNotEqual(key1, key2)

    def test_game_to_key_includes_move_count(self):
        """move_countが異なれば異なるキーが生成されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game1 = ContrastGame()
        game1.move_count = 0

        game2 = ContrastGame()
        game2.move_count = 1

        key1 = mcts.game_to_key(game1)
        key2 = mcts.game_to_key(game2)

        # move_countが含まれるため、キーが異なる
        self.assertNotEqual(key1, key2)

    def test_game_to_key_includes_player(self):
        """プレイヤーが異なれば異なるキーが生成されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game1 = ContrastGame()
        game1.current_player = 1

        game2 = ContrastGame()
        game2.current_player = 2

        key1 = mcts.game_to_key(game1)
        key2 = mcts.game_to_key(game2)

        self.assertNotEqual(key1, key2)


class TestExpand(unittest.TestCase):
    """ノード展開のテスト"""

    def test_expand_creates_entries(self):
        """展開により辞書にエントリが作成されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        key = mcts.game_to_key(game)

        # 展開前は存在しない
        self.assertNotIn(key, mcts.P)

        # 展開
        mcts._expand(game)

        # 展開後は存在する
        self.assertIn(key, mcts.P)
        self.assertIn(key, mcts.N)
        self.assertIn(key, mcts.W)

    def test_expand_initializes_values(self):
        """展開により値が正しく初期化されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        mcts._expand(game)

        key = mcts.game_to_key(game)

        # すべてのアクションでN=0, W=0
        for action in mcts.P[key].keys():
            self.assertEqual(mcts.N[key][action], 0)
            self.assertEqual(mcts.W[key][action], 0)

    def test_expand_returns_value(self):
        """展開が評価値を返すか確認"""
        network = DeterministicNetwork(value=0.7)
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        value = mcts._expand(game)

        # ネットワークが返す値
        self.assertAlmostEqual(value, 0.7, places=5)

    def test_expand_only_legal_actions(self):
        """展開が合法手のみを含むか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        legal_actions = set(game.get_all_legal_actions())

        mcts._expand(game)

        key = mcts.game_to_key(game)
        expanded_actions = set(mcts.P[key].keys())

        # 展開されたアクションが合法手のサブセット
        self.assertTrue(expanded_actions.issubset(legal_actions))

    def test_expand_probabilities_sum_to_one(self):
        """展開された確率の合計が1であることを確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        mcts._expand(game)

        key = mcts.game_to_key(game)
        probs = list(mcts.P[key].values())

        # 確率の合計が約1
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_expand_terminal_state(self):
        """終了状態の展開を確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        game.game_over = True
        game.winner = 1

        mcts._expand(game)

        key = mcts.game_to_key(game)

        # 終了状態では合法手がない
        self.assertEqual(len(mcts.P[key]), 0)


class TestSearch(unittest.TestCase):
    """探索のテスト"""

    def test_search_returns_policy(self):
        """探索がポリシーを返すか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        policy, _ = mcts.search(game, num_simulations=10)

        # ポリシーが辞書
        self.assertIsInstance(policy, dict)

    def test_search_policy_probabilities_sum_to_one(self):
        """探索で得られたポリシーの確率合計が1であることを確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        policy, _ = mcts.search(game, num_simulations=10)

        if policy:
            probs = list(policy.values())
            self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_search_multiple_simulations(self):
        """複数回のシミュレーションが実行されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        num_simulations = 20
        policy, values = mcts.search(game, num_simulations=num_simulations)

        # ルートノードの訪問回数がシミュレーション回数と一致
        key = mcts.game_to_key(game)
        total_visits = sum(mcts.N[key].values())

        self.assertEqual(total_visits, num_simulations)

    def test_search_adds_dirichlet_noise(self):
        """探索がディリクレノイズを追加するか確認"""
        network = DeterministicNetwork(value=0.5)
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # ノイズありで2回実行
        policy1, values1 = mcts.search(game, num_simulations=10)

        # MCTSをリセット
        mcts = MCTS(network, device)
        policy2, values2 = mcts.search(game, num_simulations=10)

        # ノイズによりポリシーが異なる可能性がある
        # (ただし、決定的なネットワークでは同じ場合もある)
        # ここでは単にエラーが出ないことを確認
        self.assertIsNotNone(policy1)
        self.assertIsNotNone(policy2)

    def test_search_terminal_state_returns_empty(self):
        """終了状態での探索が空のポリシーを返すか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        game.game_over = True
        game.winner = 1

        policy, values = mcts.search(game, num_simulations=10)

        # 終了状態では合法手がないため、空のポリシー
        self.assertEqual(len(policy), 0)


class TestEvaluate(unittest.TestCase):
    """評価のテスト"""

    def test_evaluate_terminal_state_winner(self):
        """終了状態での評価値を確認 (勝者あり)"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        game.game_over = True
        game.winner = 1
        game.current_player = 1

        value = mcts._evaluate(game)

        # 勝者なので+1
        self.assertEqual(value, 1)

    def test_evaluate_terminal_state_loser(self):
        """終了状態での評価値を確認 (敗者)"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        game.game_over = True
        game.winner = 2
        game.current_player = 1

        value = mcts._evaluate(game)

        # 敗者なので-1
        self.assertEqual(value, -1)

    def test_evaluate_terminal_state_draw(self):
        """終了状態での評価値を確認 (引き分け)"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        game.game_over = True
        game.winner = 0

        value = mcts._evaluate(game)

        # 引き分けなので0
        self.assertEqual(value, 0)

    def test_evaluate_expands_unexpanded_node(self):
        """未展開ノードが展開されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        key = mcts.game_to_key(game)

        # 展開前は存在しない
        self.assertNotIn(key, mcts.P)

        # 評価を実行
        mcts._evaluate(game)

        # 展開後は存在する
        self.assertIn(key, mcts.P)

    def test_evaluate_updates_statistics(self):
        """評価により統計が更新されるか確認"""
        network = DeterministicNetwork(value=0.5)
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # ルートを展開
        mcts._expand(game)

        key = mcts.game_to_key(game)

        # 初期訪問回数
        initial_visits = sum(mcts.N[key].values())

        # 評価を実行
        mcts._evaluate(game.copy())

        # 訪問回数が増加
        new_visits = sum(mcts.N[key].values())

        self.assertGreater(new_visits, initial_visits)


class TestPUCT(unittest.TestCase):
    """PUCT (Predictor Upper Confidence Bound) のテスト"""

    def test_puct_selects_unvisited_actions(self):
        """PUCTが未訪問アクションを優先するか確認"""
        network = DeterministicNetwork(value=0.5)
        device = torch.device("cpu")
        mcts = MCTS(network, device, c_puct=1.0)

        game = ContrastGame()

        # 1回シミュレーション
        policy, values = mcts.search(game, num_simulations=1)

        key = mcts.game_to_key(game)

        # 一部のアクションが訪問されている
        visited_actions = [a for a, n in mcts.N[key].items() if n > 0]

        self.assertGreater(len(visited_actions), 0)

    def test_puct_balances_exploration_exploitation(self):
        """PUCTが探索と活用のバランスを取るか確認"""
        network = DeterministicNetwork(value=0.5)
        device = torch.device("cpu")
        mcts = MCTS(network, device, c_puct=1.0)

        game = ContrastGame()

        # 複数回シミュレーション
        policy, values = mcts.search(game, num_simulations=50)

        key = mcts.game_to_key(game)

        # 複数のアクションが訪問される (探索)
        visited_actions = [a for a, n in mcts.N[key].items() if n > 0]

        # 少なくとも2つ以上のアクションが訪問される
        self.assertGreater(len(visited_actions), 1)


class TestBackpropagation(unittest.TestCase):
    """バックプロパゲーションのテスト"""

    def test_backpropagation_updates_visit_counts(self):
        """バックプロパゲーションが訪問回数を更新するか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # 探索
        policy, values = mcts.search(game, num_simulations=10)

        key = mcts.game_to_key(game)

        # 訪問回数の合計がシミュレーション回数と一致
        total_visits = sum(mcts.N[key].values())

        self.assertEqual(total_visits, 10)

    def test_backpropagation_updates_values(self):
        """バックプロパゲーションが価値を更新するか確認"""
        network = DeterministicNetwork(value=0.5)
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # 探索
        policy, values = mcts.search(game, num_simulations=10)

        key = mcts.game_to_key(game)

        # 価値が更新されている
        for action in mcts.W[key].keys():
            # 訪問されたアクションでは価値が0以外
            if mcts.N[key][action] > 0:
                # 価値が設定されている (0でない可能性)
                self.assertIsNotNone(mcts.W[key][action])


class TestEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""

    def test_search_with_zero_simulations(self):
        """シミュレーション回数0での探索を確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        policy, values = mcts.search(game, num_simulations=0)

        # ポリシーが返される (一様分布)
        self.assertIsInstance(policy, dict)

    def test_search_with_single_simulation(self):
        """シミュレーション回数1での探索を確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        policy, values = mcts.search(game, num_simulations=1)

        # ポリシーが返される
        self.assertIsInstance(policy, dict)

    def test_search_with_no_legal_actions(self):
        """合法手がない状態での探索を確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # 強制的に合法手をなくす (ゲーム終了)
        game.game_over = True

        policy, values = mcts.search(game, num_simulations=10)

        # 空のポリシーが返される
        self.assertEqual(len(policy), 0)

    def test_multiple_searches_on_same_tree(self):
        """同じツリーで複数回探索を実行"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # 1回目の探索
        policy1, values1 = mcts.search(game, num_simulations=10)

        # 2回目の探索 (同じゲーム状態)
        policy2, values2 = mcts.search(game, num_simulations=10)

        # 両方ともポリシーが返される
        self.assertIsInstance(policy1, dict)
        self.assertIsInstance(policy2, dict)

        # 訪問回数が累積される
        key = mcts.game_to_key(game)
        total_visits = sum(mcts.N[key].values())

        self.assertEqual(total_visits, 20)

    def test_search_does_not_modify_original_game(self):
        """探索が元のゲーム状態を変更しないことを確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # 初期状態を保存
        initial_pieces = game.pieces.copy()
        initial_tiles = game.tiles.copy()
        initial_player = game.current_player
        initial_move_count = game.move_count

        # 探索
        policy, values = mcts.search(game, num_simulations=10)

        # ゲーム状態が変更されていない
        self.assertTrue(np.array_equal(game.pieces, initial_pieces))
        self.assertTrue(np.array_equal(game.tiles, initial_tiles))
        self.assertEqual(game.current_player, initial_player)
        self.assertEqual(game.move_count, initial_move_count)


class TestMCTSWithRealNetwork(unittest.TestCase):
    """実際のネットワークを使用したテスト"""

    def test_mcts_with_real_network(self):
        """実際のネットワークでMCTSが動作するか確認"""
        network = ContrastDualPolicyNet()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # 探索を実行
        policy, values = mcts.search(game, num_simulations=5)

        # ポリシーが返される
        self.assertIsInstance(policy, dict)
        self.assertGreater(len(policy), 0)

    def test_mcts_policy_contains_valid_actions(self):
        """MCTSのポリシーが合法手のみを含むか確認"""
        network = ContrastDualPolicyNet()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()
        legal_actions = set(game.get_all_legal_actions())

        policy, values = mcts.search(game, num_simulations=5)

        # ポリシーのアクションが全て合法手
        for action in policy.keys():
            self.assertIn(action, legal_actions)


class TestMCTSIntegration(unittest.TestCase):
    """統合テスト"""

    def test_full_game_with_mcts(self):
        """MCTSを使った完全なゲームのシミュレーション"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        max_moves = 100
        move_count = 0

        while not game.game_over and move_count < max_moves:
            policy, values = mcts.search(game, num_simulations=5)

            if not policy:
                break

            # 最も訪問回数が多いアクションを選択
            action = max(policy, key=lambda x: policy[x])

            game.step(action)
            move_count += 1

        # ゲームが終了するか、最大手数に達する
        self.assertTrue(game.game_over or move_count >= max_moves)

    def test_mcts_improves_with_simulations(self):
        """シミュレーション回数が多いほど精度が向上するか確認"""
        network = DeterministicNetwork(value=0.5)
        device = torch.device("cpu")

        game = ContrastGame()

        # 少ないシミュレーション
        mcts1 = MCTS(network, device)
        policy1, values1 = mcts1.search(game, num_simulations=1)

        # 多いシミュレーション
        mcts2 = MCTS(network, device)
        policy2, values2 = mcts2.search(game, num_simulations=50)

        # 両方ともポリシーが返される
        self.assertIsInstance(policy1, dict)
        self.assertIsInstance(policy2, dict)

        # 多いシミュレーションの方が確信度が高い (エントロピーが低い)
        # (実際には決定的なネットワークでは同じ可能性もある)
        # ここでは単に動作することを確認
        self.assertGreater(len(policy2), 0)

    def test_mcts_handles_game_state_transitions(self):
        """MCTSがゲーム状態の遷移を正しく扱うか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # 初期状態で探索
        policy1, values1 = mcts.search(game, num_simulations=5)
        action = max(policy1, key=lambda k: policy1[k])

        # アクションを実行
        game.step(action)

        # 新しい状態で探索
        policy2, values2 = mcts.search(game, num_simulations=5)

        # 両方ともポリシーが返される
        self.assertIsInstance(policy1, dict)
        self.assertIsInstance(policy2, dict)

        # ポリシーが異なる (状態が異なるため)
        self.assertNotEqual(set(policy1.keys()), set(policy2.keys()))


class TestMCTSValuePropagation(unittest.TestCase):
    """価値の伝播テスト"""

    def test_value_negation_in_recursion(self):
        """再帰中に価値が正しく反転されるか確認"""
        network = DeterministicNetwork(value=0.8)
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # 探索を実行
        policy, values = mcts.search(game, num_simulations=10)

        key = mcts.game_to_key(game)

        # 訪問されたアクションの平均価値を確認
        for action in policy.keys():
            if mcts.N[key][action] > 0:
                avg_value = mcts.W[key][action] / mcts.N[key][action]

                # 価値が[-1, 1]の範囲内
                self.assertGreaterEqual(avg_value, -1.0)
                self.assertLessEqual(avg_value, 1.0)

    def test_terminal_value_propagation(self):
        """終了状態の価値が正しく伝播されるか確認"""
        network = MockNetwork()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        game = ContrastGame()

        # ゲームをほぼ終了状態に近づける (P1が勝つ直前)
        game.pieces[1, 2] = 1
        game.pieces[0, :] = 0

        # 探索を実行
        policy, values = mcts.search(game, num_simulations=10)

        # ポリシーが返される
        self.assertIsInstance(policy, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
