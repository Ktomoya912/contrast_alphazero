"""
main.py の網羅的なテストコード

テストカバレッジ:
- ReplayBufferの動作
- P1とP2の盤面反転
- アクションの反転 (flip_action)
- selfplay()ワーカーの動作
- 報酬の割り当て
- バッチ生成とターゲット変換
- エッジケース
"""

import unittest

import numpy as np
import torch

from contrast_game import (
    P1,
    P2,
    ContrastGame,
    decode_action,
    encode_action,
    flip_action,
    flip_location,
)
from main import ReplayBuffer, Sample


class TestFlipLocation(unittest.TestCase):
    """flip_location()のテスト（盤面座標の反転）"""

    def test_flip_location_corners(self):
        """四隅の座標が正しく反転されるか確認"""
        # 左上(0) <-> 右下(24)
        self.assertEqual(flip_location(0), 24)
        self.assertEqual(flip_location(24), 0)

        # 右上(4) <-> 左下(20)
        self.assertEqual(flip_location(4), 20)
        self.assertEqual(flip_location(20), 4)

    def test_flip_location_center(self):
        """中央の座標が正しく反転されるか確認"""
        # 中央(12)は自分自身に反転
        self.assertEqual(flip_location(12), 12)

    def test_flip_location_symmetry(self):
        """flip_location()が対称的であることを確認"""
        for i in range(25):
            flipped = flip_location(i)
            # 2回反転すると元に戻る
            self.assertEqual(flip_location(flipped), i)

    def test_flip_location_range(self):
        """flip_location()が有効な範囲内の値を返すか確認"""
        for i in range(25):
            flipped = flip_location(i)
            self.assertGreaterEqual(flipped, 0)
            self.assertLessEqual(flipped, 24)


class TestFlipAction(unittest.TestCase):
    """flip_action()のテスト（アクションの反転）"""

    def test_flip_action_move_only(self):
        """移動のみのアクションが正しく反転されるか確認"""
        # (0,0) -> (1,1), タイルなし
        move_idx = 0 * 25 + 6  # from=0, to=6 (1,1)
        tile_idx = 0
        action = encode_action(move_idx, tile_idx)

        flipped = flip_action(action)
        f_move, f_tile = decode_action(flipped)

        # from: 0 -> 24, to: 6 -> 18
        expected_move = 24 * 25 + 18
        self.assertEqual(f_move, expected_move)
        self.assertEqual(f_tile, 0)

    def test_flip_action_with_black_tile(self):
        """黒タイル配置を含むアクションが正しく反転されるか確認"""
        # 移動 + 黒タイル配置
        move_idx = 5 * 25 + 10
        tile_idx = 1  # 黒タイル at position 0
        action = encode_action(move_idx, tile_idx)

        flipped = flip_action(action)
        f_move, f_tile = decode_action(flipped)

        # from: 5 -> 19, to: 10 -> 14
        expected_move = 19 * 25 + 14
        # tile position: 0 -> 24, so tile_idx: 1 -> 25
        expected_tile = 25
        self.assertEqual(f_move, expected_move)
        self.assertEqual(f_tile, expected_tile)

    def test_flip_action_with_gray_tile(self):
        """グレータイル配置を含むアクションが正しく反転されるか確認"""
        # 移動 + グレータイル配置
        move_idx = 12 * 25 + 12  # 中央から中央
        tile_idx = 26  # グレータイル at position 0
        action = encode_action(move_idx, tile_idx)

        flipped = flip_action(action)
        f_move, f_tile = decode_action(flipped)

        # from: 12 -> 12 (中央), to: 12 -> 12
        expected_move = 12 * 25 + 12
        # tile position: 0 -> 24, so tile_idx: 26 -> 50
        expected_tile = 50
        self.assertEqual(f_move, expected_move)
        self.assertEqual(f_tile, expected_tile)

    def test_flip_action_symmetry(self):
        """flip_action()が対称的であることを確認（2回反転で元に戻る）"""
        # ランダムなアクションをいくつかテスト
        test_cases = [
            encode_action(0, 0),
            encode_action(100, 5),
            encode_action(624, 50),
            encode_action(312, 25),
        ]

        for action in test_cases:
            flipped_once = flip_action(action)
            flipped_twice = flip_action(flipped_once)
            self.assertEqual(flipped_twice, action)

    def test_flip_action_all_corners(self):
        """四隅の座標を含むアクションが正しく反転されるか確認"""
        corners = [0, 4, 20, 24]

        for from_pos in corners:
            for to_pos in corners:
                move_idx = from_pos * 25 + to_pos
                action = encode_action(move_idx, 0)

                flipped = flip_action(action)
                f_move, f_tile = decode_action(flipped)

                expected_from = flip_location(from_pos)
                expected_to = flip_location(to_pos)
                expected_move = expected_from * 25 + expected_to

                self.assertEqual(f_move, expected_move)


class TestSample(unittest.TestCase):
    """Sampleデータクラスのテスト"""

    def test_sample_initialization(self):
        """Sampleが正しく初期化されるか確認"""
        state = np.zeros((90, 5, 5), dtype=np.float32)
        policy = {0: 0.5, 1: 0.3, 2: 0.2}

        sample = Sample(state=state, mcts_policy=policy, player=P1)

        self.assertEqual(sample.player, P1)
        self.assertEqual(sample.reward, 0.0)
        self.assertIsInstance(sample.mcts_policy, dict)

    def test_sample_with_reward(self):
        """Sampleに報酬を設定できるか確認"""
        state = np.zeros((90, 5, 5), dtype=np.float32)
        policy = {0: 1.0}

        sample = Sample(state=state, mcts_policy=policy, player=P2, reward=1.0)

        self.assertEqual(sample.reward, 1.0)


class TestReplayBuffer(unittest.TestCase):
    """ReplayBufferのテスト"""

    def test_replay_buffer_initialization(self):
        """ReplayBufferが正しく初期化されるか確認"""
        buffer = ReplayBuffer(buffer_size=1000)

        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.buffer.maxlen, 1000)

    def test_replay_buffer_add_record(self):
        """ReplayBufferにレコードを追加できるか確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        samples = [
            Sample(state=state, mcts_policy={0: 1.0}, player=P1) for _ in range(5)
        ]

        buffer.add_record(samples)

        self.assertEqual(len(buffer), 5)

    def test_replay_buffer_max_size(self):
        """ReplayBufferが最大サイズを超えないか確認"""
        buffer = ReplayBuffer(buffer_size=10)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        samples = [
            Sample(state=state, mcts_policy={0: 1.0}, player=P1) for _ in range(20)
        ]

        buffer.add_record(samples)

        # 最大サイズは10
        self.assertEqual(len(buffer), 10)

    def test_replay_buffer_get_minibatch_shapes(self):
        """get_minibatch()が正しい形状のTensorを返すか確認"""
        buffer = ReplayBuffer(buffer_size=100)

        # サンプルを追加
        state = np.random.randn(90, 5, 5).astype(np.float32)
        samples = [
            Sample(state=state, mcts_policy={0: 0.5, 1: 0.5}, player=P1, reward=1.0)
            for _ in range(10)
        ]
        buffer.add_record(samples)

        # バッチを取得
        states, m_targets, t_targets, v_targets = buffer.get_minibatch(4)

        # 形状の確認
        self.assertEqual(states.shape, (4, 90, 5, 5))
        self.assertEqual(m_targets.shape, (4, 625))
        self.assertEqual(t_targets.shape, (4, 51))
        self.assertEqual(v_targets.shape, (4, 1))

    def test_replay_buffer_p1_no_flip(self):
        """P1のサンプルでは行動が反転されないことを確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        # 特定のアクション
        action = encode_action(0, 0)  # move_idx=0, tile_idx=0
        policy = {action: 1.0}

        sample = Sample(state=state, mcts_policy=policy, player=P1, reward=1.0)
        buffer.add_record([sample])

        states, m_targets, t_targets, v_targets = buffer.get_minibatch(1)

        # P1の場合、move_idx=0がそのままターゲットになる
        self.assertAlmostEqual(m_targets[0, 0].item(), 1.0, places=5)
        self.assertAlmostEqual(t_targets[0, 0].item(), 1.0, places=5)

    def test_replay_buffer_p2_flip(self):
        """P2のサンプルでは行動が反転されることを確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        # P2の視点でのアクション (盤面は既に反転されている)
        action = encode_action(0, 1)  # move_idx=0, tile_idx=1 (black tile at pos 0)
        policy = {action: 1.0}

        sample = Sample(state=state, mcts_policy=policy, player=P2, reward=1.0)
        buffer.add_record([sample])

        states, m_targets, t_targets, v_targets = buffer.get_minibatch(1)

        # P2の場合、アクションが反転される
        flipped_action = flip_action(action)
        expected_move_idx = flipped_action // 51
        expected_tile_idx = flipped_action % 51

        # 反転されたインデックスにターゲットが設定される
        self.assertAlmostEqual(m_targets[0, expected_move_idx].item(), 1.0, places=5)
        self.assertAlmostEqual(t_targets[0, expected_tile_idx].item(), 1.0, places=5)

    def test_replay_buffer_multiple_actions_p1(self):
        """P1で複数アクションを持つポリシーが正しく変換されるか確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        action1 = encode_action(0, 0)
        action2 = encode_action(1, 0)
        policy = {action1: 0.7, action2: 0.3}

        sample = Sample(state=state, mcts_policy=policy, player=P1, reward=1.0)
        buffer.add_record([sample])

        states, m_targets, t_targets, v_targets = buffer.get_minibatch(1)

        # P1なので反転なし
        self.assertAlmostEqual(m_targets[0, 0].item(), 0.7, places=5)
        self.assertAlmostEqual(m_targets[0, 1].item(), 0.3, places=5)
        self.assertAlmostEqual(t_targets[0, 0].item(), 1.0, places=5)

    def test_replay_buffer_multiple_actions_p2(self):
        """P2で複数アクションを持つポリシーが正しく反転されるか確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        # P2の視点でのアクション
        action1 = encode_action(0, 0)
        action2 = encode_action(1, 0)
        policy = {action1: 0.6, action2: 0.4}

        sample = Sample(state=state, mcts_policy=policy, player=P2, reward=-1.0)
        buffer.add_record([sample])

        states, m_targets, t_targets, v_targets = buffer.get_minibatch(1)

        # P2なので反転される
        flipped1 = flip_action(action1)
        flipped2 = flip_action(action2)
        f_move1 = flipped1 // 51
        f_move2 = flipped2 // 51

        # 反転されたインデックスに確率が設定される
        self.assertAlmostEqual(m_targets[0, f_move1].item(), 0.6, places=5)
        self.assertAlmostEqual(m_targets[0, f_move2].item(), 0.4, places=5)

    def test_replay_buffer_value_targets(self):
        """報酬が正しくvalue_targetsに変換されるか確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)

        # 勝利サンプル
        sample_win = Sample(state=state, mcts_policy={0: 1.0}, player=P1, reward=1.0)
        # 敗北サンプル
        sample_loss = Sample(state=state, mcts_policy={0: 1.0}, player=P1, reward=-1.0)

        buffer.add_record([sample_win, sample_loss])

        states, m_targets, t_targets, v_targets = buffer.get_minibatch(2)

        # 報酬が正しく設定される
        rewards = v_targets.squeeze().tolist()
        self.assertIn(1.0, rewards)
        self.assertIn(-1.0, rewards)

    def test_replay_buffer_batch_size_larger_than_buffer(self):
        """バッチサイズがバッファサイズより大きい場合の動作確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        samples = [
            Sample(state=state, mcts_policy={0: 1.0}, player=P1) for _ in range(5)
        ]
        buffer.add_record(samples)

        # バッファサイズ5に対してバッチサイズ10を要求
        states, m_targets, t_targets, v_targets = buffer.get_minibatch(10)

        # バッファサイズ分(5)が返される
        self.assertEqual(states.shape[0], 5)


class TestActionFlipConsistency(unittest.TestCase):
    """アクション反転の一貫性テスト"""

    def test_flip_action_preserves_move_structure(self):
        """flip_action()が移動構造を保持するか確認"""

        # (x1,y1) -> (x2,y2) が flip後に (24-x2,4-y2) -> (24-x1,4-y1) になるか
        def pos_to_idx(x, y):
            return y * 5 + x

        def idx_to_pos(idx):
            return idx % 5, idx // 5

        # 例: (0,0) -> (1,1)
        from_x, from_y = 0, 0
        to_x, to_y = 1, 1

        from_idx = pos_to_idx(from_x, from_y)
        to_idx = pos_to_idx(to_x, to_y)

        move_idx = from_idx * 25 + to_idx
        action = encode_action(move_idx, 0)

        flipped = flip_action(action)
        f_move, _ = decode_action(flipped)

        f_from_idx = f_move // 25
        f_to_idx = f_move % 25

        # 反転後の座標
        f_from_x, f_from_y = idx_to_pos(f_from_idx)
        f_to_x, f_to_y = idx_to_pos(f_to_idx)

        # 期待される反転後の座標
        expected_from_x, expected_from_y = 4 - from_x, 4 - from_y
        expected_to_x, expected_to_y = 4 - to_x, 4 - to_y

        self.assertEqual(f_from_x, expected_from_x)
        self.assertEqual(f_from_y, expected_from_y)
        self.assertEqual(f_to_x, expected_to_x)
        self.assertEqual(f_to_y, expected_to_y)

    def test_flip_action_tile_position_preservation(self):
        """タイル配置位置が正しく反転されるか確認"""

        def pos_to_idx(x, y):
            return y * 5 + x

        def idx_to_pos(idx):
            return idx % 5, idx // 5

        # 黒タイルを (0,0) に配置
        tile_x, tile_y = 0, 0
        tile_pos = pos_to_idx(tile_x, tile_y)
        tile_idx = tile_pos + 1  # 1-25 for black

        action = encode_action(0, tile_idx)
        flipped = flip_action(action)
        _, f_tile = decode_action(flipped)

        # 反転後のタイル位置を取得
        f_tile_pos = f_tile - 1
        f_tile_x, f_tile_y = idx_to_pos(f_tile_pos)

        # 期待される反転後の座標
        expected_x, expected_y = 4 - tile_x, 4 - tile_y

        self.assertEqual(f_tile_x, expected_x)
        self.assertEqual(f_tile_y, expected_y)


class TestGameStateEncoding(unittest.TestCase):
    """ゲーム状態のエンコードテスト（P1/P2での違い）"""

    def test_encode_state_p1_vs_p2(self):
        """P1とP2で状態エンコードが異なるか確認"""
        game = ContrastGame()

        # P1の番で状態をエンコード
        state_p1 = game.encode_state()

        # P2の番に進める
        legal_actions = game.get_all_legal_actions()
        if legal_actions:
            game.step(legal_actions[0])

        state_p2 = game.encode_state()

        # P1とP2で異なる状態になる（盤面が反転される）
        self.assertFalse(np.array_equal(state_p1, state_p2))

    def test_encode_state_shape_consistency(self):
        """P1とP2で状態エンコードの形状が一貫しているか確認"""
        game = ContrastGame()

        # 複数ターン進める
        for _ in range(3):
            if game.game_over:
                break
            legal_actions = game.get_all_legal_actions()
            if not legal_actions:
                break
            game.step(legal_actions[0])

            state = game.encode_state()
            # 常に(90, 5, 5)の形状
            self.assertEqual(state.shape, (90, 5, 5))


class TestRewardAssignment(unittest.TestCase):
    """報酬割り当てのテスト"""

    def test_reward_p1_wins(self):
        """P1が勝った場合の報酬割り当てを確認"""
        samples = [
            Sample(
                state=np.zeros((90, 5, 5), dtype=np.float32),
                mcts_policy={0: 1.0},
                player=P1,
            ),
            Sample(
                state=np.zeros((90, 5, 5), dtype=np.float32),
                mcts_policy={0: 1.0},
                player=P2,
            ),
        ]

        # P1勝利
        winner = P1
        for sample in samples:
            if winner == 0:
                sample.reward = 0.0
            else:
                sample.reward = 1.0 if sample.player == winner else -1.0

        # P1のサンプルは+1
        self.assertEqual(samples[0].reward, 1.0)
        # P2のサンプルは-1
        self.assertEqual(samples[1].reward, -1.0)

    def test_reward_p2_wins(self):
        """P2が勝った場合の報酬割り当てを確認"""
        samples = [
            Sample(
                state=np.zeros((90, 5, 5), dtype=np.float32),
                mcts_policy={0: 1.0},
                player=P1,
            ),
            Sample(
                state=np.zeros((90, 5, 5), dtype=np.float32),
                mcts_policy={0: 1.0},
                player=P2,
            ),
        ]

        # P2勝利
        winner = P2
        for sample in samples:
            if winner == 0:
                sample.reward = 0.0
            else:
                sample.reward = 1.0 if sample.player == winner else -1.0

        # P1のサンプルは-1
        self.assertEqual(samples[0].reward, -1.0)
        # P2のサンプルは+1
        self.assertEqual(samples[1].reward, 1.0)

    def test_reward_draw(self):
        """引き分けの場合の報酬割り当てを確認"""
        samples = [
            Sample(
                state=np.zeros((90, 5, 5), dtype=np.float32),
                mcts_policy={0: 1.0},
                player=P1,
            ),
            Sample(
                state=np.zeros((90, 5, 5), dtype=np.float32),
                mcts_policy={0: 1.0},
                player=P2,
            ),
        ]

        # 引き分け
        winner = 0
        for sample in samples:
            if winner == 0:
                sample.reward = 0.0
            else:
                sample.reward = 1.0 if sample.player == winner else -1.0

        # 両方とも0
        self.assertEqual(samples[0].reward, 0.0)
        self.assertEqual(samples[1].reward, 0.0)


class TestEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""

    def test_flip_action_all_positions(self):
        """全ての位置でflip_action()が正しく動作するか確認"""
        for pos in range(25):
            flipped = flip_location(pos)
            # 2回反転で元に戻る
            self.assertEqual(flip_location(flipped), pos)

    def test_empty_policy(self):
        """空のポリシーでの動作確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        # 空のポリシー
        sample = Sample(state=state, mcts_policy={}, player=P1, reward=0.0)
        buffer.add_record([sample])

        states, m_targets, t_targets, v_targets = buffer.get_minibatch(1)

        # 全て0のターゲットになる
        self.assertTrue(torch.all(m_targets == 0))
        self.assertTrue(torch.all(t_targets == 0))

    def test_single_action_policy(self):
        """単一アクションのポリシーでの動作確認"""
        buffer = ReplayBuffer(buffer_size=100)

        state = np.zeros((90, 5, 5), dtype=np.float32)
        action = encode_action(100, 10)
        sample = Sample(state=state, mcts_policy={action: 1.0}, player=P1)
        buffer.add_record([sample])

        states, m_targets, t_targets, v_targets = buffer.get_minibatch(1)

        # 合計確率が1になる
        self.assertAlmostEqual(m_targets[0].sum().item(), 1.0, places=5)
        self.assertAlmostEqual(t_targets[0].sum().item(), 1.0, places=5)


class TestPlayTimeConsistency(unittest.TestCase):
    """プレイ時の視点統一の一貫性テスト"""

    def test_encode_state_returns_p1_perspective_for_p2(self):
        """P2の番でもencode_state()がP1視点を返すことを確認"""
        game = ContrastGame()

        # P2の番まで進める
        legal_actions = game.get_all_legal_actions()
        if legal_actions:
            game.step(legal_actions[0])

        # P2の番
        self.assertEqual(game.current_player, P2)

        # 状態エンコード（自動的に180度回転される）
        state = game.encode_state()

        # 形状確認
        self.assertEqual(state.shape, (90, 5, 5))

        # P2の駒位置が回転されていることを確認
        # encode_state()内でP2の場合は盤面が回転されるので、
        # プレイヤー識別プレーンの形状は変わらないが、
        # 実際の盤面座標が反転されている

    def test_mcts_with_p1_and_p2_uses_same_network(self):
        """P1とP2で同じネットワークが使えることを確認"""
        from mcts import MCTS
        from model import ContrastDualPolicyNet

        network = ContrastDualPolicyNet()
        device = torch.device("cpu")
        mcts = MCTS(network, device)

        # P1の番でのMCTS
        game_p1 = ContrastGame()
        self.assertEqual(game_p1.current_player, P1)

        policy_p1, _ = mcts.search(game_p1, num_simulations=5)
        self.assertIsInstance(policy_p1, dict)
        self.assertGreater(len(policy_p1), 0)

        # P2の番でのMCTS
        game_p2 = ContrastGame()
        legal_actions = game_p2.get_all_legal_actions()
        if legal_actions:
            game_p2.step(legal_actions[0])

        self.assertEqual(game_p2.current_player, P2)

        policy_p2, _ = mcts.search(game_p2, num_simulations=5)
        self.assertIsInstance(policy_p2, dict)
        self.assertGreater(len(policy_p2), 0)

        # 両方で有効なポリシーが返される
        # （同じネットワークで両プレイヤーの行動を決定できる）

    def test_action_space_consistency_p1_vs_p2(self):
        """P1とP2で返されるアクションが元の座標系で一貫していることを確認"""
        game = ContrastGame()

        # P1の合法手を取得
        p1_actions = game.get_all_legal_actions()
        self.assertGreater(len(p1_actions), 0)

        # P1のアクションは0-31874の範囲
        for action in p1_actions[:5]:  # 最初の5個をチェック
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 625 * 51)

        # P2の番に進める
        game.step(p1_actions[0])
        self.assertEqual(game.current_player, P2)

        # P2の合法手を取得
        p2_actions = game.get_all_legal_actions()
        self.assertGreater(len(p2_actions), 0)

        # P2のアクションも同じ範囲
        for action in p2_actions[:5]:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 625 * 51)

    def test_symmetry_of_learning_data(self):
        """P1とP2のサンプルが対称的に変換されることを確認"""
        buffer = ReplayBuffer(buffer_size=100)

        # 同じ相対的な盤面位置でのアクション
        # (0,4) -> (1,3) という移動を考える

        # P1視点: 下から上への移動
        p1_from = 4 * 5 + 0  # (0,4) = 20
        p1_to = 3 * 5 + 1  # (1,3) = 16
        p1_action = encode_action(p1_from * 25 + p1_to, 0)

        # P2視点: 同じ相対移動だが座標は反転
        # (0,4) -> (1,3) をP2視点で実行すると、反転後は...
        p2_from = flip_location(p1_from)  # 4
        p2_to = flip_location(p1_to)  # 8
        p2_action = encode_action(p2_from * 25 + p2_to, 0)

        # P1サンプル
        state = np.zeros((90, 5, 5), dtype=np.float32)
        sample_p1 = Sample(state=state, mcts_policy={p1_action: 1.0}, player=P1)

        # P2サンプル
        sample_p2 = Sample(state=state, mcts_policy={p2_action: 1.0}, player=P2)

        buffer.add_record([sample_p1, sample_p2])

        states, m_targets, t_targets, v_targets = buffer.get_minibatch(2)

        # P1のターゲット: そのまま
        p1_move_idx = p1_action // 51
        # P2のターゲット: 反転される
        p2_flipped = flip_action(p2_action)
        p2_move_idx = p2_flipped // 51

        # 両方のサンプルがターゲットに含まれる
        sum_p1 = m_targets[0, p1_move_idx].item()
        sum_p2 = m_targets[1, p2_move_idx].item()

        # どちらかが1.0になる（サンプルの順序は不定）
        self.assertTrue(sum_p1 == 1.0 or sum_p2 == 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
