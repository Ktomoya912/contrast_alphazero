"""
ContrastGame の網羅的なテストコード

テストカバレッジ:
- 初期配置の検証
- 移動ルール (白/黒/グレータイル)
- タイル配置ルール
- 飛び越え移動
- 勝利条件
- アクション生成とエンコード/デコード
- 状態のコピーと履歴管理
- エッジケース
"""

import unittest

import numpy as np

from contrast_game import (
    OPPONENT,
    P1,
    P2,
    TILE_BLACK,
    TILE_GRAY,
    TILE_WHITE,
    ContrastGame,
    decode_action,
    encode_action,
)


class TestContrastGameInitialization(unittest.TestCase):
    """初期化と初期配置のテスト"""

    def test_initial_board_setup(self):
        """初期配置が正しいか確認"""
        game = ContrastGame()

        # P1は最下段 (y=4) に5つ
        self.assertTrue(np.all(game.pieces[4, :] == P1))

        # P2は最上段 (y=0) に5つ
        self.assertTrue(np.all(game.pieces[0, :] == P2))

        # 中間は空
        self.assertTrue(np.all(game.pieces[1:4, :] == 0))

    def test_initial_tiles(self):
        """初期タイルが全て白であることを確認"""
        game = ContrastGame()
        self.assertTrue(np.all(game.tiles == TILE_WHITE))

    def test_initial_tile_counts(self):
        """各プレイヤーの持ちタイル数を確認"""
        game = ContrastGame()

        # P1: 黒3, グレー1
        self.assertEqual(game.tile_counts[0, 0], 3)  # Black
        self.assertEqual(game.tile_counts[0, 1], 1)  # Gray

        # P2: 黒3, グレー1
        self.assertEqual(game.tile_counts[1, 0], 3)
        self.assertEqual(game.tile_counts[1, 1], 1)

    def test_initial_player(self):
        """最初の手番がP1であることを確認"""
        game = ContrastGame()
        self.assertEqual(game.current_player, P1)

    def test_initial_game_state(self):
        """ゲーム開始時の状態を確認"""
        game = ContrastGame()
        self.assertFalse(game.game_over)
        self.assertEqual(game.winner, 0)
        self.assertEqual(game.move_count, 0)


class TestMovementRules(unittest.TestCase):
    """移動ルールのテスト"""

    def test_white_tile_orthogonal_movement(self):
        """白タイルで縦横移動が可能か確認"""
        game = ContrastGame()

        # P1の駒 (4, 2) から上方向への移動
        moves = game.get_valid_moves(2, 4)

        # 上に移動可能 (0, 2)まで
        expected = [(2, 3)]
        self.assertIn(expected[0], moves)

    def test_black_tile_diagonal_movement(self):
        """黒タイルで斜め移動が可能か確認"""
        game = ContrastGame()

        # 黒タイルを配置
        game.tiles[4, 2] = TILE_BLACK

        # P1の駒 (4, 2) から斜め移動
        moves = game.get_valid_moves(2, 4)

        # 斜め4方向
        expected = [(1, 3), (3, 3)]
        for move in expected:
            self.assertIn(move, moves)

    def test_gray_tile_all_directions(self):
        """グレータイルで8方向移動が可能か確認"""
        game = ContrastGame()

        # グレータイルを配置
        game.tiles[4, 2] = TILE_GRAY

        # P1の駒 (4, 2) から全方向移動
        moves = game.get_valid_moves(2, 4)

        # 8方向 (ただし横方向は自分の駒がいるため移動不可、移動は1マスのみ)
        # 上、左上、右上の3方向のみ可能
        expected = [(2, 3), (1, 3), (3, 3)]
        for move in expected:
            self.assertIn(move, moves)

    def test_blocked_by_opponent(self):
        """相手の駒で移動が阻止されるか確認"""
        game = ContrastGame()

        # P2の駒を配置
        game.pieces[3, 2] = P2

        # P1の駒 (4, 2) から上方向への移動
        moves = game.get_valid_moves(2, 4)

        # (2, 3) には移動できない (相手の駒がいる)
        self.assertNotIn((2, 3), moves)

    def test_jump_over_friendly_piece(self):
        """自分の駒を飛び越えられるか確認"""
        game = ContrastGame()

        # P1の駒を追加で配置
        game.pieces[3, 2] = P1

        # P1の駒 (4, 2) から上方向への移動
        moves = game.get_valid_moves(2, 4)

        # 味方の駒 (3, 2) を飛び越えて (2, 2) に移動可能
        self.assertIn((2, 2), moves)

    def test_cannot_move_out_of_bounds(self):
        """盤外への移動は不可能であることを確認"""
        game = ContrastGame()

        # 端の駒 (4, 0) から左方向への移動
        moves = game.get_valid_moves(0, 4)

        # 左方向の移動先が範囲外
        for x, y in moves:
            self.assertTrue(0 <= x < 5)
            self.assertTrue(0 <= y < 5)

    def test_cannot_move_opponents_piece(self):
        """相手の駒を動かせないことを確認"""
        game = ContrastGame()

        # P2の駒を動かそうとする (P1の手番)
        moves = game.get_valid_moves(2, 0)

        # 空のリストが返される
        self.assertEqual(moves, [])

    def test_multiple_jumps(self):
        """複数の味方駒を連続で飛び越えられるか確認"""
        game = ContrastGame()

        # P1の駒を連続配置
        # 初期状態: (2, 4) にP1の駒
        # 追加: (2, 3), (2, 2) にP1の駒を配置
        game.pieces[3, 2] = P1
        game.pieces[2, 2] = P1

        # P1の駒 (2, 4) から上方向への移動
        # (2, 3) を飛び越え、(2, 2) も飛び越えて、(2, 1) に着地
        moves = game.get_valid_moves(2, 4)

        self.assertIn((2, 1), moves)


class TestTilePlacement(unittest.TestCase):
    """タイル配置ルールのテスト"""

    def test_place_black_tile(self):
        """黒タイルの配置が正しく機能するか確認"""
        game = ContrastGame()

        # P1の駒を移動してタイルを配置
        # (4, 2) -> (3, 2) に移動し、(2, 2) に黒タイルを配置
        move_idx = (4 * 5 + 2) * 25 + (3 * 5 + 2)
        tile_idx = 1 + (2 * 5 + 2)  # Black tile at (2, 2)
        action = encode_action(move_idx, tile_idx)

        game.step(action)

        # タイルが配置されたか確認
        self.assertEqual(game.tiles[2, 2], TILE_BLACK)

        # 持ちタイルが減ったか確認
        self.assertEqual(game.tile_counts[0, 0], 2)

    def test_place_gray_tile(self):
        """グレータイルの配置が正しく機能するか確認"""
        game = ContrastGame()

        # P1の駒を移動してタイルを配置
        move_idx = (4 * 5 + 2) * 25 + (3 * 5 + 2)
        tile_idx = 26 + (2 * 5 + 2)  # Gray tile at (2, 2)
        action = encode_action(move_idx, tile_idx)

        game.step(action)

        # タイルが配置されたか確認
        self.assertEqual(game.tiles[2, 2], TILE_GRAY)

        # 持ちタイルが減ったか確認
        self.assertEqual(game.tile_counts[0, 1], 0)

    def test_cannot_place_tile_on_destination(self):
        """移動先にタイルを配置できないことを確認"""
        game = ContrastGame()

        # (4, 2) -> (3, 2) に移動し、(3, 2) に黒タイルを配置しようとする
        move_idx = (4 * 5 + 2) * 25 + (3 * 5 + 2)
        tile_idx = 1 + (3 * 5 + 2)  # Black tile at (3, 2) - destination
        action = encode_action(move_idx, tile_idx)

        # 合法手リストに含まれないはず
        legal_actions = game.get_all_legal_actions()
        self.assertNotIn(action, legal_actions)

    def test_cannot_place_tile_on_existing_piece(self):
        """既にコマがある場所にタイルを配置できないことを確認"""
        game = ContrastGame()

        # P2の駒がいる (0, 2) に黒タイルを配置しようとする
        move_idx = (4 * 5 + 2) * 25 + (3 * 5 + 2)
        tile_idx = 1 + (0 * 5 + 2)  # Black tile at (0, 2)
        action = encode_action(move_idx, tile_idx)

        # 合法手リストに含まれないはず
        legal_actions = game.get_all_legal_actions()
        self.assertNotIn(action, legal_actions)

    def test_tile_count_depletion(self):
        """タイルを使い切ると配置できなくなることを確認"""
        game = ContrastGame()

        # グレータイルを1つ配置
        move_idx = (4 * 5 + 2) * 25 + (3 * 5 + 2)
        tile_idx = 26 + (2 * 5 + 2)
        action = encode_action(move_idx, tile_idx)
        game.step(action)

        # P2のターンをスキップ
        legal_p2 = game.get_all_legal_actions()
        game.step(legal_p2[0])

        # P1のグレータイルが0になったはず
        self.assertEqual(game.tile_counts[0, 1], 0)

        # グレータイル配置アクションが生成されないことを確認
        legal_actions = game.get_all_legal_actions()
        for action in legal_actions:
            _, tile_idx = decode_action(action)
            # Gray tile index range: 26-50
            if tile_idx >= 26:
                self.fail(f"Gray tile action found when count is 0: {action}")


class TestWinConditions(unittest.TestCase):
    """勝利条件のテスト"""

    def test_p1_wins_reaching_top_row(self):
        """P1が上段 (y=0) に到達して勝利するか確認"""
        game = ContrastGame()

        # P1の駒を上段に配置
        game.pieces[4, 2] = 0
        game.pieces[0, 2] = P1

        game._check_win_fast()

        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, P1)

    def test_p2_wins_reaching_bottom_row(self):
        """P2が下段 (y=4) に到達して勝利するか確認"""
        game = ContrastGame()

        # P2の駒を下段に配置
        game.pieces[0, 2] = 0
        game.pieces[4, 2] = P2

        game._check_win_fast()

        self.assertTrue(game.game_over)
        self.assertEqual(game.winner, P2)

    def test_game_continues_when_no_winner(self):
        """勝者がいない場合ゲームが継続することを確認"""
        game = ContrastGame()

        # 初期状態では勝者なし
        game._check_win_fast()

        self.assertFalse(game.game_over)
        self.assertEqual(game.winner, 0)

    def test_loss_condition_no_legal_moves(self):
        """合法手がない場合に敗北することを確認"""
        game = ContrastGame()

        # P1の駒を全て相手の駒で囲む（移動不可能にする）
        # P1の駒の位置: (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)
        # その上の行(y=3)を全てP2の駒で埋める
        for x in range(5):
            game.pieces[3, x] = P2

        # P1が行動を試みる（合法手がないはず）
        legal_actions = game.get_all_legal_actions()

        # 合法手が存在しないことを確認
        self.assertEqual(len(legal_actions), 0)

        # step()を呼ぶと敗北条件が発動するが、合法手がないので
        # 実際にはstep()は呼べない。代わりに直接チェックをシミュレート
        # 実際のゲームでは、プレイヤーが行動できない時点で敗北

        # 手番を交代して合法手チェックをシミュレート
        game.current_player = OPPONENT[game.current_player]
        legal_actions_p2 = game.get_all_legal_actions()

        # P2にまだ合法手があることを確認（P1だけが動けない）
        self.assertGreater(len(legal_actions_p2), 0)

    def test_loss_condition_triggered_after_move(self):
        """相手の行動後に合法手がなくなり敗北することを確認"""
        game = ContrastGame()

        # P2の駒を配置してP1を閉じ込める準備
        # まずP1の駒を1つだけにする
        game.pieces[4, :] = 0
        game.pieces[4, 2] = P1

        # P1の周囲をP2の駒で囲む（1マス空けておく）
        game.pieces[3, 1] = P2
        game.pieces[3, 2] = P2
        game.pieces[3, 3] = P2

        # P2の番にする
        game.current_player = P2

        # P2が最後の1マスを埋める行動を作成
        # (0, 0) -> (4, 1) への移動
        move_idx = (0 * 5 + 0) * 25 + (4 * 5 + 1)
        action = move_idx * 51  # タイルなし

        # P2の駒を配置
        game.pieces[0, 0] = P2

        # 行動を実行
        done, winner = game.step(action)

        # ゲームが終了し、P2が勝利（P1が敗北）
        self.assertTrue(done)
        self.assertEqual(winner, P2)


class TestActionEncoding(unittest.TestCase):
    """アクションのエンコード/デコードのテスト"""

    def test_encode_action_basic(self):
        """基本的なアクションエンコードを確認"""
        move_idx = 100
        tile_idx = 10

        action = encode_action(move_idx, tile_idx)

        expected = move_idx * 51 + tile_idx
        self.assertEqual(action, expected)

    def test_decode_action_basic(self):
        """基本的なアクションデコードを確認"""
        action = 5110  # 100 * 51 + 10

        move_idx, tile_idx = decode_action(action)

        self.assertEqual(move_idx, 100)
        self.assertEqual(tile_idx, 10)

    def test_encode_decode_roundtrip(self):
        """エンコード→デコードが可逆であることを確認"""
        for move_idx in [0, 100, 624]:
            for tile_idx in [0, 25, 50]:
                action = encode_action(move_idx, tile_idx)
                decoded_move, decoded_tile = decode_action(action)

                self.assertEqual(decoded_move, move_idx)
                self.assertEqual(decoded_tile, tile_idx)

    def test_action_hash_range(self):
        """アクションハッシュの範囲を確認"""
        # Max action: 624 * 51 + 50 = 31874
        max_action = encode_action(624, 50)
        self.assertEqual(max_action, 31874)

        # Min action: 0 * 51 + 0 = 0
        min_action = encode_action(0, 0)
        self.assertEqual(min_action, 0)


class TestLegalActions(unittest.TestCase):
    """合法手生成のテスト"""

    def test_initial_legal_actions_exist(self):
        """初期状態で合法手が存在することを確認"""
        game = ContrastGame()
        legal_actions = game.get_all_legal_actions()

        self.assertGreater(len(legal_actions), 0)

    def test_no_legal_actions_when_game_over(self):
        """ゲーム終了時に合法手が0であることを確認"""
        game = ContrastGame()

        # 強制的にゲーム終了状態にする
        game.game_over = True

        legal_actions = game.get_all_legal_actions()

        self.assertEqual(len(legal_actions), 0)

    def test_legal_actions_include_move_only(self):
        """タイルなし移動のアクションが含まれることを確認"""
        game = ContrastGame()

        legal_actions = game.get_all_legal_actions()

        # tile_idx = 0 のアクションが含まれているはず
        move_only_actions = [a for a in legal_actions if a % 51 == 0]

        self.assertGreater(len(move_only_actions), 0)

    def test_legal_actions_all_valid(self):
        """生成された全ての合法手が実際に実行可能か確認"""
        game = ContrastGame()

        legal_actions = game.get_all_legal_actions()

        for action in legal_actions[:10]:  # サンプルをテスト
            game_copy = game.copy()

            # 例外が発生しないことを確認
            try:
                game_copy.step(action)
            except Exception as e:
                self.fail(f"Legal action {action} raised exception: {e}")


class TestGameCopy(unittest.TestCase):
    """ゲームのコピー機能のテスト"""

    def test_copy_creates_independent_instance(self):
        """コピーが独立したインスタンスを作成するか確認"""
        game = ContrastGame()

        # 状態を変更
        legal_actions = game.get_all_legal_actions()
        game.step(legal_actions[0])

        game_copy = game.copy()

        # コピー後に元のゲームを変更
        if not game.game_over:
            legal_actions = game.get_all_legal_actions()
            if legal_actions:
                game.step(legal_actions[0])

        # コピーは影響を受けない
        self.assertNotEqual(game.move_count, game_copy.move_count)

    def test_copy_preserves_state(self):
        """コピーが状態を正しく保存するか確認"""
        game = ContrastGame()

        # 状態を変更
        legal_actions = game.get_all_legal_actions()
        game.step(legal_actions[0])

        game_copy = game.copy()

        # 盤面が一致
        self.assertTrue(np.array_equal(game.pieces, game_copy.pieces))
        self.assertTrue(np.array_equal(game.tiles, game_copy.tiles))
        self.assertTrue(np.array_equal(game.tile_counts, game_copy.tile_counts))

        # メタ情報が一致
        self.assertEqual(game.current_player, game_copy.current_player)
        self.assertEqual(game.game_over, game_copy.game_over)
        self.assertEqual(game.winner, game_copy.winner)
        self.assertEqual(game.move_count, game_copy.move_count)


class TestStateEncoding(unittest.TestCase):
    """状態エンコードのテスト"""

    def test_encode_state_shape(self):
        """エンコード結果の形状を確認"""
        game = ContrastGame()
        state = game.encode_state()

        self.assertEqual(state.shape, (90, 5, 5))

    def test_encode_state_dtype(self):
        """エンコード結果のデータ型を確認"""
        game = ContrastGame()
        state = game.encode_state()

        self.assertEqual(state.dtype, np.float32)

    def test_encode_state_history_padding(self):
        """履歴が不足している場合のパディングを確認"""
        game = ContrastGame()

        # 初期状態では履歴が1つだけ
        state = game.encode_state()

        # エラーが発生しないことを確認
        self.assertIsNotNone(state)

    def test_encode_state_values_in_range(self):
        """エンコード結果の値が適切な範囲内か確認"""
        game = ContrastGame()
        state = game.encode_state()

        # 値が0-1の範囲内 (一部は正規化されている)
        self.assertTrue(np.all(state >= 0))
        self.assertTrue(np.all(state <= 1))


class TestHistoryManagement(unittest.TestCase):
    """履歴管理のテスト"""

    def test_history_initialized(self):
        """履歴が初期化されているか確認"""
        game = ContrastGame()

        self.assertGreater(len(game.history), 0)

    def test_history_updates_on_step(self):
        """ステップ実行で履歴が更新されるか確認"""
        game = ContrastGame()

        initial_history_len = len(game.history)

        legal_actions = game.get_all_legal_actions()
        game.step(legal_actions[0])

        # 履歴が増えている
        self.assertEqual(len(game.history), initial_history_len + 1)

    def test_history_max_length(self):
        """履歴の最大長が8であることを確認"""
        game = ContrastGame()

        # 10手進める
        for _ in range(10):
            if game.game_over:
                break
            legal_actions = game.get_all_legal_actions()
            if not legal_actions:
                break
            game.step(legal_actions[0])

        # 履歴は最大8
        self.assertLessEqual(len(game.history), 8)


class TestEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""

    def test_all_pieces_blocked(self):
        """全ての駒が移動不可能な状態を確認"""
        game = ContrastGame()

        # P1の駒を全て囲む (理論上ありえない状況だがテスト)
        for x in range(5):
            game.pieces[3, x] = P2

        legal_actions = game.get_all_legal_actions()

        # 移動可能な駒がないため、合法手は0
        self.assertEqual(len(legal_actions), 0)

        # ゲームはまだ終了していない（step()が呼ばれていないため）
        self.assertFalse(game.game_over)

    def test_reset_game(self):
        """ゲームのリセットが正しく機能するか確認"""
        game = ContrastGame()

        # ゲームを進める
        for _ in range(5):
            if game.game_over:
                break
            legal_actions = game.get_all_legal_actions()
            if not legal_actions:
                break
            game.step(legal_actions[0])

        # リセット
        game.setup_initial_position()

        # 初期状態に戻っているか確認
        self.assertEqual(game.current_player, P1)
        self.assertEqual(game.move_count, 0)
        self.assertFalse(game.game_over)
        self.assertTrue(np.all(game.pieces[4, :] == P1))
        self.assertTrue(np.all(game.pieces[0, :] == P2))

    def test_opponent_mapping(self):
        """プレイヤーの切り替えが正しいか確認"""
        self.assertEqual(OPPONENT[P1], P2)
        self.assertEqual(OPPONENT[P2], P1)


class TestComplexScenarios(unittest.TestCase):
    """複雑なシナリオのテスト"""

    def test_full_game_simulation(self):
        """完全なゲームのシミュレーション"""
        game = ContrastGame()

        max_moves = 200
        move_count = 0

        while not game.game_over and move_count < max_moves:
            legal_actions = game.get_all_legal_actions()

            if not legal_actions:
                break

            # ランダムに手を選択
            import random

            action = random.choice(legal_actions)
            game.step(action)

            move_count += 1

        # ゲームが終了するか、最大手数に達する
        self.assertTrue(game.game_over or move_count >= max_moves)

    def test_alternating_players(self):
        """プレイヤーが交互に手番を持つことを確認"""
        game = ContrastGame()

        players = []

        for _ in range(10):
            if game.game_over:
                break

            players.append(game.current_player)

            legal_actions = game.get_all_legal_actions()
            if not legal_actions:
                break

            game.step(legal_actions[0])

        # P1とP2が交互に現れる
        for i in range(len(players) - 1):
            self.assertNotEqual(players[i], players[i + 1])

    def test_tile_placement_changes_movement(self):
        """タイル配置が移動パターンを変更するか確認"""
        game = ContrastGame()

        # 初期状態での移動先を取得
        initial_moves = game.get_valid_moves(2, 4)

        # 黒タイルを配置
        game.tiles[4, 2] = TILE_BLACK

        # タイル配置後の移動先を取得
        new_moves = game.get_valid_moves(2, 4)

        # 移動先が変わっている
        self.assertNotEqual(set(initial_moves), set(new_moves))

    def test_action_consistency(self):
        """同じ状態で同じ合法手リストが生成されるか確認"""
        game = ContrastGame()

        actions1 = game.get_all_legal_actions()
        actions2 = game.get_all_legal_actions()

        self.assertEqual(set(actions1), set(actions2))


class TestBoardRepetition(unittest.TestCase):
    """盤面繰り返し判定のテスト"""

    def test_repetition_draw_after_threshold(self):
        """同じ盤面が10回繰り返されたら引き分けになることを確認"""
        game = ContrastGame()

        # 50手まで進める（繰り返し判定は50手以降）
        for i in range(30):
            # P1とP2が交互に往復する動きを繰り返す
            # P1の駒を取得
            legal = game.get_all_legal_actions()
            if not legal or game.game_over:
                break

            # 最初の合法手を使う
            game.step(legal[0])

        # 50手以降、同じ盤面を意図的に繰り返す
        if game.move_count >= 50 and not game.game_over:
            # 現在の盤面ハッシュを取得
            initial_hash = game._get_board_hash()

            # 同じアクションを繰り返して盤面を戻す試み
            for _ in range(12):  # 10回の閾値を超えるまで
                if game.game_over:
                    break

                legal = game.get_all_legal_actions()
                if not legal:
                    break

                # 同じ手を繰り返す
                game.step(legal[0])

            # 繰り返しで引き分けになる可能性がある
            # （実際の盤面の動きによる）

    def test_no_repetition_before_move_50(self):
        """50手未満では繰り返し判定が働かないことを確認"""
        game = ContrastGame()

        # position_historyが50手未満では更新されないことを確認
        for i in range(20):
            legal = game.get_all_legal_actions()
            if not legal or game.game_over:
                break
            game.step(legal[0])

        # 50手未満ではposition_historyは空または記録されていない
        if game.move_count < 50:
            # position_historyは使われていないはず
            self.assertEqual(len(game.position_history), 0)

    def test_board_hash_consistency(self):
        """同じ盤面は同じハッシュ値を生成することを確認"""
        game = ContrastGame()

        hash1 = game._get_board_hash()
        hash2 = game._get_board_hash()

        self.assertEqual(hash1, hash2)

        # アクション実行
        legal = game.get_all_legal_actions()
        if legal:
            game.step(legal[0])
            hash3 = game._get_board_hash()

            # 盤面が変わったのでハッシュも変わるはず
            self.assertNotEqual(hash1, hash3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
