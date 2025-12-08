import argparse
from pathlib import Path

import torch

from contrast_game import (
    OPPONENT,
    P1,
    P2,
    TILE_BLACK,
    TILE_GRAY,
    TILE_WHITE,
    ContrastGame,
    decode_action,
)
from logger import get_logger, setup_logger
from mcts import MCTS
from model import ContrastDualPolicyNet

logger = get_logger(__name__)


class HumanVsAI:
    def __init__(
        self, model_path, num_simulations=50, player1_type="human", player2_type="ai"
    ):
        """
        Args:
            model_path: 学習済みモデルのパス
            num_simulations: MCTSのシミュレーション回数
            player1_type: プレイヤー1のタイプ ("human", "ai", "random", "rule")
            player2_type: プレイヤー2のタイプ ("human", "ai", "random", "rule")
        """
        self.player1_type = player1_type
        self.player2_type = player2_type
        self.num_simulations = num_simulations
        self.action_history = []

        # デバイス設定
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # モデルのロード
        self.model = ContrastDualPolicyNet().to(self.device)
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(
                f"Model file not found: {model_path}. Using untrained model."
            )
        self.model.eval()

        # MCTS初期化
        self.mcts = MCTS(network=self.model, device=self.device)

        # ゲーム初期化
        self.game = ContrastGame()

    def display_board(self):
        """盤面を表示"""
        print("\n" + "=" * 50)
        print("現在の盤面:")
        print("=" * 50)

        # タイルの表示
        tile_symbols = {TILE_WHITE: "□", TILE_BLACK: "■", TILE_GRAY: "▦"}

        # 列ラベル (a-e)
        print("   ", end="")
        for x in range(5):
            print(f" {chr(ord('a') + x)} ", end="")
        print()

        # 行は5から1へ（下から上）
        for y in range(5):
            row_label = 5 - y  # 5, 4, 3, 2, 1
            print(f" {row_label} ", end="")
            for x in range(5):
                piece = self.game.pieces[y, x]
                tile = self.game.tiles[y, x]

                if piece == P1:
                    symbol = f"[1{tile_symbols[tile]}]"
                elif piece == P2:
                    symbol = f"[2{tile_symbols[tile]}]"
                else:
                    symbol = f" {tile_symbols[tile]} "

                print(symbol, end="")
            print()

        print("\n持ちタイル:")
        print(
            f"  プレイヤー1: 黒={self.game.tile_counts[0, 0]}, グレー={self.game.tile_counts[0, 1]}"
        )
        print(
            f"  プレイヤー2: 黒={self.game.tile_counts[1, 0]}, グレー={self.game.tile_counts[1, 1]}"
        )
        print(f"\n手数: {self.game.move_count}")
        print("=" * 50)

    def parse_position(self, pos_str):
        """位置文字列(例: 'b3')を内部座標(x, y)に変換

        Args:
            pos_str: 'a1'-'e5'形式の文字列
        Returns:
            (x, y): 内部座標 (0-4, 0-4)
        """
        if len(pos_str) != 2:
            raise ValueError("座標は2文字で指定してください (例: b3)")

        col = pos_str[0].lower()
        row = pos_str[1]

        if col not in "abcde":
            raise ValueError("列はa-eで指定してください")
        if row not in "12345":
            raise ValueError("行は1-5で指定してください")

        x = ord(col) - ord("a")  # a=0, b=1, ..., e=4
        y = 5 - int(row)  # 1=4, 2=3, 3=2, 4=1, 5=0 (下から上)

        return x, y

    def format_position(self, x, y):
        """内部座標(x, y)を位置文字列に変換

        Args:
            x, y: 内部座標 (0-4, 0-4)
        Returns:
            'a1'-'e5'形式の文字列
        """
        col = chr(ord("a") + x)
        row = 5 - y
        return f"{col}{row}"

    def get_human_action(self):
        """人間から行動を入力

        入力形式: <移動前>,<移動後> <配置座標><タイルカラー>
        例:
          b1,b2 b3g  (b1からb2へ移動、b3にグレータイルを配置)
          a5,a4 b1b  (a5からa4へ移動、b1に黒タイルを配置)
          c5,c4      (c5からc4へ移動、タイル配置なし)
        """
        print(f"\nあなたの番です (プレイヤー{self.game.current_player})")
        print("入力形式: <移動前>,<移動後> <配置座標><タイルカラー>")
        print("例: b1,b2 b3g (b1→b2へ移動、b3にグレータイル配置)")
        print("    c5,c4 (タイル配置なし)")

        p_idx = self.game.current_player - 1
        has_black = self.game.tile_counts[p_idx, 0] > 0
        has_gray = self.game.tile_counts[p_idx, 1] > 0
        print(
            f"持ちタイル: 黒(b)={self.game.tile_counts[p_idx, 0]}, グレー(g)={self.game.tile_counts[p_idx, 1]}"
        )

        while True:
            try:
                user_input = input("\n行動を入力: ").strip()

                if not user_input:
                    print("エラー: 入力が空です")
                    continue

                # スペースで分割: [移動部分, タイル部分(optional)]
                parts = user_input.split()

                if len(parts) == 0:
                    print("エラー: 入力が空です")
                    continue

                # 移動部分をパース
                move_part = parts[0]
                if "," not in move_part:
                    print(
                        "エラー: 移動は'<移動前>,<移動後>'の形式で入力してください (例: b1,b2)"
                    )
                    continue

                from_pos, to_pos = move_part.split(",")

                # 座標変換
                fx, fy = self.parse_position(from_pos)
                tx, ty = self.parse_position(to_pos)

                # 移動元の駒チェック
                if self.game.pieces[fy, fx] != self.game.current_player:
                    print(f"エラー: {from_pos}に自分の駒がありません")
                    continue

                # 移動先の有効性チェック
                valid_moves = self.game.get_valid_moves(fx, fy)
                if not valid_moves:
                    print(f"エラー: {from_pos}の駒は移動できません")
                    continue

                if (tx, ty) not in valid_moves:
                    # 移動可能な場所を新形式で表示
                    valid_pos_str = [
                        self.format_position(vx, vy) for vx, vy in valid_moves
                    ]
                    print(
                        f"エラー: {to_pos}には移動できません。移動可能: {', '.join(valid_pos_str)}"
                    )
                    continue

                # タイル配置部分をパース
                tile_type = 0
                tile_x, tile_y = 0, 0

                if len(parts) >= 2:
                    tile_part = parts[1]

                    if len(tile_part) < 3:
                        print(
                            "エラー: タイル配置は'<座標><色>'の形式で入力してください (例: b3g)"
                        )
                        continue

                    tile_pos = tile_part[:2]
                    tile_color = tile_part[2].lower()

                    if tile_color not in ["b", "g"]:
                        print(
                            "エラー: タイルの色はb(黒)またはg(グレー)を指定してください"
                        )
                        continue

                    # タイルの色を決定
                    if tile_color == "b":
                        if not has_black:
                            print("エラー: 黒タイルの持ち駒がありません")
                            continue
                        tile_type = TILE_BLACK
                    else:  # 'g'
                        if not has_gray:
                            print("エラー: グレータイルの持ち駒がありません")
                            continue
                        tile_type = TILE_GRAY

                    # タイル配置座標を変換
                    tile_x, tile_y = self.parse_position(tile_pos)

                    # タイル配置の有効性チェック
                    if self.game.tiles[tile_y, tile_x] != TILE_WHITE:
                        print(f"エラー: {tile_pos}は白タイルではありません")
                        continue

                    if tile_x == tx and tile_y == ty:
                        print("エラー: 移動先にはタイルを配置できません")
                        continue

                    if self.game.pieces[tile_y, tile_x] != 0 and not (
                        tile_x == fx and tile_y == fy
                    ):
                        print(f"エラー: {tile_pos}には駒があります（移動元以外）")
                        continue

                # アクションハッシュを生成
                move_idx = (fy * 5 + fx) * 25 + (ty * 5 + tx)

                if tile_type == 0:
                    tile_idx = 0
                elif tile_type == TILE_BLACK:
                    tile_idx = 1 + (tile_y * 5 + tile_x)
                else:  # TILE_GRAY
                    tile_idx = 26 + (tile_y * 5 + tile_x)

                action_hash = move_idx * 51 + tile_idx
                self.action_history.append(
                    (action_hash, self.game.current_player, None)
                )
                return action_hash

            except ValueError as e:
                print(f"エラー: {e}")
                continue
            except KeyboardInterrupt:
                raise

    def get_random_action(self):
        """ランダムな行動を取得"""
        import random

        valid_actions = self.game.get_all_legal_actions()
        if not valid_actions:
            logger.error("有効なアクションがありません")
            return None

        action = random.choice(valid_actions)

        # アクションを解釈して表示
        move_idx, tile_idx = decode_action(action)
        from_idx = move_idx // 25
        to_idx = move_idx % 25
        fx, fy = from_idx % 5, from_idx // 5
        tx, ty = to_idx % 5, to_idx // 5

        from_pos = self.format_position(fx, fy)
        to_pos = self.format_position(tx, ty)
        print(f"ランダムの行動: {from_pos},{to_pos}", end="")

        if tile_idx > 0:
            if tile_idx <= 25:
                tile_color = "b"
                tile_type_jp = "黒タイル"
                idx = tile_idx - 1
            else:
                tile_color = "g"
                tile_type_jp = "グレータイル"
                idx = tile_idx - 26

            tile_x, tile_y = idx % 5, idx // 5
            tile_pos = self.format_position(tile_x, tile_y)
            print(f" {tile_pos}{tile_color} ({tile_type_jp})", end="")

        print()

        self.action_history.append((action, self.game.current_player, None))
        return action

    def get_rule_based_action(self):
        """ルールベースの行動を取得（シンプルな戦略）

        戦略:
        1. 相手のゴールライン近くに駒があれば前進を妨害
        2. 自分の駒をゴールに向けて前進
        3. 可能なら黒タイルを相手の進路に配置
        """
        import random

        valid_actions = self.game.get_all_legal_actions()
        if not valid_actions:
            logger.error("有効なアクションがありません")
            return None

        best_action = None
        best_score = -1000

        current_player = self.game.current_player
        target_row = 0 if current_player == P1 else 4  # P1はy=0、P2はy=4を目指す
        opponent_target_row = 4 if current_player == P1 else 0

        for action in valid_actions:
            score = 0
            move_idx, tile_idx = decode_action(action)

            from_idx = move_idx // 25
            to_idx = move_idx % 25
            fx, fy = from_idx % 5, from_idx // 5
            tx, ty = to_idx % 5, to_idx // 5

            # ゴールに近づく移動を高評価
            if current_player == P1:
                progress = fy - ty  # y座標が減るほど良い
            else:
                progress = ty - fy  # y座標が増るほど良い
            score += progress * 10

            # ゴールラインに到達する手は最優先
            if ty == target_row:
                score += 100

            # 相手の駒を妨害する位置への移動
            opponent_pieces = self.game.pieces == OPPONENT[current_player]
            if opponent_pieces[opponent_target_row].any():
                # 相手のゴールライン近くに駒がある場合、妨害を優先
                score += 20

            # タイル配置のボーナス
            if tile_idx > 0:
                score += 5  # タイルを配置する手を少し優先

                if tile_idx <= 25:  # 黒タイル
                    score += 3  # 黒タイルは少し優先

            # ランダム性を加える
            score += random.random()

            if score > best_score:
                best_score = score
                best_action = action

        # アクションを解釈して表示
        move_idx, tile_idx = decode_action(best_action)
        from_idx = move_idx // 25
        to_idx = move_idx % 25
        fx, fy = from_idx % 5, from_idx // 5
        tx, ty = to_idx % 5, to_idx // 5

        from_pos = self.format_position(fx, fy)
        to_pos = self.format_position(tx, ty)
        print(f"ルールベースの行動: {from_pos},{to_pos}", end="")

        if tile_idx > 0:
            if tile_idx <= 25:
                tile_color = "b"
                tile_type_jp = "黒タイル"
                idx = tile_idx - 1
            else:
                tile_color = "g"
                tile_type_jp = "グレータイル"
                idx = tile_idx - 26

            tile_x, tile_y = idx % 5, idx // 5
            tile_pos = self.format_position(tile_x, tile_y)
            print(f" {tile_pos}{tile_color} ({tile_type_jp})", end="")

        print(f" (スコア: {best_score:.2f})")

        self.action_history.append((best_action, self.game.current_player, best_score))
        return best_action

    def get_ai_action(self):
        """AIの行動を取得"""
        print(f"\nAIの思考中... (プレイヤー{self.game.current_player})")

        # MCTS実行
        policy, values = self.mcts.search(self.game, self.num_simulations)

        if not policy:
            logger.error("AIが行動を選択できませんでした")
            return None

        # 最も訪問回数が多いアクションを選択
        action = max(policy, key=policy.get)
        value = values.get(action, 0.0)

        # アクションを解釈して表示
        move_idx, tile_idx = decode_action(action)

        from_idx = move_idx // 25
        to_idx = move_idx % 25
        fx, fy = from_idx % 5, from_idx // 5
        tx, ty = to_idx % 5, to_idx // 5

        from_pos = self.format_position(fx, fy)
        to_pos = self.format_position(tx, ty)
        print(f"AIの行動: {from_pos},{to_pos}", end="")

        if tile_idx > 0:
            if tile_idx <= 25:
                tile_color = "b"
                tile_type_jp = "黒タイル"
                idx = tile_idx - 1
            else:
                tile_color = "g"
                tile_type_jp = "グレータイル"
                idx = tile_idx - 26

            tile_x, tile_y = idx % 5, idx // 5
            tile_pos = self.format_position(tile_x, tile_y)
            print(f" {tile_pos}{tile_color} ({tile_type_jp})", end="")

        print(f" (評価値: {value:.3f})")

        self.action_history.append((action, self.game.current_player, value))
        return action

    def get_action_for_player(self, player):
        """指定されたプレイヤーの行動を取得"""
        player_type = self.player1_type if player == P1 else self.player2_type

        if player_type == "human":
            return self.get_human_action()
        elif player_type == "ai":
            return self.get_ai_action()
        elif player_type == "random":
            return self.get_random_action()
        elif player_type == "rule":
            return self.get_rule_based_action()
        else:
            logger.error(f"不明なプレイヤータイプ: {player_type}")
            return None

    def play(self):
        """ゲームをプレイ"""
        logger.info(
            f"ゲーム開始: プレイヤー1={self.player1_type}, プレイヤー2={self.player2_type}"
        )

        self.display_board()

        while not self.game.game_over:
            action = self.get_action_for_player(self.game.current_player)

            if action is None:
                logger.error("無効なアクションです")
                break

            # アクション実行
            done, winner = self.game.step(action)

            self.display_board()

            if done:
                break

        # 結果表示
        print("\n" + "=" * 50)
        print("ゲーム終了!")
        print("=" * 50)

        if self.game.winner == 0:
            print("引き分けです")
        elif self.game.winner == P1:
            print(f"🎉 プレイヤー1 ({self.player1_type}) の勝利です！")
        else:
            print(f"🎉 プレイヤー2 ({self.player2_type}) の勝利です！")

        print(f"総手数: {self.game.move_count}")
        print("=" * 50)
        print("行動履歴:")
        for idx, (action, player, value) in enumerate(self.action_history):
            move_idx, tile_idx = decode_action(action)
            from_idx = move_idx // 25
            to_idx = move_idx % 25
            fx, fy = from_idx % 5, from_idx // 5
            tx, ty = to_idx % 5, to_idx // 5

            from_pos = self.format_position(fx, fy)
            to_pos = self.format_position(tx, ty)
            action_str = (
                f"手数 {idx + 1}: プレイヤー{player} の行動: {from_pos},{to_pos}"
            )

            if tile_idx > 0:
                if tile_idx <= 25:
                    tile_color = "b"
                    tile_type_jp = "黒タイル"
                    idx_tile = tile_idx - 1
                else:
                    tile_color = "g"
                    tile_type_jp = "グレータイル"
                    idx_tile = tile_idx - 26

                tile_x, tile_y = idx_tile % 5, idx_tile // 5
                tile_pos = self.format_position(tile_x, tile_y)
                action_str += f" {tile_pos}{tile_color} ({tile_type_jp})"

            if value is not None:
                action_str += f" | 評価値: {value:.3f}"

            print(action_str)


def main():
    parser = argparse.ArgumentParser(description="学習済みモデルと対戦")
    parser.add_argument(
        "--model",
        type=str,
        default="contrast_model_final.pth",
        help="学習済みモデルのパス",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTSのシミュレーション回数 (デフォルト: 100)",
    )
    parser.add_argument(
        "--player1",
        type=str,
        choices=["human", "ai", "random", "rule"],
        default="human",
        help="プレイヤー1のタイプ (human/ai/random/rule, デフォルト: human)",
    )
    parser.add_argument(
        "--player2",
        type=str,
        choices=["human", "ai", "random", "rule"],
        default="ai",
        help="プレイヤー2のタイプ (human/ai/random/rule, デフォルト: ai)",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logger()

    # ゲーム開始
    game = HumanVsAI(
        model_path=args.model,
        num_simulations=args.simulations,
        player1_type=args.player1,
        player2_type=args.player2,
    )

    try:
        game.play()
    except KeyboardInterrupt:
        print("\n\nゲームを中断しました")
        logger.info("Game interrupted by user")


if __name__ == "__main__":
    main()
