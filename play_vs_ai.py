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
    def __init__(self, model_path, num_simulations=50, human_player=P1):
        """
        Args:
            model_path: 学習済みモデルのパス
            num_simulations: MCTSのシミュレーション回数
            human_player: 人間が操作するプレイヤー (P1 or P2)
        """
        self.human_player = human_player
        self.ai_player = OPPONENT[human_player]
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

        # 列番号
        print("   ", end="")
        for x in range(5):
            print(f" {x} ", end="")
        print()

        for y in range(5):
            print(f" {y} ", end="")
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

    def get_human_action(self):
        """人間から行動を入力"""
        print(f"\nあなたの番です (プレイヤー{self.human_player})")

        # 移動する駒を選択
        while True:
            try:
                from_input = input("移動する駒の座標を入力 (例: 2,4): ").strip()
                fx, fy = map(int, from_input.split(","))

                if not (0 <= fx < 5 and 0 <= fy < 5):
                    print("エラー: 座標は0-4の範囲で入力してください")
                    continue

                if self.game.pieces[fy, fx] != self.game.current_player:
                    print("エラー: 自分の駒を選択してください")
                    continue

                # 有効な移動先を表示
                valid_moves = self.game.get_valid_moves(fx, fy)
                if not valid_moves:
                    print("エラー: この駒は移動できません")
                    continue

                print(f"移動可能な場所: {valid_moves}")
                break

            except (ValueError, KeyboardInterrupt):
                print("エラー: 正しい形式で入力してください (例: 2,4)")
                continue

        # 移動先を選択
        while True:
            try:
                to_input = input("移動先の座標を入力 (例: 2,3): ").strip()
                tx, ty = map(int, to_input.split(","))

                if (tx, ty) not in valid_moves:
                    print(f"エラー: 移動できない場所です。有効な移動先: {valid_moves}")
                    continue

                break

            except (ValueError, KeyboardInterrupt):
                print("エラー: 正しい形式で入力してください")
                continue

        # タイル配置を選択
        p_idx = self.game.current_player - 1
        has_black = self.game.tile_counts[p_idx, 0] > 0
        has_gray = self.game.tile_counts[p_idx, 1] > 0

        tile_type = 0  # デフォルトはタイルなし
        tile_x, tile_y = 0, 0

        if has_black or has_gray:
            while True:
                try:
                    tile_choice = input(
                        f"タイルを配置しますか? (0:なし, 1:黒タイル[残{self.game.tile_counts[p_idx, 0]}], 2:グレータイル[残{self.game.tile_counts[p_idx, 1]}]): "
                    ).strip()

                    if tile_choice == "0":
                        break
                    elif tile_choice == "1" and has_black:
                        tile_type = TILE_BLACK
                        break
                    elif tile_choice == "2" and has_gray:
                        tile_type = TILE_GRAY
                        break
                    else:
                        print("エラー: 無効な選択です")
                        continue

                except (ValueError, KeyboardInterrupt):
                    print("エラー: 0, 1, 2 のいずれかを入力してください")
                    continue

            if tile_type != 0:
                while True:
                    try:
                        tile_input = input(
                            "タイル配置先の座標を入力 (例: 2,2): "
                        ).strip()
                        tile_x, tile_y = map(int, tile_input.split(","))

                        if not (0 <= tile_x < 5 and 0 <= tile_y < 5):
                            print("エラー: 座標は0-4の範囲で入力してください")
                            continue

                        # 白タイルでない場所には配置できない
                        if self.game.tiles[tile_y, tile_x] != TILE_WHITE:
                            print("エラー: 白タイル以外の場所には配置できません")
                            continue

                        # 移動先には配置できない
                        if tile_x == tx and tile_y == ty:
                            print("エラー: 移動先には配置できません")
                            continue

                        # 移動元以外にコマがある場所には配置できない
                        if self.game.pieces[tile_y, tile_x] != 0 and not (
                            tile_x == fx and tile_y == fy
                        ):
                            print("エラー: コマがある場所には配置できません")
                            continue

                        break

                    except (ValueError, KeyboardInterrupt):
                        print("エラー: 正しい形式で入力してください")
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
        self.action_history.append((action_hash, self.game.current_player, None))
        return action_hash

    def get_random_action(self):
        """ランダムな行動を取得（デバッグ用）"""
        import random

        valid_actions = self.game.get_all_legal_actions()
        if not valid_actions:
            logger.error("有効なアクションがありません")
            return None

        action = random.choice(valid_actions)
        logger.debug(f"ランダムに選択された行動: {action}")
        self.action_history.append((action, self.game.current_player, None))
        return action

    def get_ai_action(self):
        """AIの行動を取得"""
        print(f"\nAIの思考中... (プレイヤー{self.ai_player})")

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

        print(f"AIの行動: ({fx},{fy}) → ({tx},{ty})", end="")

        if tile_idx > 0:
            if tile_idx <= 25:
                tile_type = "黒タイル"
                idx = tile_idx - 1
            else:
                tile_type = "グレータイル"
                idx = tile_idx - 26

            tile_x, tile_y = idx % 5, idx // 5
            print(f" + {tile_type}を({tile_x},{tile_y})に配置", end="")

        print(f" (評価値: {value:.3f})")

        self.action_history.append((action, self.game.current_player, value))
        return action

    def play(self):
        """ゲームをプレイ"""
        logger.info(
            f"ゲーム開始: 人間=プレイヤー{self.human_player}, AI=プレイヤー{self.ai_player}"
        )

        self.display_board()

        while not self.game.game_over:
            if self.game.current_player == self.human_player:
                # 人間のターン
                # action = self.get_random_action()
                action = self.get_human_action()
                # action = self.get_ai_action()
            else:
                # AIのターン
                action = self.get_ai_action()

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
        elif self.game.winner == self.human_player:
            print("🎉 あなたの勝利です！")
        else:
            print("😢 AIの勝利です")

        print(f"総手数: {self.game.move_count}")
        print("=" * 50)
        print("行動履歴:")
        for idx, (action, player, value) in enumerate(self.action_history):
            move_idx, tile_idx = decode_action(action)
            from_idx = move_idx // 25
            to_idx = move_idx % 25
            fx, fy = from_idx % 5, from_idx // 5
            tx, ty = to_idx % 5, to_idx // 5

            action_str = (
                f"手数 {idx + 1}: プレイヤー{player} の行動: ({fx},{fy}) → ({tx},{ty})"
            )
            if tile_idx > 0:
                if tile_idx <= 25:
                    tile_type = "黒タイル"
                    idx_tile = tile_idx - 1
                else:
                    tile_type = "グレータイル"
                    idx_tile = tile_idx - 26

                tile_x, tile_y = idx_tile % 5, idx_tile // 5
                action_str += f" + {tile_type}を({tile_x},{tile_y})に配置"

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
        "--player",
        type=int,
        choices=[1, 2],
        default=1,
        help="人間が操作するプレイヤー (1 or 2, デフォルト: 1)",
    )

    args = parser.parse_args()

    # ロギング設定
    setup_logger()

    # ゲーム開始
    game = HumanVsAI(
        model_path=args.model,
        num_simulations=args.simulations,
        human_player=args.player,
    )

    try:
        game.play()
    except KeyboardInterrupt:
        print("\n\nゲームを中断しました")
        logger.info("Game interrupted by user")


if __name__ == "__main__":
    main()
