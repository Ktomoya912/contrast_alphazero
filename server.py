import copy
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from contrast_game import P1, P2, TILE_BLACK, TILE_GRAY, ContrastGame
from players.alpha_zero import AlphaZeroPlayer  # 作成済みのクラスを活用

app = Flask(__name__)
base_dir = Path(__file__).parent
# MODEL_PATH = "models/model_step_262700_elo_1427.pth"  # contrast_model_final.pth
MODEL_PATH = "models/contrast_model_final.pth"  # contrast_model_final.pth
SIMULATIONS = 200


class GameManager:
    """ゲームの進行と状態を一元管理するクラス"""

    def __init__(self):
        self.game = ContrastGame()
        self.ai_player = AlphaZeroPlayer(
            player_id=P2, model_path=MODEL_PATH, num_simulations=SIMULATIONS
        )
        self.last_ai_value = 0.0
        self.history = []
        self.initialize()

    def initialize(self):
        """ゲームとAIの初期化"""
        self.game = ContrastGame()

        # すでに作成済みのAlphaZeroPlayerクラスを使用
        # モデルのロードやMCTSの準備はすべてこのクラスにお任せ
        self.ai_player = AlphaZeroPlayer(
            player_id=P2, model_path=MODEL_PATH, num_simulations=SIMULATIONS
        )
        print("Game and AI initialized.")

    def reset(self, human_player_id=P1):
        """ゲームのリセット（先手後手の設定）"""
        self.game = ContrastGame()
        self.last_ai_value = 0.0
        self.history = []  # ★追加: リセット時は履歴もクリア

        ai_id = P2 if human_player_id == P1 else P1
        self.ai_player.player_id = ai_id

        print(f"Game reset. Human: P{human_player_id}, AI: P{ai_id}")

        if ai_id == P1:
            self.process_ai_move()

    def get_state(self):
        return {
            "pieces": self.game.pieces.tolist(),
            "tiles": self.game.tiles.tolist(),
            "current_player": self.game.current_player,
            "game_over": self.game.game_over,
            "winner": self.game.winner,
            "move_count": self.game.move_count,
            "tile_counts": self.game.tile_counts.tolist(),
            "ai_value": self.last_ai_value,
            "ai_player_id": self.ai_player.player_id,  # ★追加: AIのプレイヤーID
        }

    def process_human_move(self, data):
        try:
            self.history.append(copy.deepcopy(self.game))
            # 座標データの取得とインデックス計算
            fx, fy = data["from"]
            tx, ty = data["to"]
            move_idx = (fy * 5 + fx) * 25 + (ty * 5 + tx)

            # タイル情報の処理
            tile_idx = 0
            if data.get("tile"):
                t_type = data["tile"]["type"]
                t_x, t_y = data["tile"]["x"], data["tile"]["y"]
                idx = t_y * 5 + t_x
                if t_type == TILE_BLACK:
                    tile_idx = 1 + idx
                elif t_type == TILE_GRAY:
                    tile_idx = 26 + idx

            action_hash = move_idx * 51 + tile_idx

            # 合法手チェック
            legal_actions = self.game.get_all_legal_actions()
            if action_hash not in legal_actions:
                # 失敗した場合は履歴から削除（保存した意味がないため）
                self.history.pop()
                return False, "Illegal move"

            self.game.step(action_hash)
            return True, "OK"

        except Exception as e:
            if self.history:
                self.history.pop()
            return False, str(e)

    def process_ai_move(self):
        if (
            not self.game.game_over
            and self.game.current_player == self.ai_player.player_id
        ):
            action_data = self.ai_player.get_action(self.game)

            if action_data is not None:
                if isinstance(action_data, tuple):
                    action, value = action_data
                    self.last_ai_value = value  # ★値を保存
                else:
                    action = action_data

                self.game.step(action)
                return True
        return False

    def undo(self):
        if not self.history:
            return False

        # 履歴スタックの最後（人間が動く直前の状態）を取り出して復元
        self.game = self.history.pop()

        # 評価値などの付帯情報も必要なら戻すべきだが、
        # ゲーム進行には影響しないのでそのままでOK
        return True


gm = GameManager()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    return jsonify(gm.get_state())


@app.route("/api/move", methods=["POST"])
def make_move():
    data = request.json

    # 1. 人間の移動
    if data and data.get("is_human"):
        # 手番チェック
        if gm.game.current_player != gm.ai_player.player_id:  # AIの番でなければ人間
            success, msg = gm.process_human_move(data)
            if not success:
                return jsonify({"error": msg}), 400
        else:
            return jsonify({"error": "Not your turn"}), 400

    # 2. AIの移動 (人間が動いた後、またはAIの手番なら)
    if not gm.game.game_over:
        gm.process_ai_move()

    return jsonify({"status": "ok"})


@app.route("/api/reset", methods=["POST"])
def reset_game():
    data = request.json or {}
    # リクエストから人間の手番IDを取得 (デフォルトはP1)
    human_player = data.get("human_player", P1)
    gm.reset(human_player_id=human_player)
    return jsonify({"status": "ok"})


@app.route("/api/undo", methods=["POST"])
def undo_move():
    if gm.undo():
        return jsonify({"status": "ok"})
    else:
        return jsonify({"error": "Cannot undo"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
