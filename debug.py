# debug.py
import torch

from contrast_game import ContrastGame
from logger import get_logger, setup_logger
from mcts import MCTS
from model import ContrastDualPolicyNet

logger = get_logger(__name__)


def debug_selfplay():
    logger.info("デバッグ開始: モデル初期化")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastDualPolicyNet().to(device)
    model.eval()

    game = ContrastGame()
    # デバッグ用にシミュレーション回数を減らす
    mcts = MCTS(network=model, device=device, alpha=0.3)
    num_simulations = 10

    logger.info("ゲームループ開始")
    step = 0
    max_steps = 200  # 無限ループ防止用の強制終了ライン

    while not game.game_over and step < max_steps:
        # MCTS実行
        mcts_policy, _ = mcts.search(game, num_simulations)

        # 行動選択
        action = max(mcts_policy, key=lambda x: mcts_policy[x])

        # 実行
        game.step(action)
        step += 1

        # 状況表示
        if step % 10 == 0:
            logger.debug(
                f"Step {step}: Player {game.current_player}, Move Count {game.move_count}"
            )
            # game.board.display() # 必要なら盤面表示メソッドを呼ぶ

    logger.info(f"ゲーム終了: Winner = {game.winner}, Total Steps = {step}")


if __name__ == "__main__":
    # エントリーポイントでロギングを初期化
    setup_logger()
    debug_selfplay()
