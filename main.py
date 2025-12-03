import datetime
import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import ray
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from contrast_game import ContrastGame
from logger import get_logger, setup_logger
from mcts import MCTS
from model import ContrastDualPolicyNet

logger = get_logger(__name__)

# 定数定義
NUM_CPUS = 4  # 環境に合わせて変更
NUM_GPUS = 1 if torch.cuda.is_available() else 0
BATCH_SIZE = 128
BUFFER_SIZE = 40000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_STEPS = 150  # ★追加: これ以上長引いたら強制終了


@dataclass
class Sample:
    state: np.ndarray  # (90, 5, 5)
    mcts_policy: dict  # {action_hash: prob}
    player: int  # 1 or 2
    reward: float = 0.0  # 後で埋める


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add_record(self, record):
        self.buffer.extend(record)

    def __len__(self):
        return len(self.buffer)

    def get_minibatch(self, batch_size):
        """
        バッチを取り出し、PyTorchのTensor形式（Dual Head用ターゲット）に変換して返す
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))

        states = []
        move_targets = []
        tile_targets = []
        value_targets = []

        for sample in batch:
            states.append(sample.state)
            value_targets.append(sample.reward)

            # --- MCTSのSparseなPolicyをDual HeadのDenseなTargetに変換 ---
            # Move Target: (625,), Tile Target: (51,)
            m_target = np.zeros(625, dtype=np.float32)
            t_target = np.zeros(51, dtype=np.float32)

            for action_hash, prob in sample.mcts_policy.items():
                # Hash -> (Move, Tile)
                m_idx = action_hash // 51
                t_idx = action_hash % 51

                # 確率を加算 (周辺化)
                m_target[m_idx] += prob
                t_target[t_idx] += prob

            move_targets.append(m_target)
            tile_targets.append(t_target)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(move_targets), dtype=torch.float32),
            torch.tensor(np.array(tile_targets), dtype=torch.float32),
            torch.tensor(np.array(value_targets), dtype=torch.float32).unsqueeze(1),
        )


@ray.remote(num_cpus=1, num_gpus=0)
def selfplay(weights, num_mcts_simulations, dirichlet_alpha=0.3):
    """
    Ray Worker: Self-playを実行してデータを収集
    """
    # モデルの初期化 (CPU)
    model = ContrastDualPolicyNet()
    model.load_state_dict(weights)
    model.eval()

    # ゲームとMCTSの初期化
    game = ContrastGame()
    mcts = MCTS(network=model, device=torch.device("cpu"), alpha=dirichlet_alpha)

    record = []
    done = False
    step = 0

    while not done:
        # MCTS実行
        # mcts_policy: {action_hash: prob}
        mcts_policy = mcts.search(game, num_mcts_simulations)

        # 強制終了判定
        if step >= MAX_STEPS:
            done = True
            winner = 0  # 引き分け扱い
            break

        # 温度パラメータの制御
        # 序盤はランダム性を残し、中盤以降はGreedyに
        actions = list(mcts_policy.keys())
        probs = list(mcts_policy.values())

        if step < 10:
            # 温度 = 1 (確率に従って選択)
            action = np.random.choice(actions, p=probs)
        else:
            # 温度 = 0 (最大確率の手を選択)
            # dictの値が最大のものを選ぶ
            action = max(mcts_policy, key=mcts_policy.get)

        # 記録 (現在の状態、MCTSの分布、手番)
        # encode_stateは (90, 5, 5) を返す
        record.append(
            Sample(
                state=game.encode_state(),
                mcts_policy=mcts_policy,
                player=game.current_player,
            )
        )

        # 実行
        done, winner = game.step(action)
        step += 1

        if step % 10 == 0:
            logger.debug(f"Worker Progress: step {step}")

    # 報酬の割り当て (Winner視点)
    # game.winner: P1(1) or P2(2) or Draw(0)
    for sample in record:
        if winner == 0:
            sample.reward = 0.0
        else:
            # 自分の手番で勝ったなら+1, 負けたなら-1
            sample.reward = 1.0 if sample.player == winner else -1.0

    return record


def main(n_parallel_selfplay=10, num_mcts_simulations=50):
    # Ray初期化
    ray.init(ignore_reinit_error=True)

    # デバイス設定 (TrainerはGPU推奨)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # モデルとオプティマイザ
    network = ContrastDualPolicyNet().to(device)
    optimizer = optim.Adam(
        network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 初期ウェイトをRay Object Storeへ
    # CPUに落としてから送る
    current_weights_ref = ray.put(network.to("cpu").state_dict())
    network.to(device)  # Trainer用にGPUに戻す

    replay = ReplayBuffer(buffer_size=BUFFER_SIZE)

    # Workerの起動
    work_in_progresses = [
        selfplay.remote(current_weights_ref, num_mcts_simulations)
        for _ in range(n_parallel_selfplay)
    ]

    # トレーニングループ
    total_steps = 0
    max_steps = 10000

    # プログレスバー
    pbar = tqdm(total=max_steps, desc="Training Steps")

    while total_steps < max_steps:
        # 1. データ収集フェーズ (非同期実行の待ち受け)
        # 一定数(例: 10ゲーム分)集まるまで待つ、あるいは完了した順に処理

        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)

        # 完了したタスクから結果を取得
        game_records = ray.get(finished[0])
        replay.add_record(game_records)

        # Workerを再起動 (最新のウェイトを渡す)
        work_in_progresses.append(
            selfplay.remote(current_weights_ref, num_mcts_simulations)
        )

        # 2. 学習フェーズ
        # バッファがある程度たまったら学習開始
        if len(replay) > BATCH_SIZE:
            # データ取得 (GPU転送込み)
            states, m_targets, t_targets, v_targets = replay.get_minibatch(BATCH_SIZE)
            states = states.to(device)
            m_targets = m_targets.to(device)
            t_targets = t_targets.to(device)
            v_targets = v_targets.to(device)

            # 勾配リセット
            optimizer.zero_grad()

            # 推論
            m_logits, t_logits, v_pred = network(states)

            # 損失計算
            # Value Loss: MSE
            value_loss = F.mse_loss(v_pred, v_targets)

            # Policy Loss: Cross Entropy
            # PyTorchのCrossEntropyLossはTargetがクラスインデックスであることを期待するが、
            # AlphaZeroはソフトターゲット(確率分布)を使うため、
            # LogSoftmax + Sum(target * log_prob) の形式で計算する (KL Divergence相当)

            m_log_probs = F.log_softmax(m_logits, dim=1)
            t_log_probs = F.log_softmax(t_logits, dim=1)

            move_loss = -torch.mean(torch.sum(m_targets * m_log_probs, dim=1))
            tile_loss = -torch.mean(torch.sum(t_targets * t_log_probs, dim=1))

            loss = value_loss + move_loss + tile_loss

            # バックプロパゲーション
            loss.backward()
            optimizer.step()

            # 3. ウェイトの更新
            # 一定ステップごとにRay上のウェイトを更新
            if total_steps % 50 == 0:
                current_weights_ref = ray.put(network.to("cpu").state_dict())
                network.to(device)

                # ログ出力
                tqdm.write(
                    f"Step {total_steps}: Loss={loss.item():.4f} (V={value_loss.item():.4f}, M={move_loss.item():.4f}, T={tile_loss.item():.4f})"
                )

            total_steps += 1
            pbar.update(1)

    # 終了処理
    torch.save(network.state_dict(), "contrast_model_final.pth")
    logger.info("Training finished. Model saved.")


if __name__ == "__main__":
    # エントリーポイントでロギングを初期化
    setup_logger(
        log_file=f"logs/training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    main()
