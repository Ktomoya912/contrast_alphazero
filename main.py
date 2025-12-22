import datetime
import logging
import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ray
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (
    evaluation_config,
    mcts_config,
    network_config,
    path_config,
    training_config,
)
from contrast_game import P2, ContrastGame, flip_action
from elo_evaluator import EloEvaluator
from logger import get_logger, setup_logger
from mcts import MCTS
from model import ContrastDualPolicyNet, loss_function

logger = get_logger(__name__)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class Sample:
    state: np.ndarray  # (90, 5, 5)
    mcts_policy: dict[int, float]  # {action_hash: prob}
    player: int  # 1 or 2
    reward: float = 0.0  # 後で埋める


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer: deque[Sample] = deque(maxlen=buffer_size)

    def add_record(self, record):
        self.buffer.extend(record)

    def __len__(self):
        return len(self.buffer)

    def get_minibatch(self, batch_size):
        """
        バッチを取り出し、PyTorchのTensor形式（Dual Head用ターゲット）に変換して返す
        """
        batch: list[Sample] = random.sample(
            self.buffer, min(len(self.buffer), batch_size)
        )

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

            should_flip = sample.player == P2

            for action_hash, prob in sample.mcts_policy.items():
                # ネットワークが学習すべきは「反転された盤面に対する、反転された行動」
                if should_flip:
                    target_hash = flip_action(action_hash)
                else:
                    target_hash = action_hash

                m_idx = target_hash // 51
                t_idx = target_hash % 51

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
def selfplay(weights, num_mcts_simulations, index, dirichlet_alpha=0.3):
    """
    Ray Worker: Self-playを実行してデータを収集

    Args:
        weights: モデルの重み
        num_mcts_simulations: MCTSのシミュレーション回数
        dirichlet_alpha: ディリクレノイズのパラメータ

    Returns:
        ゲームのデータサンプルリスト
    """
    pid = os.getpid()
    log_filename = Path(__file__).parent / f"logs/proc/worker_selfplay_{index}.log"
    setup_logger(log_level=logging.INFO, log_file=log_filename, show_console=False)
    logger = logging.getLogger(__name__)
    logger.info(f"Worker process started. PID: {pid}")
    torch.set_num_threads(1)
    # デフォルト値の設定 (config.pyから)
    if dirichlet_alpha is None:
        dirichlet_alpha = mcts_config.DIRICHLET_ALPHA

    # モデルの初期化 (CPU)
    model = ContrastDualPolicyNet()
    model.load_state_dict(weights)
    model.eval()

    # ゲームとMCTSの初期化
    game = ContrastGame()
    mcts = MCTS(
        network=model,
        device=torch.device("cpu"),
        alpha=dirichlet_alpha,
        c_puct=mcts_config.C_PUCT,
        epsilon=mcts_config.DIRICHLET_EPSILON,
    )

    record: list[Sample] = []
    done = False
    step = 0

    while not done:
        # MCTS実行
        # mcts_policy: {action_hash: prob}
        mcts_policy, values = mcts.search(game, num_mcts_simulations)
        # 強制終了判定
        if step >= training_config.MAX_STEPS:
            done = True
            winner = 0  # 引き分け扱い
            break

        # 温度パラメータの制御 (config.pyから)
        # 序盤はランダム性を残し、中盤以降はGreedyに
        actions = list(mcts_policy.keys())
        probs = list(mcts_policy.values())

        if step < mcts_config.TEMPERATURE_THRESHOLD:
            # 温度 = 1 (確率に従って選択) - 序盤は多様な手を試す
            action = np.random.choice(actions, p=probs)
        else:
            # 温度 = 0 (最大確率の手を選択) - 中盤以降は最善手
            action = max(mcts_policy, key=lambda x: mcts_policy[x])
        action_prob = mcts_policy[action]
        action_value = values.get(action, 0.0)  # valuesはMCTSのQ値
        logger.debug(
            f"P{game.current_player} Step{step}: action={action}, "
            f"prob={action_prob:.4f}, value={action_value:.4f}"
        )
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

    # 報酬の割り当て (Winner視点)
    # game.winner: P1(1) or P2(2) or Draw(0)
    for sample in record:
        if winner == 0:
            # 引き分けにはペナルティを与える（決着をつけることを促す）
            sample.reward = -0.1
        else:
            # 自分の手番で勝ったなら+1, 負けたなら-1
            sample.reward = 1.0 if sample.player == winner else -1.0

    if winner == 0:
        result_str = f"Selfplay result: DRAW (Step {step})"
    else:
        result_str = f"Selfplay result: WIN P{winner} (Step {step})"
    logger.info(f"{result_str}, MCTS sims: {num_mcts_simulations}")

    return record


def main(n_parallel_selfplay=2, num_mcts_simulations=50):
    """メイン学習ループ

    Args:
        n_parallel_selfplay: 並列化するSelf-playの数
        num_mcts_simulations: MCTSのシミュレーション回数
    """
    # ワーカー数の検証
    if n_parallel_selfplay < 1:
        logger.warning(
            f"Invalid n_parallel_selfplay={n_parallel_selfplay}, setting to 1"
        )
        n_parallel_selfplay = 1

    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        configure_logging=True,
    )
    device = training_config.DEVICE
    logger.info(f"Training started on {device}")
    logger.info(
        f"Config: Parallel workers={n_parallel_selfplay}, "
        f"MCTS sims={num_mcts_simulations}, "
        f"{training_config}"
    )

    network = ContrastDualPolicyNet().to(device)
    optimizer = optim.Adam(
        network.parameters(),
        lr=training_config.LEARNING_RATE,
        weight_decay=training_config.WEIGHT_DECAY,
    )
    # 学習率スケジューラ (config.pyから設定を取得)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=training_config.LR_STEP_SIZE,
        gamma=training_config.LR_GAMMA,
    )

    # ★変更: 評価用ActorをCPUで起動
    elo_evaluator = EloEvaluator.remote(device_str="cpu", baseline_elo=1000)
    evaluation_future = None  # 評価タスクのハンドル

    current_weights_ref = ray.put(network.to("cpu").state_dict())
    network.to(device)

    replay = ReplayBuffer(buffer_size=training_config.BUFFER_SIZE)
    work_in_progresses = [
        selfplay.remote(current_weights_ref, num_mcts_simulations, number)
        for number in range(n_parallel_selfplay)
    ]
    exp_name = (
        f"{timestamp}_"
        f"res{network_config.NUM_RES_BLOCKS}_"
        f"filt{network_config.NUM_FILTERS}_"
        f"sim{num_mcts_simulations}_"
        f"bs{training_config.BATCH_SIZE}_"
        f"lr{training_config.LEARNING_RATE}"
    )

    # TensorBoard Writerの初期化
    log_dir = f"{path_config.LOGS_DIR}/tensorboard/{exp_name}"
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard log dir: {log_dir}")

    total_steps = 0
    pbar = tqdm(total=training_config.MAX_EPOCH, desc="Training")
    number = -1
    while total_steps < training_config.MAX_EPOCH:
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        replay.add_record(ray.get(finished[0]))
        number = (number + 1) % n_parallel_selfplay
        work_in_progresses.append(
            selfplay.remote(current_weights_ref, num_mcts_simulations, number)
        )

        if len(replay) > training_config.BATCH_SIZE:
            states, m_targets, t_targets, v_targets = replay.get_minibatch(
                training_config.BATCH_SIZE
            )
            states = states.to(device)
            m_targets = m_targets.to(device)
            t_targets = t_targets.to(device)
            v_targets = v_targets.to(device)
            # 勾配リセット
            optimizer.zero_grad()
            # 推論
            m_logits, t_logits, v_pred = network(states)
            loss, (v_loss, m_loss, t_loss) = loss_function(
                m_logits, t_logits, v_pred, m_targets, t_targets, v_targets
            )
            # バックプロパゲーション
            loss.backward()
            optimizer.step()
            scheduler.step()
            # 3. ウエイトの更新
            # 一定ステップごとにRay上のウエイトを更新 (config.pyから間隔を取得)
            if total_steps % training_config.LOG_INTERVAL == 0:
                writer.add_scalar("Loss/Total", loss.item(), total_steps)
                writer.add_scalar("Loss/Value", v_loss, total_steps)
                writer.add_scalar("Loss/Move", m_loss, total_steps)
                writer.add_scalar("Loss/Tile", t_loss, total_steps)
                writer.add_scalar(
                    "Training/LR", scheduler.get_last_lr()[0], total_steps
                )
                current_weights_ref = ray.put(network.to("cpu").state_dict())
                network.to(device)
                current_lr = scheduler.get_last_lr()[0]

                # 詳細なメトリクスログ
                log_msg = (
                    f"Step {total_steps}: Loss={loss.item():.4f} "
                    f"(Value={v_loss:.4f}, "
                    f"Move={m_loss:.4f}, "
                    f"Tile={t_loss:.4f}) | "
                    f"LR={current_lr:.6f} | "
                    f"Buffer={len(replay)}/{training_config.BUFFER_SIZE}"
                )
                logger.info(log_msg)

            # ★変更: 非同期評価ロジック
            # 前回の評価が終わっているかチェック
            if evaluation_future is not None:
                # timeout=0で即座に確認（終わってなければ空リストが返る）
                ready, _ = ray.wait([evaluation_future], timeout=0)
                if ready:
                    try:
                        elo, win_rate = ray.get(evaluation_future)
                        writer.add_scalar("Evaluation/ELO", elo, total_steps)
                        writer.add_scalar("Evaluation/WinRate", win_rate, total_steps)
                        tqdm.write(
                            f"Evaluation Result: ELO={elo:.1f}, WinRate={win_rate:.1f}%"
                        )
                    except Exception as e:
                        logger.error(f"Evaluation Error: {e}")
                    evaluation_future = None  # タスク完了、リセット

            # 新しい評価を投げる (インターバル経過 かつ 前の評価が終わっている場合)
            if (
                total_steps > 0
                and total_steps % evaluation_config.EVAL_INTERVAL == 0
                and evaluation_future is None
            ):
                tqdm.write(f"Step {total_steps}: Starting async evaluation...")
                evaluation_future = elo_evaluator.evaluate.remote(
                    current_weights_ref,
                    total_steps,
                    evaluation_config.EVAL_NUM_GAMES,
                    evaluation_config.EVAL_MCTS_SIMS,
                )

            total_steps += 1
            pbar.update(1)

    save_path = path_config.FINAL_MODEL_PATH
    torch.save(network.state_dict(), save_path)
    logger.info(f"Training finished. Final model saved to {save_path}")
    logger.info(f"Total training steps: {total_steps}")


if __name__ == "__main__":
    # エントリーポイントでロギングを初期化

    log_file = f"{path_config.LOGS_DIR}/training_{timestamp}.log"
    setup_logger(
        log_level=logging.INFO,
        log_file=log_file,
    )

    logger.info("=" * 60)
    logger.info("Contrast AlphaZero Training Started")
    logger.info(f"Timestamp: {timestamp}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 60)

    # 並列selfplayワーカー数の決定（メモリ制約を考慮）
    # メモリ不足を防ぐため、CPUの半分程度に制限
    n_workers = max(1, min(training_config.NUM_CPUS // 2, 4))
    logger.info(f"Starting training with {n_workers} parallel selfplay workers")
    main(n_parallel_selfplay=n_workers)
