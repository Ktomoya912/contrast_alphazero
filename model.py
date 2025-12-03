import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import get_logger

logger = get_logger(__name__)


class ResidualBlock(nn.Module):
    """
    ResNetブロック (共通)
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class ContrastDualPolicyNet(nn.Module):
    """
    移動(Move)とタイル配置(Tile)を同時に出力するマルチヘッドモデル
    """

    def __init__(
        self, input_channels=90, board_size=5, num_res_blocks=4, num_filters=64
    ):
        super(ContrastDualPolicyNet, self).__init__()
        self.board_size = board_size

        # --- Action Sizes ---
        # Move: From(25) * To(25) = 625
        self.move_size = board_size**2 * board_size**2
        # Tile: Pass(1) + Black(25) + Gray(25) = 51
        self.tile_size = 1 + (board_size**2) * 2

        # --- 1. Initial Block ---
        self.conv_input = nn.Conv2d(
            input_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)

        # --- 2. Residual Tower ---
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # --- 3. Move Policy Head ---
        self.move_conv = nn.Conv2d(num_filters, 32, kernel_size=1, stride=1, bias=False)
        self.move_bn = nn.BatchNorm2d(32)
        self.move_fc = nn.Linear(32 * board_size * board_size, self.move_size)

        # --- 4. Tile Policy Head ---
        self.tile_conv = nn.Conv2d(num_filters, 16, kernel_size=1, stride=1, bias=False)
        self.tile_bn = nn.BatchNorm2d(16)
        self.tile_fc = nn.Linear(16 * board_size * board_size, self.tile_size)

        # --- 5. Value Head ---
        self.value_conv = nn.Conv2d(num_filters, 4, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * board_size * board_size, 32)
        self.value_fc2 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Returns:
            move_logits: (Batch, 625)
            tile_logits: (Batch, 51)
            value:       (Batch, 1)
        """
        # Backbone
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.relu(x)

        for block in self.res_blocks:
            x = block(x)

        # Move Head
        m = self.move_conv(x)
        m = self.move_bn(m)
        m = self.relu(m)
        m = m.view(m.size(0), -1)
        move_logits = self.move_fc(m)

        # Tile Head
        t = self.tile_conv(x)
        t = self.tile_bn(t)
        t = self.relu(t)
        t = t.view(t.size(0), -1)
        tile_logits = self.tile_fc(t)

        # Value Head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = self.relu(v)
        v = self.value_fc2(v)
        value = torch.tanh(v)

        return move_logits, tile_logits, value


# --- 損失関数の定義例 ---
def loss_function(
    move_logits, tile_logits, value_pred, move_targets, tile_targets, value_targets
):
    """
    AlphaZero Loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
    ここではMoveとTileのCrossEntropyを合算します。
    """
    # 1. Value Loss (MSE)
    value_loss = F.mse_loss(value_pred.view(-1), value_targets.view(-1))

    # 2. Move Policy Loss (CrossEntropy)
    move_loss = F.cross_entropy(move_logits, move_targets)

    # 3. Tile Policy Loss (CrossEntropy)
    tile_loss = F.cross_entropy(tile_logits, tile_targets)

    # 総損失 (重み付けは調整可能ですが、基本は等倍でOK)
    total_loss = value_loss + move_loss + tile_loss

    return total_loss, (value_loss.item(), move_loss.item(), tile_loss.item())


# --- 動作確認 ---
if __name__ == "__main__":
    from logger import setup_logger

    # エントリーポイントでロギングを初期化
    setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastDualPolicyNet().to(device)

    # ダミー入力 (Batch=4)
    dummy_input = torch.randn(4, 90, 5, 5).to(device)

    m_out, t_out, v_out = model(dummy_input)

    logger.info(f"Move Output: {m_out.shape}")  # [4, 625]
    logger.info(f"Tile Output: {t_out.shape}")  # [4, 51]
    logger.info(f"Value Output: {v_out.shape}")  # [4, 1]
