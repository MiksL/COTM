import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

class BaseChessNN(pl.LightningModule):
    """
    Base class for the Chess Neural Network architecture.
    Defines the core layers and forward pass.
    Training logic should be implemented in subclasses.
    """
    def __init__(self, input_channels=18, num_res_blocks=10, policy_output_size=1968, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks
        self.policy_output_size = policy_output_size # 1968

        # Input layer
        self.conv_input = nn.Conv2d(self.input_channels, 128, 3, padding=1)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(128) for _ in range(self.num_res_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, 1)
        # Calculate flattened size dynamically based on intermediate conv output
        # Assuming 8x8 board after convolutions
        self.policy_fc_input_size = 32 * 8 * 8
        self.policy_head = nn.Linear(self.policy_fc_input_size, self.policy_output_size)

        # Value head
        self.value_conv = nn.Conv2d(128, 32, 1)
        # Calculate flattened size dynamically
        self.value_fc_input_size = 32 * 8 * 8
        self.value_fc = nn.Linear(self.value_fc_input_size, 1)

        # Try converting to channels_last memory format if beneficial
        # self = self.to(memory_format=torch.channels_last)

    def _make_res_block(self, channels):
        """Creates a residual block."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        """ Forward pass through the network """
        # Ensure input is float
        x = x.float()

        # Apply channels_last if possible and beneficial
        if x.dim() == 4 and x.is_cuda:
             # Only apply if input is 4D and on GPU
             # Check if stride suggests it's not already channels last
             if x.stride(-1) == 1 and x.stride(1) != 1:
                  x = x.to(memory_format=torch.channels_last)


        # Initial convolution -> ReLU
        x = F.relu(self.conv_input(x))

        # Residual blocks -> ReLU
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = F.relu(x + residual) # Add residual connection back

        # Policy head -> ReLU -> Linear
        policy_intermediate = F.relu(self.policy_conv(x))
        policy_logits = self.policy_head(policy_intermediate.flatten(1)) # Flatten from dim 1

        # Value head -> ReLU -> Linear -> Tanh
        value_intermediate = F.relu(self.value_conv(x))
        value = torch.tanh(self.value_fc(value_intermediate.flatten(1))) # Flatten from dim 1

        # Ensure value has shape (batch_size, 1)
        if value.dim() == 1:
            value = value.unsqueeze(1)

        return policy_logits, value

    def configure_optimizers(self):
        """Default optimizer configuration."""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # Example scheduler (can be customized in subclasses)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss', # Default monitor, might change in subclass
                'interval': 'epoch',
                'frequency': 1
            }
        }

    # Placeholder training/validation steps - MUST be overridden by subclasses
    def training_step(self, batch, batch_idx):
        raise NotImplementedError("training_step must be implemented in subclasses")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("validation_step must be implemented in subclasses")