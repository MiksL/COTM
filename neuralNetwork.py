import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

# Tensor cores for faster training
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

class ChessNN(pl.LightningModule):
    def __init__(self, input_channels=18, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        
        # Input layer 
        self.conv_input = nn.Conv2d(input_channels, 128, 3, padding=1)
        
        # 2 Residual blocks - TODO: test increased number block effect on training speed and performance
        self.res_blocks = nn.ModuleList([
            self._make_res_block(128) for _ in range(2)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_head = nn.Linear(32 * 64, 64 * 69)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 32, 1)
        self.value_fc = nn.Linear(32 * 64, 1)
        
        # Convert to channels_last memory format - better performance(?)
        self = self.to(memory_format=torch.channels_last)
    
    # Residual block - 2 conv layers with batch norm and ReLU
    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        ''' Forward pass through the network '''	
        
        # Initial convolution - ReLU
        x = F.relu(self.conv_input(x))
        
        # Residual blocks - ReLU
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = F.relu(x + residual)
        
        # Policy head - ReLU with a linear layer
        policy = F.relu(self.policy_conv(x))
        policy = self.policy_head(policy.flatten(1))
        
        # Value head
        value = F.relu(self.value_conv(x))
        value = torch.tanh(self.value_fc(value.flatten(1)))
        
        # Ensure value has shape (batch_size, 1)
        if value.dim() == 1:
            value = value.unsqueeze(1)
            
        return policy, value
    
    def _compute_loss(self, batch):
        positions, move_indices, values = batch
        
        # Convert positions to channels_last memory format - better performance(?)
        if positions.dim() == 4 and positions.shape[1] > 1:
            positions = positions.to(memory_format=torch.channels_last)
            
        policy_out, value_out = self(positions)
        
        # Get policy loss, both policy_out and move_indices have shape (N, 69), 69 being the number of possible moves
        policy_loss = F.cross_entropy(policy_out, move_indices)
        
        # Value loss calculation with MSE, value_out and values both having the shape (N, 1)
        value_loss = F.mse_loss(value_out, values)
        
        return policy_loss + 0.5 * value_loss, policy_out, move_indices
    
    def _shared_step(self, batch, step_type='train'):
        ''' Shared training/validation step check '''
        loss, policy_out, move_indices = self._compute_loss(batch)
        
        # Get predicted moves and create an accuracy metric
        _, predicted_moves = torch.max(policy_out, 1)
        correct = (predicted_moves == move_indices).sum().item()
        accuracy = correct / move_indices.size(0)
        
        # Log metrics
        self.log(f'{step_type}_loss', loss, prog_bar=True)
        self.log(f'{step_type}_accuracy', accuracy, prog_bar=True)
        
        return loss

    def training_step(self, batch):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch):
        return self._shared_step(batch, 'val')
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau( # TODO - test other schedulers more
            optimizer, patience=3, factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch'
            }
        }