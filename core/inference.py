import numpy as np
import torch
import torch.nn.functional as F

def predict_moves(model, positions, device=None):
    """Run inference on pre-encoded chess positions"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure model is on the right device
    model = model.to(device)
    model.eval()
    
    # Convert to float32 if needed
    if isinstance(positions, np.ndarray) and positions.dtype != np.float32:
        positions = positions.astype(np.float32)
    
    # Convert to torch.FloatTensor
    if isinstance(positions, np.ndarray):
        position_tensor = torch.FloatTensor(positions).to(device)
    else:
        position_tensor = positions.to(device)
    
    # Convert to channels_last format for better performance(?)
    if position_tensor.dim() == 4 and position_tensor.shape[1] > 1:
        position_tensor = position_tensor.to(memory_format=torch.channels_last)
    
    # Run inference with mixed precision
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        policy_out, value_out = model(position_tensor)
        policy_probs = F.softmax(policy_out, dim=1)
    
    return policy_probs.cpu().numpy(), value_out.cpu().numpy()