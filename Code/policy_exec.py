# policy_exec.py
import torch
import numpy as np
from bc_train import BCNet

class Policy:
    def __init__(self, model_path, in_dim):
        # Device: MPS for Apple Silicon, fallback to CPU
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'

        # Load model architecture
        self.net = BCNet(in_dim, 2).to(self.device)

        # Load pretrained weights
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()

    def act(self, state_np):
        """
        state_np: numpy array, shape (in_dim,)
        Returns action: np.array of shape (2,) = (dx, dy)
        """
        x = torch.tensor(state_np.reshape(1, -1), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            a = self.net(x).cpu().numpy().reshape(-1)
        return a

