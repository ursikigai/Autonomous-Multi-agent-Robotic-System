# bc_train.py
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, os
from torch.utils.data import TensorDataset, DataLoader

class BCNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_bc(states, actions, model_out='models/bc.pth', epochs=40, bs=128, lr=1e-3, device='cpu'):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    X = torch.tensor(states, dtype=torch.float32)
    Y = torch.tensor(actions, dtype=torch.float32)

    ds = TensorDataset(X, Y)
    dl = DataLoader(ds, batch_size=bs, shuffle=True)

    model = BCNet(X.shape[1], Y.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()

    for ep in range(epochs):
        total = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = lossfn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            n += xb.size(0)

        print(f"Epoch {ep+1}/{epochs} - Loss: {total/n:.6f}")

    torch.save(model.state_dict(), model_out)
    print("Model saved to", model_out)

    return model

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='results/dataset_synth.npz')
    p.add_argument('--out', default='models/bc.pth')
    p.add_argument('--epochs', type=int, default=40)
    args = p.parse_args()

    data = np.load(args.dataset)
    states, actions = data['states'], data['actions']

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("Training on device:", device)

    train_bc(states, actions, model_out=args.out, epochs=args.epochs, device=device)

