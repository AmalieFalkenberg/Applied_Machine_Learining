import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import random
import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool, global_add_pool, Set2Set
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy



import argparse

parser = argparse.ArgumentParser(description="Train CGCNN for Tc prediction")

parser.add_argument("--hd",    type=int,   default=128)
parser.add_argument("--c",        type=int,   default=3)
parser.add_argument("--m",         type=int,   default=2)
parser.add_argument("--title",         type=str,   default="CGCNN Tc prediction")


args = parser.parse_args()


class CGCNN(nn.Module):
    def __init__(
        self,
        node_features=3,      # Z, occupancy, disorder
        edge_features=100,    # Gaussian expansion steps
        hidden_dim=128,
        n_conv=3,
        n_mlp=2,
        dropout=0.1,
    ):
        super().__init__()

        # Project node features up to hidden_dim
        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        # CGConv layers (handles edge features natively)
        self.convs = nn.ModuleList([
            CGConv(channels=hidden_dim, dim=edge_features, batch_norm=False)
            for _ in range(n_conv)
        ])

        # Readout MLP
        mlp_layers = []
        in_dim = hidden_dim
        for _ in range(n_mlp - 1):
            mlp_layers += [nn.Linear(in_dim, in_dim // 2), nn.SiLU(), nn.Dropout(dropout)]
            in_dim = in_dim // 2
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        x = self.node_embedding(x)

        for conv in self.convs:
            x = x + conv(x, edge_index, edge_attr)  # residual connection

        x = global_mean_pool(x, batch)
        return self.mlp(x).squeeze(-1)



def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            preds.extend(pred.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    return np.array(preds), np.array(targets)

def denormalize(x, mean=3.3, std=2.5):
    """Reverse the log-normalization: x = (log(tc) - mean) / std"""
    return np.exp(x * std + mean)

def plot_results(preds, targets, ax, title="CGCNN Predictions vs Targets", denorm=True):
    if denorm:
        preds   = denormalize(preds)
        targets = denormalize(targets)
        mask =(preds < 135)
        preds = preds[mask]
        targets = targets[mask]

    mae  = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2   = 1 - ss_res / ss_tot

    # --- Parity plot ---
    ax.scatter(targets, preds, alpha=0.3, s=10, color='steelblue', label='predictions')
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='perfect')
    ax.set_xlabel("Target Tc (K)")
    ax.set_ylabel("Predicted Tc (K)")
    #ax.set_xlim(0,135)
    #ax.set_ylim(0,135)
    ax.set_title(f" {title}\nMAE={mae:.2f} K  RMSE={rmse:.2f} K  R²={r2:.3f}")
    ax.legend()


    print(f"MAE:  {mae:.2f} K")
    print(f"RMSE: {rmse:.2f} K")
    print(f"R²:   {r2:.3f}")
    print(f"Pred std:   {preds.std():.3f}")
    print(f"Target std: {targets.std():.3f}")


print("Loading graphs...")
graphs = torch.load('crystal_graphs_vClaude_ICSD_exp.pt', weights_only=False)

print("Preparing data loaders...")
# Assume you have: graphs = list of Data objects
rng = random.Random(42)
rng.shuffle(graphs)
n = len(graphs)
train_graphs = graphs[:int(0.9 * n)]
val_graphs   = graphs[int(0.9 * n):n]


train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=64)


node_features = graphs[0].x.shape[1]
edge_features = graphs[0].edge_attr.shape[1]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


model = CGCNN(
    node_features=node_features,
    edge_features=edge_features,
    hidden_dim=args.hd,
    n_conv=args.c,
    n_mlp=args.m,
).to(device)

model_path = f'cgcnn_vClaude_ICSD_c{args.c}_hd{args.hd}_m{args.m}_exp.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
fig, ax = plt.subplots(1,3 ,figsize=(12, 6))

# --- Run ---


# Optionally evaluate on train and val too to see the gap clearly
train_preds, train_targets = predict(model, train_loader, device)
val_preds,   val_targets   = predict(model, val_loader,   device)




print("\n--- Train ---")
plot_results(train_preds, train_targets, ax[0], title="Train set")

print("\n--- Val ---")
plot_results(val_preds, val_targets, ax[1], title="Val set")

#print("\n--- Test ---")
#print("Loading graphs...")
#graphs_test = torch.load('crystal_graphs_vClaude_MP_exp.pt', weights_only=False)
#test_loader   = DataLoader(graphs_test,   batch_size=64)


#preds, targets = predict(model, test_loader, device)
#plot_results(preds, targets, ax[2], title="Test set — CGCNN Tc prediction")

fig.suptitle(args.title + "c " + str(args.c) + "hd " + str(args.hd) + "m " + str(args.m), fontsize=16)
fig.tight_layout()
plt.savefig(f'plots/cgcnn_vClaude_ICSD_c{args.c}_hd{args.hd}_m{args.m}_exp_results.png', dpi=300)
plt.show()