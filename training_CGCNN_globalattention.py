import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool, global_add_pool, Set2Set


import random


from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy



from torch_geometric.nn import CGConv, GlobalAttention, global_mean_pool

class CGCNN(nn.Module):
    def __init__(
        self,
        node_features=3,
        edge_features=100,
        hidden_dim=128,
        n_conv=3,
        n_mlp=2,
        dropout=0.1,
    ):
        super().__init__()

        self.node_embedding = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )

        self.convs = nn.ModuleList([
            CGConv(channels=hidden_dim, dim=edge_features, batch_norm=False)
            for _ in range(n_conv)
        ])

        # Attention gate: learned scalar score per atom
        self.pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )

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
            x = x + conv(x, edge_index, edge_attr)

        x = self.pool(x, batch)  # replaces global_mean_pool
        return self.mlp(x).squeeze(-1)
    






def weighted_mse(pred, target, weight):
    return (1 * (pred - target) ** 2).mean()

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = weighted_mse(pred, batch.y.squeeze(), batch.w.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = weighted_mse(pred, batch.y.squeeze(), batch.w.squeeze())
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def train(model, train_loader, val_loader, epochs=300, lr=3e-4, patience=30, device='cuda'):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=1e-5)

    best_val = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = eval_epoch(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


if True:
    print("Loading graphs...")
    graphs_train =     graphs = torch.load('crystal_graphs_vClaude_ICSD_train_exp.pt', weights_only=False)
    graphs_val   =     graphs = torch.load('crystal_graphs_vClaude_ICSD_val_exp.pt', weights_only=False)
    train_loader = DataLoader(graphs_train, batch_size=64)
    val_loader   = DataLoader(graphs_val,   batch_size=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    node_features = graphs_train[0].x.shape[1]
    edge_features = graphs_train[0].edge_attr.shape[1]
    print(f"Node features: {node_features}, Edge features: {edge_features}")

    model = CGCNN(
    node_features=node_features,
    edge_features=edge_features,
    hidden_dim=128,
    n_conv=3,
    n_mlp=3,
    dropout=0.15,
    ).to(device)

    model = train(model, train_loader, val_loader, epochs=300, lr=3e-4, device=device)

    torch.save(model.state_dict(), f'cgcnn_vglobalattention_ICSD_c{4}_hd{128}_m{3}_exp_split_final_unweighted.pt')



if False:
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")



    node_features = graphs[0].x.shape[1]
    edge_features = graphs[0].edge_attr.shape[1]
    print(f"Node features: {node_features}, Edge features: {edge_features}")

    model = CGCNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=128,
        n_conv=3,
        n_mlp=3,
        dropout=0.15,
    ).to(device)

    model = train(model, train_loader, val_loader, epochs=300, lr=3e-4, device=device)

    torch.save(model.state_dict(), f'cgcnn_vglobalattention_ICSD_c{3}_hd{128}_m{3}_exp_final.pt')
