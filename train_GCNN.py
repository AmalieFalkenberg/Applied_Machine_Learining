import random

import torch
import torch.nn as nn
from torch_geometric.nn import CGConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np

class CGCNN(nn.Module):
    def __init__(self, node_features=8, edge_features=40,
                 hidden_dim=64, num_conv_layers=3, dropout=0.1):
        super(CGCNN, self).__init__()

        # Initial embedding of node features
        self.node_embedding = nn.Linear(node_features, hidden_dim)

        # Graph convolutional layers
        self.conv_layers = nn.ModuleList([
            CGConv(hidden_dim, dim=edge_features, batch_norm=True)
            for _ in range(num_conv_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Readout MLP
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )

        # Embed node features
        x = self.node_embedding(x)
        x = torch.relu(x)

        # Graph convolutions
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = torch.relu(x)
            x = self.dropout(x)

        # Global pooling: aggregate all atoms into one vector
        x = global_mean_pool(x, batch)

        # Predict
        out = self.fc(x)
        return out.squeeze(-1)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)

        y = batch.y.view(-1)
        w = batch.w.view(-1)  # sample weights

        loss = (w * (pred - y) ** 2).mean()  # weighted MSE
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)

            y = batch.y.view(-1)
            w = batch.w.view(-1)

            loss = (w * (pred - y) ** 2).mean()  # weighted MSE
            total_loss += loss.item()
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())

    mae = np.mean(np.abs(np.array(preds) - np.array(targets)))
    return total_loss / len(loader), mae


def train_cgcnn(graphs, test_size=0.0, val_size=0.1,
                hidden_dim=64, num_conv_layers=3,
                epochs=100, batch_size=32, lr=1e-3, dropout=0.1, 
                weight_decay=1e-4, patience=50):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Split dataset
    rng = random.Random(42)
    rng.shuffle(graphs)
    n = len(graphs)
    train_graphs = graphs[:int(0.9 * n)]
    val_graphs   = graphs[int(0.9 * n):n]


    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=64)

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=batch_size)

    # Model, optimizer, loss
    node_features = graphs[0].x.shape[1]
    edge_features = graphs[0].edge_attr.shape[1]

    model = CGCNN(
        node_features=node_features,
        edge_features=edge_features,
        hidden_dim=hidden_dim,
        num_conv_layers=num_conv_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10)

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    loss_a = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_mae = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0          # reset on improvement
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.4f} | "
                  f"Patience: {patience_counter}/{patience}")
            loss_a.append([train_loss, val_loss])

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} — "
                  f"val loss hasn't improved for {patience} epochs.")
            break

    model.load_state_dict(best_model_state)
    return model, loss_a



graphs = torch.load('graphs/crystal_graphs_vClaude_ICSD_exp.pt', weights_only=False)



model, loss_list = train_cgcnn(
    graphs,
    hidden_dim=128,
    num_conv_layers=2,
    epochs=500,
    batch_size=32,
    lr=1e-3,
    dropout=0.15,  # light regularization
    weight_decay=1e-4,
)

# Save trained model
torch.save(model.state_dict(), 'old_cgcnn_vClaude_ICSD_hd128_bs32_c2.pt')


