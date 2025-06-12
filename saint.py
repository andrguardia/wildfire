import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ----------------------
# Data Preprocessing
# ----------------------
def load_and_preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df.ffill(inplace=True)
    df.columns = df.columns.str.strip()
    cat_cols = ['EVT']
    num_cols = [
        'EVH', 'NDVI', 'PRES_max', 'SPFH_max', 'TMP_max', 'WIND_max',
        'elevation', 'sm_profile', 'sm_profile_wetness', 'sm_rootzone',
        'sm_rootzone_wetness', 'sm_surface', 'sm_surface_wetness'
    ]
    # Extract date parts if present
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        num_cols += ['year', 'month', 'day']
        df.drop(columns=['date'], inplace=True)
    # Ensure target is int
    df['burned'] = df['burned'].astype(int)
    # Extract coordinates from .geo
    def extract_coords(geo_str):
        geo = json.loads(geo_str)
        return pd.Series(geo['coordinates'], index=['longitude', 'latitude'])
    if '.geo' in df.columns:
        df[['longitude', 'latitude']] = df['.geo'].apply(extract_coords)
        num_cols += ['longitude', 'latitude']
        df.drop(columns=['.geo'], inplace=True)
    # Encode categorical columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    cat_dims = [df[col].nunique() for col in cat_cols]
    # Scale numerical columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    X = df[cat_cols + num_cols]
    y = df['burned']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Ensure categorical columns are first
    all_cols = cat_cols + [col for col in X_train.columns if col not in cat_cols]
    X_train = X_train[all_cols]
    X_test = X_test[all_cols]
    # Double-check categorical values
    for i, col in enumerate(cat_cols):
        max_val = X_train[col].max()
        assert max_val < cat_dims[i], f"Column {col} has value {max_val} >= {cat_dims[i]}"
    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return X_train, X_test, y_train, y_test, cat_cols, num_cols, cat_dims

# ----------------------
# SAINT Model
# ----------------------
class SAINT(pl.LightningModule):
    def __init__(self, num_cont, cat_dims=[], embed_dim=32, 
                 num_heads=4, num_layers=3, dropout=0.1, pos_weight=None, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['pos_weight'])
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in cat_dims
        ])
        self.cont_proj = nn.Linear(num_cont, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            batch_first=True, dim_feedforward=embed_dim*4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
    def forward(self, x_cat, x_cont):
        tokens = []
        for i, emb in enumerate(self.embeddings):
            tokens.append(emb(x_cat[:, i]))
        tokens.append(self.cont_proj(x_cont))
        tokens = torch.stack(tokens, dim=1)
        attn_output = self.transformer(tokens)
        cls_token = attn_output.mean(dim=1)
        return self.classifier(cls_token).squeeze(-1)
    def training_step(self, batch, batch_idx):
        x_cat, x_cont, y = batch
        logits = self(x_cat, x_cont)
        loss = self.loss_fn(logits, y.float())
        self.log('train_loss', loss)
        return loss
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

# ----------------------
# DataModule
# ----------------------
class FireDataModule(pl.LightningDataModule):
    def __init__(self, X_train, y_train, X_test, y_test, cat_cols, batch_size=64):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cat_cols = cat_cols
        self.batch_size = batch_size
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    def setup(self, stage=None):
        n_cat = len(self.cat_cols)
        X_train_cat = torch.tensor(self.X_train.iloc[:, :n_cat].values, dtype=torch.long, device=self.device)
        X_train_cont = torch.tensor(self.X_train.iloc[:, n_cat:].values, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(self.y_train.values, dtype=torch.float32, device=self.device)
        X_test_cat = torch.tensor(self.X_test.iloc[:, :n_cat].values, dtype=torch.long, device=self.device)
        X_test_cont = torch.tensor(self.X_test.iloc[:, n_cat:].values, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(self.y_test.values, dtype=torch.float32, device=self.device)
        self.train_dataset = TensorDataset(X_train_cat, X_train_cont, y_train)
        self.val_dataset = TensorDataset(X_test_cat, X_test_cont, y_test)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# ----------------------
# Training Pipeline
# ----------------------
def train_saint(csv_path, batch_size=64, embed_dim=32, num_heads=4, num_layers=3, dropout=0.1, lr=1e-3, max_epochs=50):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    X_train, X_test, y_train, y_test, cat_cols, num_cols, cat_dims = load_and_preprocess_data(csv_path)
    model = SAINT(
        num_cont=len(num_cols),
        cat_dims=cat_dims,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr
    )
    dm = FireDataModule(X_train, y_train, X_test, y_test, cat_cols=cat_cols, batch_size=batch_size)
    accelerator = 'mps' if torch.backends.mps.is_available() else 'cpu'
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min')
        ]
    )
    trainer.fit(model, dm)
    return model, trainer

# ----------------------
# Main execution
# ----------------------
def main():
    import platform
    print(f"Processor: {platform.processor()}")
    print(f"Machine: {platform.machine()}")
    csv_path = 'dataset.csv'
    model, trainer = train_saint(csv_path)
    print('Training complete!')

if __name__ == "__main__":
    main()
