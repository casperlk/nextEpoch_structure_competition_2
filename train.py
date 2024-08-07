from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from src.submission_formatter import format_submission
import torch.nn as nn

import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = (
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

train_loader, val_loader, test_loader = get_dataloaders(batch_size = 8, max_length=50, split=0.8, max_data=1000)

# Init model, loss function, optimizer
embedding_dim = 128
model = RNA_net(embedding_dim).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([300])).to(device)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
epochs = 1000

# Training loop
train_losses = []
valid_losses = []
f1s_train = []
f1s_valid = []

for epoch in range(epochs):

    loss_train = 0.0
    f1_train = 0.0
    loss_valid = 0.0
    f1_valid = 0.0

    # Training Loop
    for batch in train_loader:
        # Predict
        sequence = batch["sequence"].to(device) # (N, L)
        structure = batch["structure"].to(device) # (N, L, L)

        structure_pred = model(sequence)

        # Optimize
        loss = loss_fn(structure_pred, structure)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        loss_train += loss.item()
        f1_train += compute_f1(structure_pred, structure)

    # Validation Loop

    for batch in val_loader:
        sequence = batch["sequence"].to(device) # (N, L)
        structure = batch["structure"].to(device) # (N, L)

        # with torch.no_grad():
        structure_pred = model(sequence)
        loss = loss_fn(structure_pred, structure)

        # Metrics
        loss_valid += loss.item()
        f1_valid += compute_f1(structure_pred, structure)

    
    train_losses.append(loss_train/len(train_loader))
    valid_losses.append(loss_valid/len(val_loader))

    f1s_train.append(f1_train/len(train_loader))
    f1s_valid.append(f1_valid/len(val_loader))

    print(f"Epoch {epoch}, F1 train: {f1s_train[-1]:.2f}, F1 valid: {f1s_valid[-1]:.2f}")




# Validation loop

# Test loop
structures = []
sequences = test_loader[1]
for sequence in sequences:
    # structure = (model(sequence.unsqueeze(0)).squeeze(0)>0.5).type(torch.int) # Has to be shape (L, L) ! 
    structure = (model(sequence.to(device).unsqueeze(0)).squeeze (0)>0.5).type(torch.int) # Has to be shape (L, L) !
    structures.append(structure)

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')