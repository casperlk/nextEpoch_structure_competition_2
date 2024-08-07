from src.data import get_dataloaders
from src.model import RNA_net
from src.util import compute_f1, compute_precision, compute_recall, plot_structures
from src.submission_formatter import format_submission
import torch.nn as nn

import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = (
    "cuda" if torch.cuda.is_available() else
    # "mps" if torch.backends.mps.is_available() else
    "cpu"
)

train_loader, val_loader, test_loader = get_dataloaders(batch_size = 4, max_length=50, split=0.8, max_data=1000)

# Init model, loss function, optimizer
embedding_dim = 128
model = RNA_net(embedding_dim).to(device)
loss_fn = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
epochs = 5

# Training loop
train_losses = []
# train_precisions = []
# test_losses = []
# test_precisions = []
for epoch in range(epochs):
    train_loss = 0.0
    # train_precision = 0.0
    # test_loss = 0.0
    # test_precision = 0.0

    # Training Loop
    for batch in train_loader:
        # Predict
        # reference, sequence, structure = batch["reference"], batch["sequence"], batch["structure"]
        reference = batch["reference"]
        sequence = batch["sequence"] # (N, L)
        structure = batch["structure"] # (N, L, L)

        structure_pred = model(sequence)
        # print(f"{structure_pred.shape=}")
        # print(f"{structure.shape=}")

        loss = loss_fn(structure_pred, structure)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        train_loss += loss.item()
        # train_precision
    train_losses.append(train_loss)
    print(f'Epoch {epoch} Train loss: {train_losses[-1]}')

    # Validation Loop





# Validation loop

# Test loop
structures = []
for sequence in test_loader[1]:
    # Replace with your model prediction !
    structure = (torch.rand(len(sequence), len(sequence))>0.9).type(torch.int) # Has to be shape (L, L) ! 
    structures.append(structure)

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')