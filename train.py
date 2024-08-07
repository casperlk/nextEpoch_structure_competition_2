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

train_loader, val_loader, test_loader = get_dataloaders(batch_size = 4, max_length=50, split=0.8, max_data=1000)

# Init model, loss function, optimizer
embedding_dim = 128
model = RNA_net(embedding_dim).to(device)
loss_fn = nn.BCEWithLogitsLoss().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
epochs = 1000

# Training loop
train_losses = []
train_precisions = []
test_losses = []
test_precisions = []
for epoch in range(epochs):
    train_loss = 0.0
    train_precision = 0.0
    test_loss = 0.0
    test_precision = 0.0

    # loss_per_epoch = 0.0
    # f1_per_epoch = 0.0

    # Training Loop
    for batch in train_loader:
        # Predict
        sequence = batch["sequence"].to(device) # (N, L)
        structure = batch["structure"].to(device) # (N, L, L)

        structure_pred = model(sequence)
        # print(f"{structure_pred.shape=}")
        # print(f"{structure.shape=}")


        # Optimize
        loss = loss_fn(structure_pred, structure)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        # loss_per_epoch += loss.item()
        # f1_per_epoch += compute_f1(structure_pred, structure)
        train_loss += loss.item()
        train_precision += torch.sum(torch.argmax(structure_pred, dim=1) == structure).item()/len(structure)


        train_loss += loss.item()
        # train_precision
    train_losses.append(train_loss)
    # loss_avg = loss_per_epoch/len(train_loader)
    # f1_avg = f1_per_epoch/len(train_loader)
    # print(f"loss: {loss_avg},       f1: {f1_avg}")
    # print(f"loss_per_epoch per batch avg={loss_per_epoch/len(train_loader)}")
    # print(f"f1_per_epoch per batch avg={f1_per_epoch/len(train_loader)}")
    # print(f"{f1_per_epoch/len(train_loader)=}")
    # print(f'Epoch {epoch} Train loss: {train_losses[-1]}')

    # Validation Loop

    for batch in val_loader:
        sequence = batch["sequence"].to(device) # (N, L)
        with torch.no_grad():
            structure_pred = model(sequence)
        loss = loss_fn(structure_pred, structure)

        # Metrics
        test_loss += loss.item()
        test_precision += torch.sum(torch.argmax(structure_pred, dim=1) == structure).item()/len(structure)
        # print(f'Epoch {epoch} Valid loss: {test_losses[-1]}, precision: {test_precisions[-1]}')
    
    train_losses.append(train_loss/len(train_loader))
    train_precisions.append(train_precision/len(train_loader))
    test_losses.append(test_loss/len(test_loader))
    test_precisions.append(test_precision/len(test_loader))
    print(f'Epoch {epoch} Train loss: {train_losses[-1]}, precision: {train_precisions[-1]}')
    print(f'Epoch {epoch} Valid loss: {test_losses[-1]}, precision: {test_precisions[-1]}')
    print("----")






# Validation loop

# Test loop
structures = []
for sequence in test_loader[1]:
    # Replace with your model prediction !
    structure = (torch.rand(len(sequence), len(sequence))>0.9).type(torch.int) # Has to be shape (L, L) ! 
    structures.append(structure)

format_submission(test_loader[0], test_loader[1], structures, 'test_pred.csv')