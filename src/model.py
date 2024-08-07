import torch
import torch.nn as nn

class RNA_embedding(nn.Module):

    '''
    This class is a simple embedding layer for RNA sequences. 

    input:
    - embedding_dim: int, dimension of the embedding space
    - vocab_size: int, size of the vocabulary (number of different nucleotides)

    output:
    - x: tensor, (N, embedding_dim, L, L), where N is the batch size, L is the length of the sequence 
    '''

    def __init__(self, embedding_dim, vocab_size=5):
        super(RNA_embedding, self).__init__()

        self.table_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc_input = nn.Linear(embedding_dim*2, embedding_dim)

    def forward(self, x): # x is (N, L) -> embedded as sequence of integer

        # Sequence representation
        s = self.table_embedding(x)                         # (N, L, embedding_dim)

        # Outer concatenation to get matrix representation
        m = s.unsqueeze(2).repeat(1, 1, s.shape[1], 1)      # (N, L, L, embedding_dim)
        m = torch.cat((m, m.permute(0, 2, 1, 3)), dim=-1)   # (N, L, L, 2*embedding_dim)

        # Bring back to embedding dimension
        m = self.fc_input(m)                                # (N, L, L, embedding_dim)    

        m = m.permute(0, 3, 1, 2)                           # (N, embedding_dim , L, L)   

        return s, m


# This is the class to be developed !
class RNA_net(nn.Module):

    def __init__(self, embedding_dim):
        super(RNA_net, self).__init__()

        self.embedding = RNA_embedding(embedding_dim)

        # self.layers = nn.Sequential(
        #     torch.nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
        #     # torch.nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
        #     # torch.nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
        #     torch.nn.Conv2d(embedding_dim, 1, 3, padding=1)
        # )


        self.layers = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),   # Batch normalization after the first conv layer
            nn.ReLU(),                      # Activation function

            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),   # Batch normalization after the second conv layer
            nn.ReLU(),                      # Activation function

            nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1),
            nn.BatchNorm2d(embedding_dim),   # Batch normalization after the third conv layer
            nn.ReLU(),                      # Activation function

            # Note: No batch normalization or activation after the final conv layer
        )

        self.final_layer = nn.Conv2d(embedding_dim, 1, 3, padding=1)

        # Your layers here

    def forward(self, sequences):
        # x.shape (N, L)

        s, m = self.embedding(sequences)        # m.shape (N, d, L, L)
        residual = m                            # [N, d, L, L]

        m = self.layers(m)                      # [N, d, L, L]
        # print(f"{m.shape}")
        m = m + residual
        m = self.final_layer(m)

        # m = self.layer1(m)                      # m.shape (N, d, L, L)
        # m = self.layer2(m)                      # m.shape (N, 1, L, L)
        m = m.squeeze(1)                        # m.shape (N, L, L)

        return m
