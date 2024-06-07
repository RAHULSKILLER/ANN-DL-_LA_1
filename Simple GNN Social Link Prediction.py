import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNLinkPredictor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(2 * hidden_channels, out_channels)

    def forward(self, x, edge_index, node_pairs):
        print("Input shape:", x.shape)
        x = F.relu(self.conv1(x, edge_index))
        print("Shape after first convolution:", x.shape)
        x = self.conv2(x, edge_index)
        print("Shape after second convolution:", x.shape)

        # Extract node embeddings for the node pairs
        node_embs = torch.cat([x[node_pairs[:, 0]], x[node_pairs[:, 1]]], dim=-1)
        print("Node embeddings shape:", node_embs.shape)
        return self.fc(node_embs)

# Example usage
model = GNNLinkPredictor(in_channels=1433, hidden_channels=64, out_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Assuming `data` is a PyG data object containing your graph
data = torch_geometric.datasets.Planetoid(root='data/Cora', name='Cora')[0]

# Define node pairs and labels for training
# `train_node_pairs` and `train_labels` should be prepared beforehand
train_node_pairs = torch.tensor([[0, 1], [2, 3], [4, 5]])  # Example node pairs
train_labels = torch.tensor([[0.], [1.], [0.]])  # Example labels with an extra dimension

# Forward pass through the model
output = model(data.x, data.edge_index, train_node_pairs)
print("Output logits:", output.detach().numpy())

# Compute the loss
loss = F.binary_cross_entropy_with_logits(output, train_labels)  # Adjusted target tensor
print("Loss:", loss.item())  # Convert loss tensor to a scalar value for printing

# Backpropagation and optimization step
loss.backward()
optimizer.step()

# Convert logits to probabilities
probabilities = torch.sigmoid(output)
print("Output probabilities:", probabilities.detach().numpy())

# Assuming a threshold of 0.5 to decide the presence of a link
predicted_links = (probabilities > 0.5).float()
print("Predicted links:", predicted_links.detach().numpy())








