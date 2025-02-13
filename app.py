import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load Pretrained BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Sample Dataset (Replace with IMDB dataset)
texts = ["This movie was amazing!", "I hated every second of it.", "An absolute masterpiece!", "Waste of time."]
labels = ["positive", "negative", "positive", "negative"]

# Convert labels to numerical form
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Tokenize and encode text using BERT
def get_bert_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Take CLS token embedding

X = get_bert_embeddings(texts)
y = np.array(labels)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement Forward-Forward Learning
class ForwardForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ForwardForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)  # Binary Classification

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return torch.sigmoid(self.layer2(x))

# Initialize Model
input_size = X_train.shape[1]
hidden_size = 128
model = ForwardForwardNetwork(input_size, hidden_size)

# Training Function (Forward-Forward)
def train_ff(model, X_train, y_train, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        inputs = torch.tensor(X_train, dtype=torch.float32)
        targets = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Train the Model
train_ff(model, X_train, y_train)
