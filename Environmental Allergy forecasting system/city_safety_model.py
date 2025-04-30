import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Define the PyTorch Dataset class
class CitySafetyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the CNN-LSTM model
class CitySafetyModel(nn.Module):
    def __init__(self, sequence_length, n_features):
        super(CitySafetyModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(sequence_length, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(16)
        self.conv4 = nn.Conv1d(16, 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(8)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(8, 50, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(100, 50, num_layers=2, batch_first=True, bidirectional=True)
        
        # Dense layers
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # CNN layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Reshape for LSTM
        x = x.permute(0, 2, 1)
        
        # LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
        
        # Take the last output
        x = x[:, -1, :]
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x

def load_data():
    # Load city data
    city_data = pd.read_csv('dataset/Indian Cities Database.csv')
    
    # Load pollen data
    pollen_data = pd.read_csv('dataset/pollen_luxembourg_Dataset.csv')
    
    # Convert date to datetime
    pollen_data['Date'] = pd.to_datetime(pollen_data['Date'])
    
    # Since we don't have city information in pollen data, we'll use a default location
    # (Luxembourg's approximate coordinates)
    pollen_data['Lat'] = 49.8153
    pollen_data['Long'] = 6.1296
    
    # Add more features
    pollen_data['TempRange'] = pollen_data['MaxAirTempC'] - pollen_data['MinAirTempC']
    pollen_data['Month'] = pollen_data['Date'].dt.month
    pollen_data['Season'] = pollen_data['Month'].apply(lambda x: (x % 12 + 3) // 3)
    
    # Calculate pollen concentration features
    pollen_columns = ['Ambrosia', 'Artemisia', 'Asteraceae', 'Alnus', 'Betula', 
                     'Ericaceae', 'Carpinus', 'Castanea', 'Quercus', 'Chenopodium']
    pollen_data['TotalPollen'] = pollen_data[pollen_columns].sum(axis=1)
    pollen_data['MaxPollen'] = pollen_data[pollen_columns].max(axis=1)
    
    # Create features for safety prediction
    features = [
        'MaxAirTempC', 'MinAirTempC', 'PrecipitationC', 'Lat', 'Long',
        'TempRange', 'Month', 'Season', 'TotalPollen', 'MaxPollen'
    ]
    
    # Create a more sophisticated safety score based on multiple factors
    pollen_data['SafetyScore'] = (
        (pollen_data['MaxAirTempC'] < 35) &  # Temperature not too high
        (pollen_data['MinAirTempC'] > 0) &   # Temperature not too low
        (pollen_data['PrecipitationC'] < 100) &  # Not too much precipitation
        (pollen_data['TempRange'] < 20) &    # Not too much temperature variation
        (pollen_data['TotalPollen'] < 100)   # Not too much pollen
    ).astype(int)
    
    return pollen_data, features

def create_sequences(data, features, sequence_length=10):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[features].iloc[i:(i + sequence_length)].values)
        y.append(data['SafetyScore'].iloc[i + sequence_length])
    return np.array(X), np.array(y)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += y_batch.size(0)
            train_correct += (predicted.squeeze() == y_batch).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += y_batch.size(0)
                val_correct += (predicted.squeeze() == y_batch).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load and preprocess data
    data, features = load_data()
    
    # Create sequences
    sequence_length = 10  # Reduced sequence length
    X, y = create_sequences(data, features, sequence_length)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Create data loaders
    train_dataset = CitySafetyDataset(X_train_scaled, y_train)
    test_dataset = CitySafetyDataset(X_test_scaled, y_test)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Reduced batch size
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model, loss function, and optimizer
    model = CitySafetyModel(sequence_length, len(features)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate
    
    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50  # Reduced epochs
    )
    
    # Save the best model
    torch.save(model.state_dict(), 'best_model.pth')
    
    # Evaluate the model
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            predicted = (outputs > 0.5).float()
            test_total += y_batch.size(0)
            test_correct += (predicted.squeeze() == y_batch).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    print(f'\nTest accuracy: {test_accuracy:.2f}%')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_training_history.png')
    plt.close()

if __name__ == "__main__":
    main() 