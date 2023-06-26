import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

def load_model(file_path):
    model = TheModelClass(*args, **kwargs)  # you need to know the exact class and parameters
    model.load_state_dict(torch.load(file_path))
    model.eval()  # make sure to call eval() to set dropout and batch normalization layers to evaluation mode
    return model

def load_dataset(url):
    return pd.read_csv(url)

def load_data():
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

    return train_loader, test_loader

def train_and_evaluate(model):
    num_epochs = 5
    learning_rate = 0.001

    train_loader, test_loader = load_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            if isinstance(model, Net):
                images = images.reshape(-1, 28*28)
            elif isinstance(model, ConvNet):
                images = images.unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item()}')

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            if isinstance(model, Net):
                images = images.reshape(-1, 28*28)
            elif isinstance(model, ConvNet):
                images = images.unsqueeze(1)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

