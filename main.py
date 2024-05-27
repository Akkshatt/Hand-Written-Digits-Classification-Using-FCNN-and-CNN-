import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])


trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)


testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)



class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64,10)
    
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

model = Network()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.03)

# Training Model.
epochs = 10
losses = []
for e in range(epochs):
    running_loss = 0
    val_loss = 0.0
    correct = 0
    total = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        val_loss += loss.item() * images.size(0)
        _, predicted = torch.max(output, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        val_loss /= len(trainloader.dataset)
        accuracy = correct / total
    else:
        average_loss = running_loss / len(trainloader)
       
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Epoch {e+1}/{epochs}")
        print(f"Training loss: {average_loss}")
        print(f"Accuracy: {accuracy:.4f}")
        losses.append(average_loss)

plt.figure(figsize=(10,5))
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Time')



correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1,784)
        with torch.no_grad():
            logps = model(img)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
            correct_count+=1
        all_count +=1
        
print("Number of Images Tests =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

torch.save(model.state_dict(), 'trained_model.pth')


