import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torch.optim as optim
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from PIL import Image
import numpy as np

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

def load_data():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    testing_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return training_data, testing_data

def create_dataloaders(training_data, test_data, batch_size=64):
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader

class NN(nn.Module):
    def __init__(self, in_c, out_c, kern_dim, stride, padding):
        super().__init__()
        self.m = nn.Softmax(dim =1)
        self.fc1 = nn.Linear((28-kern_dim[1]+1)*(28-kern_dim[1]+1)*out_c, 10) #for stride = 1
        self.conv1 = Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kern_dim)
        self.relu1 = ReLU()

    def forward(self, x):
      x = self.conv1(x)
      x = self.relu1(x)

      x = flatten(x,1)
      x = self.fc1(x)
      x = self.m(x)
      return x
# y = (len(set([y for x,y in training_data])))
  
def train_network(train_loader, model1, optimizer,loss_function, e):
    for epoch in range(e):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model1(inputs)
            labels_onehot = torch.nn.functional.one_hot(labels, num_classes= 10)
            loss = loss_function(outputs, labels_onehot)
            loss.backward()
            optimizer.step()
 

def loss_fun(y_pred, y_ground):
    v = -(y_ground * torch.log(y_pred + 0.0001))
    v = torch.sum(v)
    return v

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from torchmetrics import Precision, Recall, F1Score, Accuracy
  
from torchmetrics.classification import accuracy

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            tmp = torch.nn.functional.one_hot(y, num_classes= 10)
            pred = model(X)
            test_loss += loss_fn(pred, tmp).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    accuracy1 = Accuracy()
    print('Accuracy :', accuracy1(pred,y))
    precision = Precision(average = 'macro', num_classes = 10)
    print('precision :', precision(pred,y))

    recall = Recall(average = 'macro', num_classes = 10)
    print('recall :', recall(pred,y))
    f1_score = F1Score(average = 'macro', num_classes = 10)
    print('f1_score :', f1_score(pred,y))
    return accuracy1,precision, recall, f1_score


def run_dynamic_model():
    config1 = [(1,10,(3,3),1,'same')]
    

    for i in config1:
        model = NN(i[0],i[1],i[2],i[3],i[4])
        
        training_data, test_data = load_data()
        train_loader, test_loader = create_dataloaders(training_data, test_data, batch_size = 32)
        
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        train_network(train_loader, model, optimizer,loss_fun,10)
        test(test_loader, model, loss_fun)
