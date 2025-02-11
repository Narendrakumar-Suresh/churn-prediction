import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

#class for data

class Churnpredict:
    def __init__(self,X,y):
        self.X=X
        self.y=y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

df=pd.read_csv('./final_dataset.csv')

X=df.drop(columns=['Churn']).values
y=df['Churn'].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

batch_size=100
learning_rate=0.01
num_epochs=2
dataset = Churnpredict(X, y)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size  # This ensures the split sums up to the total dataset length

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
sample, labels = next(examples)

class NeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,hidden_size)
        self.relu2=nn.ReLU()
        self.l3=nn.Linear(hidden_size,hidden_size)
        self.relu3=nn.ReLU()
        self.l4=nn.Linear(hidden_size,num_classes)
    
    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)
        out=self.relu2(out)
        out=self.l3(out)
        out=self.relu3(out)
        out=self.l4(out)

        return out

input_size=5
hidden_size = 100
num_classes=1

model=NeuralNet(input_size=input_size,hidden_size=hidden_size,num_classes=num_classes)

criterion=nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(model.parameters(),lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (sample, labels) in enumerate(train_loader):
        sample=sample.to(device)
        labels = labels.to(device).view(-1, 1)

        # forward
        outputs = model(sample)
        loss = criterion(outputs, labels)

        # backward
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if (i + 1) % 100 == 0:
            print(
                f"epcoh{epoch + 1}/{num_epochs}, step: {i + 1}/{n_total_steps} loss: {loss.item():.4f}"
            )
