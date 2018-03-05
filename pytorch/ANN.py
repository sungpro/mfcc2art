# Sung Hah Hwang
# sunghahh@gmail.com


from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from dataset import Dataset


# Params
input_size = 39
output_size = 16
hidden_size = 100
num_epochs = 100
batch_size = 128
learning_rate = 0.001


# Input pipeline
train_dataset = Dataset('MFCC_C.archive', 'ART_C.archive')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


class Net(nn.Module):
    """
    ANN: 5 hidden layers
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.h1 = nn.Sequential(
            nn.Linear(input_size, 300),
            nn.ReLU())
        self.h2 = nn.Sequential(
            nn.Linear(300, 500),
            nn.ReLU())
        self.h3 = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU())
        self.h4 = nn.Sequential(
            nn.Linear(500, 300),
            nn.ReLU())
        self.h5 = nn.Linear(300, 100)


    def forward(self, x):
        out = self.h1(x)
        out = self.h2(out)
        out = self.h3(out)
        out = self.h4(out)
        out = self.h5(out)
        return out

# Define net
net = Net(input_size, output_size)
net.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    for i, batch in enumerate(train_loader):
        MFCC, ART = batch['MFCC'], batch['ART']
        MFCC, ART = Variable(MFCC.cuda()), Variable(ART.cuda())

        # Forward, backward, optimize
        optimizer.zero_grad() # empty gradient buffer
        pred = net(MFCC)
        loss = criterion(pred, ART)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{:6d}/{:6d}],  Step [{:10d}/{:10d}],  Loss: {:8.4f}'
                .format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Save
torch.save(net.state_dict(), 'mfcc2art_model.pkl')