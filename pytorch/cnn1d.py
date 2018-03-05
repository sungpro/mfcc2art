# Sung Hah Hwang
# sunghahh@gmail.com


from __future__ import print_function
from dataset import Dataset
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable


# Params
input_size = 39
output_size = 16
num_epochs = 100
batch_size = 128
learning_rate = 0.003


# Input pipeline
train_dataset = Dataset('MFCC_C.archive', 'ART_C.archive')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_dataset = Dataset('MFCC_C_test.archive', 'ART_C_test.archive')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


class Net(nn.Module):
    """
    ANN: 5 hidden layers
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 1, 2),
            nn.MaxPool1d(3, stride=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(1, 1, 2),
            nn.MaxPool1d(3, stride=1),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(33, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 16)


    def forward(self, x):
        out = torch.unsqueeze(x, 1) # unsqueeze to (N,C,L)
        out = self.conv1(out)
        out = self.conv2(out)
        input_size = out.size(0)
        out = out.view(input_size, -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# Define net
net = Net()
net.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def train(num_epochs):
    net.train()
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            MFCC, ART = batch['MFCC'], batch['ART']
            MFCC, ART = Variable(MFCC.cuda()), Variable(ART.cuda())
            optimizer.zero_grad()
            pred = net(MFCC)
            loss = criterion(pred, ART)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{:6d}/{:6d}],  Step [{:10d}/{:10d}],  Loss: {:8.4f}'
                    .format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


def test():
    net.eval()
    test_loss = 0
    loss = 0
    for data, target in test_loader:
        MFCC, ART = batch['MFCC'], batch['ART']
        MFCC, ART = Variable(MFCC.cuda(), volatile=True), Variable(ART.cuda())
        pred = net(MFCC)
        test_loss += criterion(pred, ART, size_average=False).data[0]

    test_loss /= batch_size
    print('Step [{:10d}/{:10d}],  Loss: {:8.4f}'
                .format(i+1, len(test_dataset)//batch_size, loss.data[0]))


train(num_epochs)
test()



# Save
torch.save(net.state_dict(), 'mfcc2art_cnn_model.pkl')