import torch.nn as nn
import torch

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=32):
        super(SqueezeExcitation, self).__init__()
        hdim = 64
        self.conv1x1_in = nn.Sequential(nn.Conv2d(channel, hdim, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(hdim),
                                        nn.ReLU(inplace=False))
        self.conv1x1_out = nn.Sequential(nn.Conv2d(hdim, channel, kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(channel),
                                        nn.ReLU(inplace=False))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hdim, hdim // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hdim // reduction, hdim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1x1_in(x)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        w=self.fc(y) #b,c
        y = self.fc(y).view(b, c, 1, 1)
        x = x * y.expand_as(x)
        x = self.conv1x1_out(x)
        return w



if __name__ == '__main__':
    '''
    m = nn.MaxPool2d(3, stride=1,padding=1)
    n=  nn.MaxPool2d(5, stride=1,padding=2)
    input = torch.randn(1, 1, 20, 20)
    output1 = m(input)
    output1= m(output1)
    print(output1)
    print('***********************************')
    output2=n(input)
    print(output2)
    
    m = nn.MaxPool2d(3, stride=1,padding=1)
    n=  nn.MaxPool2d(9, stride=1,padding=4)
    input = torch.randn(1, 1, 20, 20)
    output1 = m(input)
    output1= m(output1)
    output1= m(output1)
    output1= m(output1)
    print(output1)
    print('***********************************')
    output2=n(input)
    print(output2)
    '''
    m = nn.MaxPool2d(3, stride=1,padding=1)
    n=  nn.MaxPool2d(13, stride=1,padding=6)
    input = torch.randn(1, 1, 20, 20)
    output1 = m(input)
    output1= m(output1)
    output1= m(output1)
    output1= m(output1)
    output1= m(output1)
    output1= m(output1)
    print(output1)
    print('***********************************')
    output2=n(input)
    print(output2)