import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ResBlock, self).__init__()
        self.type = type
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.conv1 = nn.Conv2d(inchannels, outchannels, 3, padding=1)
        self.conv2 = nn.Conv2d(outchannels, outchannels, 3, padding=1)
        if inchannels != outchannels:
            self.conv1x1 = nn.Conv2d(inchannels, outchannels, 1)

    def forward(self, x):
        if self.inchannels == self.outchannels:
            x_init = x
        else:
            x_init = self.conv1x1(x)
        x = F.selu(self.conv1(x))
        x = self.conv2(x)
        x = x+x_init
        x = F.selu(x)
        return x
            
class DilationBlock(nn.Module):
    def __init__(self, channels, dilfactors): # channels[0] is the inchannels to this block
        super(DilationBlock, self).__init__()
        self.channels = channels       
        self.dilfactors = dilfactors
        self.dilblock = nn.ModuleList([nn.Sequential(
                                                     nn.Conv2d(channels[0], channels[i+1], 3, padding=dilfactors[i]-1, dilation=dilfactors[i]),
                                                     nn.SELU())
                                                     for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        out = []
        for m in self.dilblock:
            out.append(m(x))
        out = torch.cat(out, dim=1)
        out = self.pool(out)
        return out

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet1 = nn.Sequential(nn.Conv2d(3, 64, 5), nn.SELU(),
                                     nn.MaxPool2d(2, stride=2))
        self.resnet1 = ResBlock(64, 128)
        self.dilnet1 = DilationBlock([128, 128, 64, 64], [1, 2, 3])
        self.resnet2 = nn.Sequential(ResBlock(256, 256), ResBlock(256, 256))
        self.dilnet2 = DilationBlock([256, 256, 256], [1, 2])
        self.resnet3 = nn.Sequential(ResBlock(512, 512), ResBlock(512, 512),
                                     ResBlock(512, 512), ResBlock(512, 512),
                                     ResBlock(512, 512))
        self.convnet2 = nn.Sequential(nn.Conv2d(512, 1024, 3), nn.SELU(),
                                     nn.MaxPool2d(2, stride=2))
        self.resnet4 = nn.Sequential(ResBlock(1024, 1024), ResBlock(1024, 1024),
                                     ResBlock(1024, 1024))
                                         

        self.fc = nn.Linear(1024*7*7, 1024)

    def forward(self, x):
        x = self.convnet1(x)
        x = self.resnet1(x)
        x = self.dilnet1(x)
        x = self.resnet2(x)
        x = self.dilnet2(x)
        x = self.resnet3(x)
        x = self.convnet2(x)
        x = self.resnet4(x)
        x = self.fc(x.view(-1, 1024*7*7))
        x, _ = x.view(*x.shape[:1], x.shape[1] // 2, 2, *x.shape[2:]).max(2) # max out is happening here   
        #x_norm = x / x.pow(2).sum(1, keepdim=True).sqrt() #L2 normalization; I'm  doing it in the main
        return x

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        #self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(512, n_classes)

        self.alpha = nn.Parameter(torch.tensor([12.0], dtype=torch.float)) #for l2_softmax with trainable scale parameter alpha; initialized to zero

    def forward(self, x):
        output = self.embedding_net(x)
        #output = self.nonlinear(output)
        output = output / output.pow(2).sum(1, keepdim=True).sqrt() #L2 normalization
        output_s = self.alpha*output #L2 softmax with trainable scale parameter alpha
        scores = self.fc1(output_s)
        norm_scores = F.log_softmax(scores, dim=1)
        return norm_scores, output

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2
    

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

if __name__ == '__main__':
    #breakpoint()
    x = torch.rand(2, 3, 144, 144).cuda()
    m = EmbeddingNet().cuda()
    #out = m(x)
    #s = SiameseNet(m)
    #out = s(x, x)
    t = TripletNet(m)
    out = t(x, x, x)
   
