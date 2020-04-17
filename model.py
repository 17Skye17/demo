import torch
from torch import nn
from basic_ops import ConsensusModule
import torch.nn.functional as F
class Average(nn.Module):
    def __init__(self, num_ftrs, num_classes,drop_rate=0):
        super(Average,self).__init__()
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, feature1, feature2):
       avg_feature1 = feature1.mean(dim=1)
       avg_feature1 = F.normalize(avg_feature1)
       avg_feature2 = feature2.mean(dim=1)
       avg_feature2 = F.normalize(avg_feature2)
       num_ftrs = avg_feature1.size()[1] + avg_feature2.size()[1]
       feature = torch.cat([avg_feature1, avg_feature2],1)
       if self.drop_rate > 0:
          feature = self.dropout(feature)
       x = self.fc(feature)
       return x
#class RL(nn.Module):
#    def forward(self,):
class Logits(nn.Module):
    def __init__(self, num_ftrs1, num_ftrs2, num_classes,batch_size):
        super(Logits, self).__init__()
        self.consensus = ConsensusModule('avg')
        param = nn.Parameter(torch.ones(batch_size, num_classes))
        self.a = torch.autograd.Variable(param, requires_grad=True)
        #self.a = torch.autograd.Variable(torch.ones(batch_size, num_classes),requires_grad=True)

    def forward(self, feature1, feature2):
        feature1 = self.consensus(feature1)
        feature2 = self.consensus(feature2)
        score = feature1 * self.a + feature2 * (1 - self.a)
        return score

class FC3(nn.Module):
    
    def __init__(self, num_ftrs1, num_ftrs2, num_classes, drop_rate=0):
        super(FC3, self).__init__()
        self.num_classes = num_classes
        hidden_size = 200
        self.fc1 = nn.Linear(num_ftrs1, hidden_size)
        self.fc2 = nn.Linear(num_ftrs2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        self.drop_rate = drop_rate
        self.consensus = ConsensusModule('avg')
        self.dropout = nn.Dropout(self.drop_rate)

    def forward(self, feature1, feature2):
        
        #feature1 = feature1.mean(dim=1)
        #feature2 = feature2.mean(dim=1)
        
        feature1 = self.fc1(feature1)
        #feature1 = self.bn1(feature1)
        feature1 = self.consensus(feature1)
        feature1 = feature1.squeeze()
        
        feature2 = self.fc2(feature2)
        #feature2 = self.bn2(feature2)
        feature2 = self.consensus(feature2)
        feature2 = feature2.squeeze()

        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)
        feature = torch.cat([feature1, feature2],1)
        if self.drop_rate > 0:
            feature = self.dropout(feature)
        feature = self.fc(feature)
        return feature

class FC(nn.Module):
    def __init__(self, num_ftrs, num_classes):
        super(BNInceptionFC,self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.consensus = ConsensusModule('avg')
    def forward(self, feature):
        #feature = feature.mean(dim=1)
        #feature = self.consensus(feature)
        feature = self.fc(feature)
        #feature = feature.mean(dim=1)
        feature = self.consensus(feature)
        feature = feature.view(-1, self.num_classes)
        return feature

class RLModel(nn.Module):
    def __init__(self, num_classes, embedding_size, dropmask=None):
        super(RLModel, self).__init__()
        self.tot_frames = 120
        self.action_dim = 25
        self.embedding_size = embedding_size
        self.rnn_input_size = embedding_size
        
        self.hidden_size = 200
        self.num_classes = num_classes
        self.LSTM = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=self.hidden_size,        
            num_layers=1,       
            batch_first=True,
	)
        self.prednet = nn.Linear(self.embedding_size, self.num_classes)

        self.selectnet = nn.Linear(self.embedding_size, 1)
    
        self.utilnet = nn.Linear(self.embedding_size, 1)
    
    def forward(self, features, hx):
        out, (ht, ct) = self.LSTM(features, hx)
        
        prediction = self.prednet(ht.squeeze())
        logits = self.selectnet(ht.squeeze())
        values = self.utilnet(ht.squeeze())
        return ht, ct, prediction, logits, values

