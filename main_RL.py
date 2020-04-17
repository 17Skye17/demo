import os
import shutil
import torch
import warnings
from dataloader import VideoDataset
from tensorboardX import SummaryWriter
from model import *
from utils import AverageMeter, accuracy
from mapmeter import mAPMeter
from a2c_agent import A2CAgent
import torch.nn.functional as F

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
num_classes = 200
epochs = 5000
gpus = 4
num_frames = 120
best_mAP = 0 
start_epoch = 0
drop_rate = 0.5
batch_size = 2048
resume = False
validate = False
#save_dir = '/home/skye/ckpts/BNInception-frames120-RLModel/'
save_dir = './ckpts/RL-logits/'
ckpt_path = './ckpts/BNInception/ckpt.best.pth.tar'
ROOT = '/DATACENTER/skye/ActivityNet/labels/'

train_list = os.path.join(ROOT,'train_videofolder.txt')
val_list = os.path.join(ROOT,'val_videofolder.txt')
feature_list = ['bninception_logits.hdf5','resnet50_logits.hdf5']
tf_writer = SummaryWriter(log_dir='./log/anet_RL_bninception_frames120')

train_dataset = VideoDataset(feature_list, train_list, num_frames=num_frames, sample_strategy='uniform')
val_dataset = VideoDataset(feature_list, val_list, num_frames=num_frames, sample_strategy='uniform')

train_video_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=0, pin_memory=True)

val_video_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0, pin_memory=True)

#TODO model
#model = Average(1024+2048,num_classes,drop_rate=drop_rate)
#model = FC3(1024, 2048, num_classes, drop_rate=drop_rate)
#model = BNInceptionFC(1024, num_classes)
model = RLModel(num_classes, 200)
#model = torch.nn.DataParallel(model).cuda()
model = model.cuda()
agent = A2CAgent(model, episode_len=20, sampling='categorical')

if validate:
   resume = True

if resume:
    checkpoint = torch.load(ckpt_path)
    start_epoch = checkpoint['epoch']
    best_mAP = checkpoint['best_mAP']
    model.load_state_dict(checkpoint['state_dict'])
    print ("Loaded checkpoint {} epoch {} best_mAP {}".format(ckpt_path, start_epoch, best_mAP))

# optimizer 
print ('Prams to learn:')
params_to_update = []
for name, param in model.named_parameters():
    params_to_update.append(param)
    print ('\t', name)

optimizer = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9, weight_decay=1e-4)

class CrossEntropy_onehot():
    def __init__(self, size_average=True):
        self.size_average = size_average

    def __call__(self, input, target):

        logsoftmax = torch.nn.LogSoftmax()
        if self.size_average:
            return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
        else:
            return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))

criterion = CrossEntropy_onehot()

def save_checkpoint(state, is_best):
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
    filename = os.path.join(save_dir,'ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar','best.pth.tar'))

# TODO train
def train(model, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    mAP = mAPMeter()

    model.train()
    lambd = torch.nn.Parameter(torch.rand(1)).cuda()
    for i, (vid, feature, label) in enumerate(train_video_loader):
        #output = model(feature[0].cuda(), feature[1].cuda())
        #output = model(feature[0].cuda())
        label = label.cuda(non_blocking=True)
        loss_RL, output, rewards, locations = agent.rollout(feature[0].cuda(), label)
        loss_CE = criterion(output, label) 
        loss = loss_RL + lambd * loss_CE
        #print ("lambd: {}  loss_RL: {} loss_CE: {}".format(lambd, loss_RL, loss_CE))
        prec1, prec5 = accuracy(output.data, label, topk=(1,5))
        losses.update(loss.item(), output.size(0))
        top1.update(prec1,output.size(0))
        top5.update(prec5,output.size(0))
        with torch.no_grad():
            output = F.softmax(output, dim=1)
            mAP.add(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 2 == 0 :
            print ('Epoch {}  step {}: loss: {}  top1: {:.5f}  top5:{:.5f}  mAP:{:.5f}'.format(epoch, i, losses.avg, top1.val, top5.val, mAP.value()))
    
    tf_writer.add_scalar('loss_RL/train', loss_RL, epoch)
    tf_writer.add_scalar('loss_CE/train', loss_CE, epoch)

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('mAP/train', mAP.value(), epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)
    mAP.reset()

def validate(model, epoch):
    top1 = AverageMeter()
    top5  = AverageMeter()
    mAP = mAPMeter()
    model.eval()
    print ("Enter validation........\n")
    with torch.no_grad():
        for i, (vid, feature, label) in enumerate(val_video_loader):

            #output = model(feature[0].cuda(), feature[1].cuda())
            #output = model(feature[0].cuda())
            
            label = label.cuda(non_blocking=True)
            loss_RL, output, rewards, locations = agent.rollout(feature[0].cuda(), label)
            prec1, prec5 = accuracy(output.data, label, topk=(1,5))
            top1.update(prec1,output.size(0))
            top5.update(prec5,output.size(0))
            with torch.no_grad():
                output = F.softmax(output, dim=1)
                mAP.add(output, label)
            if i % 2 == 0 :
                print ('Epoch {}  step {}: top1: {:.5f}  top5:{:.5f}  mAP:{:.5f}'.format(epoch, i, top1.val, top5.val, mAP.value()))
    return mAP.value()

#if validate:
#    mAP = validate(model, start_epoch)
#    exit(0)

#agent = A2CAgent(model, episode_len=args.epoch, sampling='categorical')
for epoch in range(start_epoch, epochs):
    # TODO eval
    train(model, epoch)
    if (epoch+1) % 5 == 0:
        mAP = validate(model, epoch)
        is_best = mAP > best_mAP
        best_mAP = max(mAP, best_mAP)
        print ('Best mAP: {:.5f}'.format(best_mAP))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'bninception_resnet50',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_mAP': best_mAP,
            }, is_best)
