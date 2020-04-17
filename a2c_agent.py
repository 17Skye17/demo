import torch
import torch.nn as nn
import torch.distributions as distri
import torch.nn.functional as F
from utils import init_hidden, CUDA_or_not, gumbel_softmax
import numpy as np

class A2CAgent(object):
    def __init__(self, model, episode_len=20, sampling='normal'):
        '''
        Parameters:
        --------------

        model: the model contains LSTM(agent), selection network and utility network
        
        episode_len: the maximum number of frames to take for a video
        
        hidden_size: the hidden size for LSTM unit

        embedding_size: the embedding_size for input feature
        
        episode_rewards: deprecated 

        tot_frames: total number of video frames

        global_embedding_size: deprecated

        sampling: "normal" or "categorical"

        rnn_input_size: -1 dimension size of [h_t, global_feature]
        
        '''
        
        
        self.model = model
        self.episode_len = episode_len #self.model.max_steps
        self.hidden_size = self.model.hidden_size
        self.embedding_size = self.model.embedding_size
        self.episode_rewards = []
        self.action_dim = self.model.action_dim
        self.std = CUDA_or_not(torch.ones(1, self.action_dim)*0.1)
        
        self.tot_frames = self.model.tot_frames
        self.sampling = sampling
        self.rnn_input_size = self.model.rnn_input_size
    
    def select_action_categorical(self, logits, training):
        '''
        Function: For training, sample an action(an index) from logits
                  For inference, take the index of the max number in logits

        Parameters:
        --------------
        
        logits: a list of values in Categorical Distribution
        action: a sample from the indices of logits
        dist.log_prob(action): given the probability density function of logits, get the value of PDF
                               on action, so that we can calculate the policy loss: 
                               Expectation of action * reward
                               (REINFORCE uses reward, A2C uses Advantage)
        dist.entropy(logits): sum(-logits*log(logits))
        '''
        dist = distri.Categorical(logits=logits)
        if training:
            action = dist.sample()
        else:
            _, action = logits.max(dim=-1)

        log_prob = dist.log_prob(action).unsqueeze(-1)
        return action, log_prob, dist.entropy().unsqueeze(-1)

    # original paper selection network: training 
    def select_action_normal(self, logits, training):
        '''
        Function: sample an action from logits obey normal distribution
        
        Parameters:
        --------------

        action: a list of the same size as logits, in Gassusian Distribution
        log_prob: take the sum of values in action
        '''
        
        dist = torch.distributions.Normal(logits, self.std)
        if training:
            action = dist.sample()
        else:
            action = dist.mean
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)

        return action, log_prob, CUDA_or_not(torch.Tensor(np.zeros((log_prob.size(0), 1))))

    def get_rewards(self, predictions_history, targets):
        
        '''
        Function: Definition of reward function, calculate the reward for an action for a single step

        Parameters:
        --------------

        max_margin: the margin between the probability of GT class and the largest probability 
                    from other classes.

        rewards: max(cur_margin-max_margin,0)
                 
        predictions_history: a list of predictions before time step t, 
                             size = [time_steps, batch_size, num_classes]
        
        predictions: the latest prediction [batch_size, num_classes]
        
        lbl: the max prediction

        targets: ground truth, one-hot vector

        cur_pred: get the corresponding predictions for targets' indices 
        
        largest_values: the max prediction value except for the value in GT index
        
        largest_idx: the max prediction index except for the GT index

        cur_margin: GT_pred - max_rest_pred 
        
        all_preds: all the predictions other than the lastest prediction 
                        [timesteps-1, batch_size, num_classes]

        rest_max_preds: the max prediction other than gt class in t-1 steps
                        [timesteps-1, batch_size, num_classes]
        
        max_margin: the max margin between gt class prediction and non-gt class prediction by now

        Return:  
        --------------
        
        rewards: (cur_margin - max_margin)  a list on cpu

        '''
        predictions_history = [F.softmax(x, dim=1) for x in predictions_history]
        predictions = predictions_history[-1]
        _, lbl = predictions.max(dim=-1)
        # for single class
        # targets -> [batchsize, num_classes]   predictions -> [batchsize, num_classes]
        # targets = targets.cpu().numpy()
        # index = np.nonzero(targets)
        cur_pred = torch.zeros([predictions.size(0)])
        
        # transfer targets into 1 dim
        onehot_targets = targets.cpu().numpy()
        targets = np.zeros([predictions.size(0)])
        for i in range(predictions.size(0)):
            gt  = onehot_targets[i]
            index = np.nonzero(gt)[0]
            targets[i] = index[0]
        targets = torch.LongTensor(targets).cuda()
        cur_pred = predictions[torch.LongTensor(range(predictions.size(0))), targets]
        #print ("cur_pred={}".format(cur_pred.shape))
        # get all predictions other than the gt class
        cur_mask = torch.ones_like(predictions)
        cur_mask[torch.LongTensor(range(predictions.size(0))), targets] = 0
        
        rest_cur_preds = predictions.masked_select(cur_mask.byte()).view(predictions.size(0), -1)
        largest_values, largest_idx = rest_cur_preds.max(dim=1)
        cur_margin = cur_pred - largest_values
        
        # handle predictions in 0~(t-1)  steps
        if len(predictions_history) > 1:
            # all_preds -> [timesteps-1, batch_size, num_classes]
            all_preds = torch.stack(predictions_history[:-1])
            # gt classses preds over time
            # gt_preds -> [timesteps-1, batch_size, num_classes]
            gt_preds = all_preds[:, torch.LongTensor(range(all_preds.size(1))), targets]

            # get the max prediction in the 0~(t-1) steps
            max_pred, t = gt_preds.max(0)

            # second largest
            mask = torch.ones_like(all_preds)
            
            # get all predictions other than the gt class
            mask[:, torch.LongTensor(range(all_preds.size(1))), targets] = 0
            
            # rest_max_preds -> [timesteps-1, batch_size, num_classes]
            rest_max_preds, _ = all_preds.masked_select(mask.byte()).view(
                all_preds.size(0), all_preds.size(1), -1).max(-1)

            max_margin, _ = (gt_preds - rest_max_preds).max(0)

        else:
            # the first time step
            max_pred = cur_pred
            max_margin = cur_margin

        rewards = (cur_margin - max_margin)
        rewards = torch.clamp(rewards, min=0)
        rewards = rewards.data.cpu().numpy()

        return rewards.tolist()

    def rollout(self, inputs, targets):
        '''
        Function: Get policy gradient loss to update policy network

        Parameters:
        -------------
        targets: GT

        rollout: record all the log_probs, actions, values, rewards and entropy

        h_t: initialized h_t input to LSTM

        c_t: initialized c_t input to LSTM

        features: the first frame feature, encoded into [-1, embedding_size]

        global_feature: [batch_size, feature_size] 

        hx: initialized input to LSTM

        dropmask: for dropout in LSTM
        
        episode_len: the iteration times for a video

        embedding_size: the last dimension size of inputs

        rnn_input_size: the last dimension size of [features, global_feat] 
    
        locations: the selected frame indices in a batch 

        terminals: [batch_size], False

        value: output of critic network 
        
        actions: the actions to take at time step t

        rewards: the rewards for action at time step t
        '''
        rollout = []
        batch_size = inputs.size()[0]

        h_t = init_hidden(batch_size, self.hidden_size)
        c_t = init_hidden(batch_size, self.hidden_size)
        h_t = h_t.view(1, batch_size, self.hidden_size)
        c_t = c_t.view(1, batch_size, self.hidden_size)
        # input the first frame
        # features = inputs[:, 0, :].view(-1, self.embedding_size)
        features = inputs[:,0,:].view(-1, 1, self.embedding_size)
        # global_feature, size=1024
        #global_fea = inputs_small 
        targets = targets.long()
        hx = (h_t, c_t)
        locations = []
        predictions_history = []
        predicted_values = []
        
        # set drop out for multilayer LSTM, a dropout is applied to each LSTM layer except the last layer
        dropout = 0.5
        # utilize dropout only for training
        #if self.model.training:
        #    lockdrop = torch.rand(batch_size, self.rnn_input_size).cuda()
        #    lockdrop = lockdrop.bernoulli_(1-dropout)
        #    dropmask = torch.autograd.Variable(lockdrop, requires_grad=False) / (1 - dropout)
        #else:
        #    dropmask = None

        #self.model.wdrnn._setweights()

        for step in range(self.episode_len):

            h_t, c_t, predictions, logits, values = self.model(features, hx)
            logits = F.sigmoid(logits)
            if self.sampling == 'normal':
                actions, log_probs, entropy = self.select_action_normal(logits, self.model.training)
                actions = actions.squeeze()
                # let all values in actions are in [0.0,1.0], and take the ceil of float index 
                # as final selected index
                actions = torch.ceil(torch.clamp(actions, min=0, max=1) * (self.tot_frames - 1))
            else:
                actions, log_probs, entropy = self.select_action_categorical(logits, self.model.training)
            
            # actions-> [batch_size, 1]
            locations.append(actions)

            # get new features
            # get selected features, according to the indices in actions
            features = inputs[torch.LongTensor(range(inputs.size(0))), actions.long()]
            features = features.view(-1, 1, self.embedding_size)
            # terminals->[batch_size]
            terminals = np.array([False] * inputs.size(0))

            predictions_history.append(predictions)
            predicted_values.append(values.squeeze())
             
            # new input to LSTM
            hx = (h_t, c_t)
            #if step == self.episode_len - 1:
            #    rewards = self.get_rewards(predictions_history, targets)
            #else:
            #    rewards = self.get_rewards(predictions_history, targets)
            rewards = self.get_rewards(predictions_history, targets)

            rollout.append([log_probs, values, actions, rewards, 1-terminals, entropy])
        # the value for last action
        pending_value = self.model(features, hx)[-1]
        rollout.append([None, pending_value, None, None, None, None])
        
        # locations -> [batch_size]
        locations = torch.stack(locations, dim=1)
        rl_losses, rewards = self.process_rollout(batch_size, rollout, pending_value.data)

        #return rl_losses, predictions, rewards, locations, stop_indices+1
        # when to stop
        return rl_losses, predictions, rewards, locations

    def process_rollout(self, batch_size, rollout, returns, use_gae=False):
        '''
        Parameters:
        --------------
        returns: initialized as the value at the last time step, the futurn return of action
                 at time step t

        rewards: the rewards at time step t
        '''
        # rollout -> [episode,...] 
        processed_rollout = [None] * (len(rollout) - 1)
        advantages = CUDA_or_not(torch.zeros(batch_size, 1))
        r_stack = []
        
        # calculate the value function
        for i in reversed(range(len(rollout) - 1)):
            log_prob, value, actions, rewards, ended, entropy = rollout[i]
            ended = CUDA_or_not(torch.from_numpy(ended).unsqueeze(1).float())

            rewards = CUDA_or_not(torch.FloatTensor(rewards).unsqueeze(1))  # rewards.size() = (batchsize, 1)
            actions = actions.data.unsqueeze(1)  # actions.size() = (batchsize, 1)
            next_value = rollout[i + 1][1]  # next_value.size() = (batchsize, 1)

            # accumulate rewards with discount factor
            returns = rewards + 0.9 * returns  # returns.size() = (batchsize, 1)

            if not use_gae:
                # use advantages to reduce variance
                advantages = returns - value.data
            else:
                # generalized advantage esetimator
                td_error = rewards + 0.9*ended*next_value.data - value.data
                advantages = advantages * 0.9 * ended + td_error

            r_stack.append(returns)

            processed_rollout[i] = [log_prob, value, returns, advantages, entropy]

        return self.criterion(processed_rollout), returns.mean().data.cpu()

    def criterion(self, processed_rollout):
        '''
        Parameters:
        --------------
        policy_loss: the expectation of expected future reward. 
                     Expectation of action_PDF*advantage(rewards for REINFORCE)
        
        value_loss: regression loss for utility network(critic) 
        
        entropy_loss(?)
        '''
        log_prob, value, returns, advantages, entropy = map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob * advantages
        value_loss = 0.5 * (returns - value).pow(2)

        # take the mean entropy value of a batch
        entropy_loss = entropy.mean()
         
        loss = (policy_loss - entropy_loss + value_loss).mean()

        return loss
