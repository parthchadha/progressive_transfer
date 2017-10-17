import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from running_stat import ObsNorm
from distributions import Categorical, DiagGaussian
from utils import AddBias
import random
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        value, x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return value, action

    def evaluate_actions(self, inputs, actions):
        value, x = self(inputs)
        action_log_probs, dist_entropy = self.dist.evaluate_actions(x, actions)
        return value, action_log_probs, dist_entropy

class ProgressivePolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, previous_column, backward):
        super(ProgressivePolicy, self).__init__()
        
        print("Do you want backward connection: ", backward)
        self.previous_column = previous_column
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4, bias=False)

        alpha_list = [1, 0.1, 0.01]
        self.alpha1 = nn.Parameter(torch.from_numpy(np.array([random.choice(alpha_list)])).float())
        print(self.alpha1)
        self.alpha2 = nn.Parameter(torch.from_numpy(np.array([random.choice(alpha_list)])).float())
        print(self.alpha2)
        self.alpha3 = nn.Parameter(torch.from_numpy(np.array([random.choice(alpha_list)])).float())
        print(self.alpha2)

        if backward:
            self.alpha4 = nn.Parameter(torch.from_numpy(np.array([random.choice(alpha_list)])).float())

        #self.V1 = nn.Conv2d(32, 16, 1, stride=1, bias=True)
        self.U1 = nn.Conv2d(32, 32, 3, padding=1, stride=1, bias=True)
        self.U2 = nn.Conv2d(64, 64, 3, padding=1, stride=1, bias=True)

        self.ab1 = AddBias(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.ab2 = AddBias(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, bias=False)
        self.ab3 = AddBias(32)

        self.V3 = nn.Conv2d(32, 8, 1, stride=1, bias=True)
        self.U3 = nn.Linear(8 * 7 * 7, 32*7*7, bias=True)

        self.linear1 = nn.Linear(32 * 7 * 7, 512, bias=False)
        self.ab_fc1 = AddBias(512)

        self.critic_linear = nn.Linear(512, 1, bias=False)
        self.ab_fc2 = AddBias(1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.U1.weight.data.mul_(relu_gain)

        if action_space.__class__.__name__ == "Box":
            self.dist.fc_mean.weight.data.mul_(0.01)

        self.train()

    def forward(self, inputs):
        self.previous_column.forward(inputs)
        a1 = self.previous_column.layer1 * self.alpha1
        v1 = self.U1(a1)
        v1 = F.relu(v1)
        x = self.conv1(inputs/255.0)
        x = self.ab1(x)
        x = F.relu(x + v1)

        x = self.conv2(x)
        x = self.ab2(x)
        
        a2 = self.previous_column.layer2 * self.alpha2
        v2 = self.U2(a2)
        v2 = F.relu(v2)

        x = F.relu(x + v2)

        x = self.conv3(x)
        x = self.ab3(x)

        a3 = self.previous_column.layer3 * self.alpha3
        a3 = self.V3(a3)
        a3 = F.relu(a3)
        a3 = a3.view(-1, 8 * 7 * 7)
        a3 = self.U3(a3)
        
        #x = F.relu(x)


        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(x + a3)
        x = self.linear1(x)
        x = self.ab_fc1(x)
        x = F.relu(x)
        print(self.alpha1, self.alpha2, self.alpha3)

        return self.ab_fc2(self.critic_linear(x)), x


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(CNNPolicy, self).__init__()
        
        #self.layer1, self.layer2, self.layer3, self.layer4, self.layer5 = None
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4, bias=False)
        self.ab1 = AddBias(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=False)
        self.ab2 = AddBias(64)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1, bias=False)
        self.ab3 = AddBias(32)

        self.linear1 = nn.Linear(32 * 7 * 7, 512, bias=False)
        self.ab_fc1 = AddBias(512)

        self.critic_linear = nn.Linear(512, 1, bias=False)
        self.ab_fc2 = AddBias(1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(512, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(512, num_outputs)
        else:
            raise NotImplementedError

        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if action_space.__class__.__name__ == "Box":
            self.dist.fc_mean.weight.data.mul_(0.01)

        self.train()

    def forward(self, inputs):
        x = self.conv1(inputs / 255.0)
        x = self.ab1(x)
        x = F.relu(x)
        self.layer1 = x

        x = self.conv2(x)
        x = self.ab2(x)
        x = F.relu(x)
        self.layer2 = x

        x = self.conv3(x)
        x = self.ab3(x)
        x = F.relu(x)
        self.layer3 = x

        x = x.view(-1, 32 * 7 * 7)
        x = self.linear1(x)
        x = self.ab_fc1(x)
        x = F.relu(x)
        self.layer4 = x

        return self.ab_fc2(self.critic_linear(x)), x


def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class MLPPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space):
        super(MLPPolicy, self).__init__()

        self.obs_filter = ObsNorm((1, num_inputs), clip=5)
        self.action_space = action_space

        self.a_fc1 = nn.Linear(num_inputs, 64, bias=False)
        self.a_ab1 = AddBias(64)
        self.a_fc2 = nn.Linear(64, 64, bias=False)
        self.a_ab2 = AddBias(64)

        self.v_fc1 = nn.Linear(num_inputs, 64, bias=False)
        self.v_ab1 = AddBias(64)
        self.v_fc2 = nn.Linear(64, 64, bias=False)
        self.v_ab2 = AddBias(64)
        self.v_fc3 = nn.Linear(64, 1, bias=False)
        self.v_ab3 = AddBias(1)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(64, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(64, num_outputs)
        else:
            raise NotImplementedError

        self.apply(weights_init_mlp)

        tanh_gain = nn.init.calculate_gain('tanh')
        #self.a_fc1.weight.data.mul_(tanh_gain)
        #self.a_fc2.weight.data.mul_(tanh_gain)
        #self.v_fc1.weight.data.mul_(tanh_gain)
        #self.v_fc2.weight.data.mul_(tanh_gain)

        if action_space.__class__.__name__ == "Box":
            self.dist.fc_mean.weight.data.mul_(0.01)

        self.train()

    def cuda(self, **args):
        super(MLPPolicy, self).cuda(**args)
        self.obs_filter.cuda()

    def cpu(self, **args):
        super(MLPPolicy, self).cpu(**args)
        self.obs_filter.cpu()

    def forward(self, inputs):
        inputs.data = self.obs_filter(inputs.data)

        x = self.v_fc1(inputs)
        x = self.v_ab1(x)
        x = F.tanh(x)

        x = self.v_fc2(x)
        x = self.v_ab2(x)
        x = F.tanh(x)

        x = self.v_fc3(x)
        x = self.v_ab3(x)
        value = x

        x = self.a_fc1(inputs)
        x = self.a_ab1(x)
        x = F.tanh(x)

        x = self.a_fc2(x)
        x = self.a_ab2(x)
        x = F.tanh(x)

        return value, x
