import torch
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable
import torch.nn as nn
#import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys
sys.path.append("..")
from utils.functions import CreateOnehotVariable, TimeDistributed
import numpy as np
import pdb
import math
from .att_modules import eca_layer
import torch.nn.functional as F

class pBLSTMLayer(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, rnn_uint = "LSTM", dropout_rate=0.0):
        super(pBLSTMLayer, self).__init__()
        self.rnn_uint = getattr(nn, rnn_uint.upper())

        self.BLSTM = self.rnn_uint(
            input_feature_dim * 2,
            hidden_dim,
            1,
            bidirectional = True,
            dropout = dropout_rate,
            batch_first = True, 
        )
    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        time_reduc = int(timestep / 2)
        input_xr = input_x.contiguous().view(batch_size, time_reduc,feature_dim*2)
        output, hidden = self.BLSTM(input_xr)
        return output, hidden


class Listener(nn.Module):
    def __init__(
            self,
            input_feature_dim,
            hidden_size,
            num_layers,
            rnn_unit,
            use_gpu,
            dropout_rate = 0.0,
            **kwargs,
                 ):
        super(Listener, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_unit = rnn_unit
        self.dropout_rate = dropout_rate
        assert self.num_layers >=1, "Lisnter should have at least 1 layer"
        self.pLSTM_layer0 = pBLSTMLayer(
            input_feature_dim, hidden_size, rnn_uint=rnn_unit, dropout_rate=dropout_rate,
        )
        for i in range(1, self.num_layers):
            setattr( # to add attribute (pblstm_layer1,2,3) to class (Listener)
                self,
                "pLSTM_layer" + str(i),
                pBLSTMLayer(
                    hidden_size * 2, hidden_size, rnn_uint=rnn_unit, dropout_rate=dropout_rate,
                ),
            )
    def forward(self, input_x):
        output,_ = self.pLSTM_layer0(input_x)
        for i in range(1, self.num_layers):
            output, _ = getattr(self, "pLSTM_layer"+ str(i))(output)
        return output


#Speller specified in the paper
class Speller(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            rnn_unit,
            num_layers,
            max_label_len,
            use_mlp_in_attention,
            mlp_dim_in_attention,
            mlp_activate_in_attention,
            listener_hidden_size,
            multi_head,
            decode_mode,
            use_gpu  = True,
            **kwargs,
    ):
        super(Speller, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper())
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_label_len = max_label_len
        self.decode_mode = decode_mode
        self.use_gpu = use_gpu
        self.float_type = torch.torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
        self.label_dim = vocab_size
        self.rnn_layer = self.rnn_unit(
            vocab_size+hidden_size, hidden_size, num_layers = num_layers, batch_first = True,
        )
        self.attention = Attention(
            mlp_preprocess_input = use_mlp_in_attention,
            preprocess_mlp_dim = mlp_dim_in_attention,
            activate = mlp_activate_in_attention,
            input_feature_dim = 2 * listener_hidden_size,
            multi_head = multi_head,
        )
        self.character_distribution = nn.Linear(hidden_size*2, vocab_size)
        self.softmax  = nn.LogSoftmax(dim=-1)
    #Stepwise operation of each sequence
    def forward_step(self, input_word, last_hidden_state, listener_feature):
        rnn_output, hidden_state = self.rnn_layer(input_word, last_hidden_state)
        attention_score, context = self.attention(rnn_output, listener_feature)
        concat_feature = torch.cat([rnn_output.squeeze(dim=1), context], dim=-1)
        raw_pred = self.softmax(self.character_distribution(concat_feature))
        return raw_pred, hidden_state, context, attention_score
    def forward(self, listener_feature, ground_truth=None, teacher_force_rate=0.9):
        if ground_truth is None:
            teacher_force_rate = 0
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False

        batch_size = listener_feature.size()[0]
        output_word = CreateOnehotVariable(
            self.float_type(np.zeros((batch_size, 1))), self.label_dim
        )
        if self.use_gpu:
            output_word = output_word.cuda()
        rnn_input = torch.cat([output_word, listener_feature[:,0:1,:]], dim=-1)
        hidden_state = None
        raw_pred_seq = []
        output_seq = []
        attention_record = []
        if (ground_truth is None) or (not teacher_force):
            max_step = self.max_label_len
        else:
            max_step = ground_truth.size()[1]
        for step in range(max_step):
            raw_pred, hidden_state, context, attention_score = self.forward_step(
                rnn_input, hidden_state, listener_feature
            )
            raw_pred_seq.append(raw_pred)
            attention_record.append(attention_score)
            #Teacher force - use ground truth as next step's input
            if teacher_force:
                output_word = ground_truth[:,step : step + 1, :].type(self.float_type)
            else:
                #case 0 raw output as input
                if self.decode_mode == 0:
                    output_word = raw_pred.unsqueeze(1)
                #case 1, Pick character with max probability
                elif self.decode_mode == 1:
                    output_word = torch.zeros_like(raw_pred)
                    for idx,i in enumerate(raw_pred.topk(1)[1]): # dim = 1 according to the row
                        output_word[idx, int(i)] = 1
                    output_word = output_word.unsqueeze(1)
                # case 2 sample categotical label from raw prediction
                else:
                    sampled_word = Categorical(raw_pred).sample()
                    output_word = torch.zeros_like(raw_pred)
                    for idx, i in enumerate(sampled_word):
                        output_word[idx, int(i)] = 1
                    output_word = output_word.unsqueeze(1)
            rnn_input = torch.cat([output_word, context.unsqueeze(1)], dim = -1)
        return raw_pred_seq, attention_record

#Attention mechanism
#currently only 'dot' is implemented
#Input : Decode state           with shape [batch_size, 1, decoder hidden dimension]
#        Compressed feature from listener with shape [batch_size, T, listener feature dimensuion]
#Output : Attention score           with shape [batch_size, T , (attention score of each time step)]
#         Context vector            with shape [batch_size,  listener feature dimension]
class Attention(nn.Module):
    def __init__(
            self,
            mlp_preprocess_input,
            preprocess_mlp_dim,
            activate,
            mode="dot",
            input_feature_dim = 512,
            multi_head = 1,
    ):
        super(Attention, self).__init__()
        self.mode = mode.lower()
        self.mlp_preprocess_input = mlp_preprocess_input
        self.multi_head = multi_head
        self.softmax = nn.Softmax(dim=-1)
        if mlp_preprocess_input:
            self.preprocess_mlp_dim = preprocess_mlp_dim
            self.phi = nn.Linear(input_feature_dim, preprocess_mlp_dim * multi_head)
            self.psi = nn.Linear(input_feature_dim, preprocess_mlp_dim)
            if multi_head > 1:
                self.dim_reduce = nn.Linear(input_feature_dim*multi_head, input_feature_dim)
            if activate != "None":
                self.activate = getattr(F, activate)
            else:
                self.activate = None
    def forward(self, decoder_state, listener_feature):
        if self.mlp_preprocess_input: #True
            if self.activate:
                comp_decoder_state = self.activate(self.phi(decoder_state))
                comp_listener_feature = self.activate(TimeDistributed(self.psi, listener_feature)) # output with shape [batch_size, time_steps, -1]
            else:
                comp_decoder_state = self.phi(decoder_state)
                comp_listener_feature = TimeDistributed(self.phi, listener_feature)
        else:
            comp_decoder_state = decoder_state
            comp_listener_feature = listener_feature
        if self.mode == "dot":
            if self.multi_head == 1:
                energy = torch.bmm(
                    comp_decoder_state, comp_listener_feature.transpose(1,2)
                ).squeeze(dim=1)
                attention_score = [self.softmax(energy)]
                context = torch.sum(
                    listener_feature
                    * attention_score[0].unsqueeze(2).repeat(1,1,listener_feature.size(2)),
                    dim = 1,
                )
            else:
                attention_score = [
                    self.softmax(
                        torch.bmm(att_querry, comp_listener_feature.transpose(1,2)).squeeze(dim=1)
                    )
                    for att_querry in torch.split(
                        comp_decoder_state, self.preprocess_mlp_dim, dim=-1
                    )
                ]
                projected_src = [
                    torch.sum(
                        listener_feature * att_s.unsqueeze(2).repeat(1,1, listener_feature.size(2)),
                        dim=1,
                    )
                    for att_s in attention_score
                ]
                context =  self.dim_reduce(torch.cat(projected_src, dim=-1))
        else:
            pass
        return attention_score, context




class mmWavoice(nn.Module):
    def __init__(self, listener, speller,mmwavoicenet):
        super(mmWavoice, self).__init__()
        self.listener = listener
        self.speller = speller
        self.mmWavoiceNet = mmwavoicenet

    def forward(self, voice_batch_data, mmwave_batch_data, batch_label, teacher_force_rate, is_training = True):
        voice_batch_data = voice_batch_data.unsqueeze(1)
        mmwave_batch_data = mmwave_batch_data.unsqueeze(1)
        mmWavoice_feature = self.mmWavoiceNet(voice_batch_data, mmwave_batch_data)
        mmWavoice_feature = mmWavoice_feature.squeeze(1)
        listener_feature = self.listener(mmWavoice_feature) # input : batch_data
        if is_training:
            raw_pred_seq, attention_record = self.speller(
                listener_feature, ground_truth = batch_label, teacher_force_rate=teacher_force_rate
            )
        else:
            raw_pred_seq, attention_record = self.speller(
                listener_feature, ground_truth = None, teacher_force_rate = 0
            )
        return raw_pred_seq, attention_record

    def serialize(self, optimizer, epoch, tr_loss, val_loss):
        package = {
            #encoder
            "einput": self.listener.input_feature_dim,
            "ehidden": self.listener.hidden_size,
            "elayer": self.listener.num_layers,
            "etype": self.listener.rnn_unit,
            # decoder
            "dvocab_size": self.speller.label_dim,
            "dhidden": self.speller.hidden_size,
            "dlayer": self.speller.num_layers,
            "dtype": self.speller.rnn_unit,
            #state
            "state_dict": self.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        if tr_loss is not None:
            package["tr_loss"] = tr_loss
            package["val_loss"] = val_loss
        return package



def conv3x3(in_planes, out_planes, stride = (1,1)):
    """ 3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride = stride, padding=1, bias=False)

class ECABasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride =(1,1), downsample=None):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,(1,1))
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ECABottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample = None):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes*4)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not  None :
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7,stride=1)
        #self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self,block, planes, blocks,stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes,stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim = 1)
        return x
def eca_resnet18(pretrained = False):
    """Construct a Resnet-18 model.
    Args:
    pretrained(bool): If True, returns a model pre-trained on ImageNet
    :param pretrained:
    :return:
    """
    model = ResNet(ECABasicBlock, [2, 2, 2,2])
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class rescrossSE(nn.Module):
    def __init__(self, channel, reduction = 1):
        super(rescrossSE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x1, x2):
        b, c, _, _, = x1.size()
        y1 = self.avg_pool(x1).view(b,c)
        y2 = self.avg_pool(x2).view(b,c)
        temp_y1 = y1
        y1 = self.fc(y2).view(b,c,1,1)
        y2 = self.fc(temp_y1).view(b,c,1,1)
        out1 = x1 * y1.expand_as(x1) + x1 * 0.5
        out2 = x1 * y2.expand_as(x2) + x2 * 0.5

        return out1, out2

class interattention(nn.Module):
    def __init__(self, all_channel):
        super(interattention, self).__init__()
        self.linear_e = nn.Conv2d(all_channel, all_channel,kernel_size=1, bias=False)
        self.channel = all_channel
        self.gate = nn.Conv2d(all_channel, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        self.conv = nn.Conv2d(1,1, kernel_size=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0,0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x1, x2):

        input_size = x1.size()[2:]
       # print(x1.size())
        all_dim = input_size[0] * input_size[1]
        x1_flat = x1.view(-1, x1.size()[1], all_dim) #B, C, H*W
        x2_flat = x2.view(-1, x1.size()[1], all_dim) #B,C,H*W
        x1_t = torch.transpose(x1_flat,1,2).contiguous() # batch_size * dim * num
        #x1_corr = self.linear_e(x1_t) ### conv2d
        #print(x1_t.shape)
        A = torch.bmm(x1_t, x2_flat)
        #print(A.shape)
        A1 = F.softmax(A.clone(), dim=1) # F.softmax(A.clone(), dim=1)
        B = F.softmax(torch.transpose(A, 1,2), dim=1)

        x2_att = torch.bmm(x1_flat, A1).contiguous() # S*Va = Zb #torch.bmm(x1_flat, A1).contiguous()
        x1_att = torch.bmm(x2_flat, B).contiguous() #torch.bmm(x2_flat, B).contiguous()

        x1_att = x1_att.view(-1, x1.size()[1], input_size[0], input_size[1])
        x2_att = x2_att.view(-1, x1.size()[1], input_size[0], input_size[1])

        x1_mask = self.gate_s(x1_att)
        x2_mask = self.gate_s(x2_att)
        #x1_mask = self.gate(x1_mask)
        #x2_mask = self.gate(x2_mask)
        out1 = self.gate(x1_mask * x1)
        out2 = self.gate(x2_mask * x2)
        out  = out1 + out2 # next to try element-wise product out = out1 * out2
        out = self.conv(out)
        #out = self.gate(x1_att + x2_att)
        return out

class mmWavoiceNet(nn.Module):
    def __init__(self, block,rescrossSEblock,interattentionblock,layers):
        super(mmWavoiceNet, self).__init__()
        self.inplanes = 8
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=[2,1], padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,16, layers[0]) #block(self.inplanes, self.inplanes*2)
        self.layer2 = self._make_layer(block,24, layers[1], stride=(2,1)) #block(self.inplanes*2, self.inplanes*3)
        self.layer3 = self._make_layer(block, 24, layers[2], stride=(2,1)) #block(self.inplanes*3, self.inplanes*4)
        self.rescrossSE1 = rescrossSEblock(self.inplanes)

        self.layer4 = self._make_layer(block,24, layers[3]) #block(self.inplanes*4, self.inplanes*5)
        self.layer5 = self._make_layer(block,24, layers[4]) #block(self.inplanes*5, self.inplanes*6)

        self.interattend1 = interattentionblock(self.inplanes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, block,planes, blocks=1, stride=(1,1)):
        downsample = None
        if (stride[0] != 1 and len(stride) == 2) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x2 = self.conv1(x2)

        x1 = self.bn1(x1)
        x2 = self.bn1(x2)

        x1 = self.relu(x1)
        x2 = self.relu(x2)

        x1 = self.layer1(x1)
        x2 = self.layer1(x2)

        x1 = self.layer2(x1)
        x2 = self.layer2(x2)

        x1 = self.layer3(x1)
        x2 = self.layer3(x2)

        x1, x2 = self.rescrossSE1(x1, x2)

        x1 = self.layer4(x1)
        x2 = self.layer4(x2)

        x1 = self.layer5(x1)
        x2 = self.layer5(x2)

        out = self.interattend1(x1, x2)
        return out
























