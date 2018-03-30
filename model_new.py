import torch.nn as nn
from torch.autograd import Variable
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import init

class CapGenerator(nn.Module):
    def __init__(self, emb_dim, num_words, hidden_dim):
        super(CapGenerator,self).__init__()
        self.encoder = Encoder(emb_dim, hidden_dim)
        self.decoder = Decoder(emb_dim, num_words, hidden_dim)

    def forward(self,img_inputs, caps, lengths):
        V, v_g = self.encoder(img_inputs)

        scores, _, _, _ = self.decoder(V, v_g, caps)
        packed_scores = pack_padded_sequence(scores, lengths, batch_first=True)
        return packed_scores
    def beam_search(self, img_inputs, max_len=20, beam=4):
        V, v_g = self.encoder(img_inputs)
        bsize = img_inputs.size(0)
        start = Variable(torch.LongTensor(bsize,1).fill_(1).cuda())
        caps = Variable(torch.LongTensor(bsize, beam).cuda())
        init_states = None
        states = [None] * beam
        sampled_ids = Variable(torch.LongTensor(bsize, max_len, beam).fill_(0).cuda())
        best_ids = Variable(torch.LongTensor(bsize, max_len).fill_(0).cuda())
        caps_cddt = Variable(torch.LongTensor(bsize, beam * beam).cuda())
        scores_cddt = Variable(torch.FloatTensor(bsize, beam * beam).cuda())
        sampled_scores = Variable(torch.FloatTensor(bsize,beam).cuda())
        tmp_ids = Variable(torch.LongTensor(bsize, max_len, beam).cuda())

        scores, init_states, _, _ = self.decoder(V, v_g, start, init_states)
        scores = F.softmax(scores, 2)
        top_scores, top_ids = scores.topk(beam, dim=2)
        sampled_ids[:,0,:] = top_ids.squeeze(1)
        #print(sampled_ids[:,0,:])
        sampled_scores = torch.log(top_scores.squeeze(1))
        states = [init_states] * beam
        caps = top_ids
        for i in range(1, max_len):
            tmp_ids = Variable(torch.LongTensor(bsize, max_len, beam).fill_(0).cuda())
            prev_scores = Variable(torch.FloatTensor(bsize, beam).fill_(0).cuda())
            for b in range(beam):
                scores, states[b], _, _ = self.decoder(V, v_g, caps[:,:,b], states[b])
                scores = F.softmax(scores, 2)
                top_scores, top_ids = scores.topk(beam, dim=2)
                scores_cddt[:, b*beam:(b+1)*beam] = sampled_scores[:,b].unsqueeze(1) + torch.log(top_scores.squeeze(1))
                caps_cddt[:, b*beam:(b+1)*beam] = top_ids.squeeze(1)
            #print(scores_cddt)
            #print(caps_cddt)
            crt_scores, crt_ids = scores_cddt.topk(beam, dim=1)
            #print(crt_scores)
            crt_ids = crt_ids.data
            prev_rows = crt_ids / beam
            for batch in range(bsize):
                prev_row = prev_rows[batch]
                for b in range(beam):
                    caps[batch,0, b] = caps_cddt[batch, crt_ids[batch, b]]
                    sampled_scores[batch, b] = crt_scores[batch, b]
                    #sampled_scores[batch, b] = crt_scores[batch, prev_row[b]]
                    tmp_ids[batch, :,b] = sampled_ids[batch, :, prev_row[b]]
            sampled_ids = tmp_ids
            sampled_ids[:, i, :] = caps.squeeze(1)
            #print(sampled_ids[:,0:i+1,:])
            #if i==4:
            #    raise Exception
            best_row = sampled_scores.max(1)[1].data
            #print(sampled_scores[:,:])
            #raise Exception
            for batch in range(bsize):
                best_ids[batch, :] = sampled_ids[batch, :, best_row[batch]]
        #print(sampled_scores, best_row)
        return sampled_ids[:,:,0], None, None
        

 
    def sampler(self, img_inputs, max_len=20):
        V, v_g = self.encoder(img_inputs)
        caps = Variable(torch.LongTensor(img_inputs.size(0), 1).fill_(1).cuda())
        best_ids = []
        attention = []
        Beta = []
        scores_sum = Variable(torch.FloatTensor(img_inputs.size(0), 1).fill_(0).cuda())
        #sampled_prev = [] * beam
        states = None
        for i in range(max_len):
            scores, states, atten_weights, beta = self.decoder(V, v_g, caps, states)
            scores = F.softmax(scores,2)
            best_scores, preds = scores.max(2)
            scores_sum += torch.log(best_scores)
            caps = preds
            attention.append(atten_weights)
            Beta.append(beta)
            best_ids.append(caps)
        best_ids = torch.cat(best_ids, dim=1)
        attention = torch.cat(attention, dim=1)
        Beta = torch.cat(Beta, dim=1)
        #print(scores_sum)

        return best_ids, attention, Beta


class Decoder(nn.Module):
    def __init__(self, emb_dim, num_words, hidden_dim):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(num_words, emb_dim)
        self.LSTM = nn.LSTM(emb_dim * 2, hidden_dim, 1, batch_first=True)
        self.hidden_dim = hidden_dim
        self.att_block = AdaAttenBlock(emb_dim, num_words, hidden_dim)
        
    def forward(self, V, v_g, caps, states=None):
        emb = self.embed(caps)
        x = torch.cat((emb, v_g.unsqueeze(1).expand_as(emb)), dim=2)
        batch_size, lstm_steps = x.size(0), x.size(1)
        hid = Variable(torch.zeros(batch_size, lstm_steps, self.hidden_dim)).cuda()
        mem = Variable(torch.zeros(batch_size, lstm_steps, self.hidden_dim)).cuda()

        for t in range(lstm_steps):

            x_t = x[:, t, :]
            x_t = x_t.unsqueeze(1)

            h_t, states = self.LSTM(x_t, states)

            hid[:, t, :] = h_t
            mem[:, t, :] = states[1]

        scores, atten_weights, beta = self.att_block(x, hid, mem, V)

        # why return states?
        return scores, states, atten_weights, beta

class AdaAttenBlock(nn.Module):
    def __init__(self, emb_dim, num_words, hidden_dim):
        super(AdaAttenBlock, self).__init__()
        self.sentinel = Sentinel(emb_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim)
        self.mlp = nn.Linear(hidden_dim, num_words)
        self.h2score = nn.Linear(hidden_dim, num_words)
        self.dropout = nn.Dropout(0.5)
        self.hidden_dim = hidden_dim
        self.init_weight()

    def init_weight(self):
        init.kaiming_uniform(self.mlp.weight, mode='fan_in')
        self.mlp.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data

        # create tensor dim=[1, bsz, hidden_dim] and data_type same with weight
        h0 = Variable(weight.new(1, bsz, self.hidden_dim).zero_().cuda())
        h1 = Variable(weight.new(1, bsz, self.hidden_dim).zero_().cuda())
        return (h0, h1)

    def forward(self, x, hid, mem, V):
        batch_size = x.size(0)
        h0 = self.init_hidden(batch_size)[0].transpose(0, 1)

        if hid.size(1) > 1:
            hid_t_1 = torch.cat((h0, hid[:,:-1,:]),dim=1)
        else:
            hid_t_1 = h0

        sentinel = self.sentinel(x, hid_t_1, mem)
        c_hat, atten_weights, beta = self.att(V, hid, sentinel)
        #scores = self.mlp(self.dropout(c_hat + hid))
        scores = self.mlp(self.dropout(c_hat)) + self.h2score(self.dropout(hid))

        return scores, atten_weights, beta

class Sentinel(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Sentinel, self).__init__()
        self.h2s = nn.Linear(input_size, hidden_dim, bias=False)
        self.x2s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.init_weight()

    def forward(self, x_t, h_t_1, mem_t):
        gate_s = F.sigmoid(self.x2s(self.dropout(x_t)) + self.h2s(self.dropout(h_t_1)))
        s = gate_s * F.tanh(mem_t)

        return s

    def init_weight(self):
        init.xavier_uniform(self.h2s.weight)
        init.xavier_uniform(self.x2s.weight)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.h2att = nn.Linear(hidden_dim, 49, bias=False)
        self.h2att2 = nn.Linear(hidden_dim, 49, bias=False)
        self.v2att = nn.Linear(hidden_dim, 49, bias=False)
        self.s2att = nn.Linear(hidden_dim, 49, bias=False)
        self.attW = nn.Linear(49, 1, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        init.xavier_uniform(self.v2att.weight)
        init.xavier_uniform(self.h2att.weight)
        init.xavier_uniform(self.h2att2.weight)
        init.xavier_uniform(self.s2att.weight)
        init.xavier_uniform(self.attW.weight)

    def forward(self, V, h_t, s_t):
        content_v = self.v2att(self.dropout(V)).unsqueeze(1) + self.h2att(self.dropout(h_t)).unsqueeze(2)
        z_v = self.attW(self.dropout(F.tanh(content_v))).squeeze(3)
        content_s = self.s2att(self.dropout(s_t)) + self.h2att2(self.dropout(h_t))
        z_s = self.attW(self.dropout(F.tanh(content_s)))
        z_total = torch.cat((z_v, z_s), dim=2)
        alpha_hat = F.softmax(z_total, 2)
        beta = alpha_hat[:, :, -1].unsqueeze(2)
        alpha = alpha_hat[:, :, :-1]

        c_hat = beta * s_t + torch.bmm(alpha, V)

        return c_hat, alpha*(1-beta), beta


class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        resnet_conv = nn.Sequential(*modules)

        self.resnet_conv = resnet_conv
        self.avgpool = nn.AvgPool2d(7)
        self.ai2v = nn.Linear(2048, hidden_dim)
        self.ag2v = nn.Linear(2048, emb_dim)
        self.dropout = nn.Dropout()
        self.init_weight()

    def init_weight(self):
        init.kaiming_uniform(self.ai2v.weight, mode='fan_in')
        init.kaiming_uniform(self.ag2v.weight, mode='fan_in')
        self.ai2v.bias.data.fill_(0)
        self.ag2v.bias.data.fill_(0)

    def forward(self, img_inputs):
        features = self.resnet_conv(img_inputs)
        batch_size = features.shape[0]
        channels = features.shape[1]
        ag = self.avgpool(features).view(batch_size, -1)
        ai = features.view(batch_size, channels, -1).transpose(1, 2)
        V = F.relu(self.ai2v(self.dropout(ai)))
        v_g = F.relu(self.ag2v(self.dropout(ag)))

        return V, v_g
