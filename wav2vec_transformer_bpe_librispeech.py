# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:39:36 2020

@author: a-kojima
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import numpy as np
import os
import sentencepiece as sp
import fairseq
import math

torch.autograd.set_detect_anomaly(True)



def get_enc_mask(enc_length, target_length):
    # 0 means removing
    mask = torch.ones(len(enc_length), max(target_length), max(enc_length))
    for index, i in enumerate(enc_length):
        mask[index, target_length[index]:, i:] = 0    
    return mask.to(DEVICE)


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1))
    return subsequent_mask.repeat(1, 1, 1) .to(DEVICE)


def get_mask(len_array_):
    # 0 means signal section    
    mask_hh = torch.ones(len(len_array_), max(len_array_))    
    for i, len_ in enumerate(len_array_):
        mask_hh[i, :len_] = 0        
    return mask_hh.to(DEVICE)


def init_weight(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(-0.1, 0.1)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)
    if classname.find('LSTM') != -1:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            if 'bias' in name:
                param.data.fill_(0)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if isinstance(m.bias, nn.parameter.Parameter):
            m.bias.data.fill_(0)
            
class PositionwiseFeedForward(nn.Module):
    def __init__(self, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(HEAD * HEAD_DIM, DENSE_FF)
        self.w_2 = nn.Linear(DENSE_FF, HEAD * HEAD_DIM)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(HEAD * HEAD_DIM)

    def forward(self, x):       
        residual = x                
        x = self.layer_norm(x)
        output = self.dropout(F.relu(self.w_1(x)))        
        output = self.dropout2(self.w_2(output))
        output = output + residual
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_):
        length = input_.size(1)
        return self.pe[:, :length]
            
class Transformer_Encoder_block(nn.Module):
    def __init__(self):
        super(Transformer_Encoder_block, self).__init__()
        
        self.layer_norm = nn.LayerNorm(INPUT_DIM)
        self.positional_emb = PositionalEncoding(INPUT_DIM, 5000)    
        self.dropout = nn.Dropout(DROP)
        
        self.Q = nn.Linear(INPUT_DIM, HEAD_DIM * HEAD)
        self.V = nn.Linear(INPUT_DIM, INPUT_DIM)
        self.K = nn.Linear(INPUT_DIM, HEAD_DIM * HEAD)            
        self.feed_forward = PositionwiseFeedForward()        
        self.dropout = nn.Dropout(DROP)                
    
    def forward(self, emb, mask, enc_mask):
        
        B, T, _ = emb.size()        
        
        # ================================        
        # layer norm
        # ================================        
        residual = emb        
        emb = self.layer_norm(emb)                
        emb = self.dropout(emb + self.positional_emb(emb))

        # ================================        
        # multi head attention
        # ================================        
        q_out = self.Q(emb)
        v_out = self.V(emb)
        k_out = self.K(emb)

        q_out2 = q_out.view(B, T, HEAD, HEAD_DIM)
        k_out2 = k_out.view(B, T, HEAD, HEAD_DIM)
        v_out2 = v_out.view(B, T, HEAD, INPUT_DIM//HEAD)                
        
        q_out3 = q_out2.permute(2, 0, 1, 3).contiguous().view(-1, T, HEAD_DIM) # (B * H) * T * d_q
        v_out3 = v_out2.permute(2, 0, 1, 3).contiguous().view(-1, T, INPUT_DIM//HEAD) # (B * H) * T * d_v
        k_out3 = k_out2.permute(2, 0, 1, 3).contiguous().view(-1, T, HEAD_DIM) # (B * H) * T * d_k
        
        energy = torch.bmm(q_out3, k_out3.transpose(1, 2))        
        energy.masked_fill_(mask.repeat(HEAD, 1, 1).to(DEVICE), -float('inf'))
        
        attention = F.softmax(energy / np.sqrt(HEAD_DIM), dim=2) # (HEAD * B) * T * T
        output = torch.bmm(attention, v_out3)
        
        output = output.view(HEAD, B, T, INPUT_DIM//HEAD)
        output = output.permute(1, 2, 0, 3).contiguous().view(B, T, -1) # b x lq x (n*dv)
        output.masked_fill_(enc_mask.to(DEVICE), -float(0))
        output = output + residual
        output.masked_fill_(enc_mask.to(DEVICE), -float(0))
        output = self.feed_forward(output)
        output.masked_fill_(enc_mask.to(DEVICE), -float(0))
        output = self.dropout(output)        
        return output
    

# ========================
# Transformer encoder
# =======================
class Transformer_Encoder(nn.Module):
    def __init__(self):
        super(Transformer_Encoder, self).__init__()
        # embedding
        self.input_embedding = nn.Linear(WAV2VEC_DIM, INPUT_DIM)
        
        # encoder block
        self.layer_stack = nn.ModuleList([
                Transformer_Encoder_block()
            for _ in range(TRANSFORMER_BLOCK)])
        
    def forward(self, x, length_enc_):    
        
        # x: B * T * F
        # ===================================================
        # make mask 
        # ===================================================
        # self attention B * T * T
        B, max_seq, feature_dim = x.size()
        self_attention_mask = torch.zeros((B, max_seq, max_seq), requires_grad=False)     
        for po in range(0, len(length_enc_)):
            self_attention_mask.data[po, int(length_enc_[po]):, int(length_enc_[po]):] = 1.0                
        self_attention_mask = self_attention_mask.data.bool()
        
        # encoder output mask
        encoder_mask = torch.zeros((B, max_seq, HEAD_DIM * HEAD), requires_grad=False)     
        for po in range(0, len(length_enc_)):
            encoder_mask.data[po, int(length_enc_[po]):, :] = 1.0                
        encoder_mask = encoder_mask.data.bool()
        
        # ===================================================
        # step1. embedding
        # ===================================================
        input_ = self.input_embedding(x)
        
        # ===================================================
        # step2. block 
        # ===================================================
        for enc_layer in self.layer_stack:
            input_ = enc_layer(input_, self_attention_mask, encoder_mask)     
            
        return input_


class CTCModel(nn.Module):
    def __init__(self):        
        super(CTCModel, self).__init__()
        # wav2vec encoder
        checkpoint = torch.load(CP_PATH)       
        # freeeze         
        if IS_REGULARIZE == False:
            checkpoint['args'].mask_prob = 0
            checkpoint['args'].activation_dropout = 0
            checkpoint['args'].attention_dropout = 0
            checkpoint['args'].dropout = 0
            checkpoint['args'].dropout_features = 0
            checkpoint['args'].dropout_input = 0
            checkpoint['args'].encoder_layerdrop = 0
            checkpoint['args'].pooler_dropout = 0
        
        cfg = fairseq.dataclass.utils.convert_namespace_to_omegaconf(checkpoint['args'])        
        wav2vec2_encoder = fairseq.models.wav2vec.Wav2Vec2Model.build_model(cfg.model)
        wav2vec2_encoder.load_state_dict(checkpoint['model'])
        self.wav2vec2_encoder = wav2vec2_encoder        
        
        # decoder         
        self.transformer = Transformer_Encoder()
        
        self.output_layer = nn.Linear(DENSE_DIM1, NUM_CLASSES, bias=False)

    def forward(self, padded_x, x_len, target_seq, target_seq_mask, target_len_):
        # waveform mask
        mask_ = get_mask(x_len)        
        # feed wav2vec model
        hh = self.wav2vec2_encoder(padded_x, features_only=True, padding_mask=mask_)    
        h = hh['x']
        padded_mask = hh['padding_mask'] 
        wav2vec_input_length_array = (1 - padded_mask.long()).sum(-1)        
        decoder_out = self.transformer(h, wav2vec_input_length_array)        
        prediction = self.output_layer(decoder_out)
        return prediction, wav2vec_input_length_array

def train_epoch(model, optimizer, training_data, learning_rate):

    acc_utts = 0
    global global_optimize
    
    
    while acc_utts < len(training_data):
        xs = []
        ts = []
        batch_size = 0
        
        while batch_size < BATCH_SIZE and acc_utts < len(training_data):
            
                split_info = training_data[acc_utts].replace('\n', '')
                
                #wav_path = BASE_PATH + '\\' + split_info.split('\t')[1]
                
                wav_path = BASE_PATH + '/' + split_info.split('\t')[1]

                trans = split_info.split('\t')[0]

            
                # this is shape is  frame * mel_dim
                if os.path.exists(wav_path) == True:
                    print(wav_path)
                    cpudat = sf.read(wav_path)[0]
                    cpulab = sp_bpe.encode_as_ids(trans)
                                        
                    # calculate wav2vec length
                    wav2vec_length = ((len(cpudat) - WAV2VEC_FRAME) // WAV2VEC_SHIFT) + 1
                    
                    # avoid memory error
                    if wav2vec_length <= MAX_SAMPLE and len(cpulab) <= MAX_LABEL \
                                and wav2vec_length >= len(cpulab) * 2 and MIN_SAMPLE <= wav2vec_length \
                                and np.sum(cpudat) != 0 and len(cpulab) >=3 and np.isnan(cpudat).any() == False:
                    
                        xs.append(torch.tensor(cpudat, device = DEVICE).float())
                        ts.append(torch.tensor(cpulab, device = DEVICE).long())
            
                        batch_size += 1
                    else:
                        print('pass', wav_path)
                    pass
    
                acc_utts += 1
         
        if len(xs) != BATCH_SIZE:
            break        

        xs_lengths = torch.tensor(np.array([len(x) for x in xs], dtype=np.int32), device = DEVICE)
        ts_lengths = torch.tensor(np.array([len(t) for t in ts], dtype=np.int32), device = DEVICE)
        
            

        sorted_xs_lengths, perm_index = xs_lengths.sort(0, descending = True)
        sorted_ts_lengths = ts_lengths[perm_index]
        
        
        padded_xs = nn.utils.rnn.pad_sequence(xs, batch_first = True) # NxTxD
        padded_ts = nn.utils.rnn.pad_sequence(ts, batch_first = True) # NxT

        padded_sorted_xs = padded_xs[perm_index] # NxTxD
        padded_sorted_ts = padded_ts[perm_index] # NxT
        
        # make data for transformer decoder        
        target_mask = get_subsequent_mask(padded_sorted_ts)

        ctc_prediction, wav2vec_seq_len = model(padded_sorted_xs, sorted_xs_lengths, padded_sorted_ts, target_mask, sorted_ts_lengths)                
        
        loss = F.ctc_loss(F.log_softmax(ctc_prediction, dim=2).transpose(0,1),
                             padded_sorted_ts,
                             wav2vec_seq_len.tolist(),
                             sorted_ts_lengths.tolist(),
                             blank = BLANK)
        
        print(loss)
        loss.backward()

        
        clip = 5.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        model.zero_grad() 
        
        optimizer.zero_grad()
        
        
        
        
        if INCREASE_STEP+1 >= global_optimize:        
            learning_rate = np.min((learning_rate + DIFF_LR1, TARGET_LR1))                       
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        global_optimize+=1
                            
        # start to train transformer layers 
        if global_optimize == CLASSIFICAION_UPDATE:
            for param in model.transformer.parameters():
                param.requires_grad = True            
                
    return learning_rate, optimizer

if __name__ == '__main__':

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    IS_REGULARIZE = False
    
    # ==================================================================
    # ASR model
    # ==================================================================
    WAV2VEC_DIM = 768
    NUM_CLASSES = 128    
    HEAD = 4
    HEAD_DIM = 32 # 
    INPUT_DIM = HEAD * HEAD_DIM
    DROP = 0.1
    TRANSFORMER_BLOCK = 4
    DENSE_DIM1 = 128
    DENSE_FF = INPUT_DIM * 4

    # ==================================================================
    # training stategy
    # ==================================================================    
    BATCH_SIZE = 1 #20
    FS = 16000
    # number of wav2vec frames
    MAX_SAMPLE = 1000
    MIN_SAMPLE = -float('inf')
    WAV2VEC_SHIFT = int(0.02 * FS)
    WAV2VEC_FRAME = int(0.025 * FS)
    MAX_LABEL = 200
    
    # please download from it
    # https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt    
    CP_PATH = '../wav2vec_small.pt'    

    FIRST_LR = 0
    TARGET_LR1 = 1e-3
    TARGET_LR2 = 2.5e-6
    
    script_file1 = r'transcription.txt'
    training_data = [line for line in open(script_file1, encoding='utf-8')]
    
        
    INCREASE_STEP = 5000
    
    CLASSIFICAION_UPDATE = 10000
    DIFF_LR1 = (TARGET_LR1 - FIRST_LR) / INCREASE_STEP
    
    BLANK = 0
    
    
    global global_optimize
    global_optimize = 0
    
        
    # =============================
    # dictionary
    # =============================           
    sp_bpe = sp.SentencePieceProcessor()
    sp_bpe.load('./vocab_model/256m.model')
    BASE_PATH = r'./wav'
    

    # ==================================================================
    # min param
    # ==================================================================    

    model = CTCModel().to(DEVICE)
    model.train()
    
    # freeze weight
    for param in model.wav2vec2_encoder.parameters():
        param.requires_grad = False
        
    # initialize transformer layers
    for param in model.transformer.parameters():
        param.requires_grad = False    
    
#    global learning_rate
    learning_rate = FIRST_LR
    
#    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.005)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-6)
    
    
    for epoch in range(0, 20):        
        learning_rate, optimizer = train_epoch(model, optimizer, training_data, learning_rate)
        if epoch >= 18:
            learning_rate = learning_rate * 0.85
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        
    
        torch.save(model.state_dict(), "./model_save_{}".format(epoch + 1))
