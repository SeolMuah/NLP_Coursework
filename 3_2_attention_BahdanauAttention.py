"""
    Attention 101 > BahdanauAttention

        - this code is for educational purpose.
        - the code is written for easy understanding not for optimized code.

    Author : Sangkeun Jung (hugmanskj@gmail.com)
    All rights reserved. (2021)
"""

   
# In this code, we will implement
#   - Badanau attention mechanism which is the first attention algorithm.
#   - Note that Badanau attention mechanism is one of the additive attention mechanism.


import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Attention > Additive Attention > Bahdanau approach 

    Inputs:
        query_vector  : [hidden_size]
        multiple_items: [batch_size, num_of_items, hidden_size]
    Returns:
        blendded_vector:    [batch_size, item_vector hidden_size]
        attention_scores:   [batch_size, num_of_items]
    """
    def __init__(self, item_dim, query_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.item_dim = item_dim            # dim. of multiple item vector
        self.query_dim = query_dim          # dim. of query vector
        self.attention_dim = attention_dim  # dim. of projected item or query vector

        # W is used for project query to the attention dimension
        # U is used for project each item to the attention dimension
        self.W = nn.Linear(self.query_dim,  self.attention_dim, bias=False) #쿼리를 어텐션 차원으로 바꿈
        self.U = nn.Linear(self.item_dim,   self.attention_dim, bias=False) #아이템을 어텐션 차원으로 바꿈
        
        # v is used for calculating attention score which is scalar value
        self.v = nn.Parameter(torch.randn(1, attention_dim, dtype=torch.float)) #torch.Size([1, 512])
        
      

    def _calculate_reactivity(self, query_vector, multiple_items):
        B, N, H = multiple_items.shape  # [B,N,H]

        #D는 어텐션 벡터 차원
        # linear projection is applied to the last dimension
        query_vector    = query_vector.unsqueeze(1) #torch.Size([200, 512]) => torch.Size([200, 1, 512])
        projected_q     = self.W(query_vector)      # [B,1,Q] --> [B, 1, D] in case of Q=D,  torch.Size([200, 1, 512]) => torch.Size([200, 1, 512]), 어텐션 차원으로 바꿔줌
        projected_item  = self.U(multiple_items)    # [B,N,H] --> [B, N, D] in case of H=D,  torch.Size([200, 12, 512]) => torch.Size([200, 12, 512]), 어텐션 차원으로 바꿔줌

        # note that broadcasting is performed when adding different shape
        added_itmes = projected_q + projected_item  # [B, 1, D] + [B, N, D] --> [B, N, D] 아이템 벡터 모두에 쿼리 벡터를 더함
        tanh_items  = torch.tanh(added_itmes)       # [B,N,D] -1 ~ 1 로 정규화
       
        v_t = self.v.transpose(1,0) ##torch.Size([1, 512]) => #torch.Size([512, 1])
        batch_v = v_t.expand(B,self.attention_dim,1)        # [B, D, 1] torch.Size([200, 512, 1])
        
        #mm은 행렬간 곱, bmm은 배치가 있는 경우 사용
        reactivity_scores = torch.bmm(tanh_items, batch_v)  # [B,N,D] x [B,D,1] --> [B, N, 1] #각 아이템별로 v_t를 닷프로덕트 진행
        reactivity_scores = reactivity_scores.squeeze(-1)   # [B, N, 1] --> [B, N]
        return reactivity_scores #[B, N]

    def forward(self, query_vector, multiple_items, mask):
        """
        Inputs:
            query_vector:   [query_vector hidden_size] #torch.Size([200, 512])
            multiple_items: [batch_size, num_of_items, item_vector hidden_size] #torch.Size([200, 12, 512])
            mask:           [batch_size, num_of_items]  1 for valid item, 0 for invalid item #torch.Size([200, 12])
        Returns:
            blendded_vector:    [batch_size, item_vector hidden_size]
            attention_scores:   [batch_size, num_of_items]
        """
        assert mask is not None, "mask is required"

        # B : batch_size, N : number of multiple items, H : hidden size of item
        B, N, H = multiple_items.size() 
        
        # Three Steps
        # 1) [reactivity] try to check the reactivity with ( item_t and query_vector ) N times
        # 2) [masking]    try to penalize invalid items such as <pad>
        # 3) [attention]  try to get proper attention scores (=propability form) over the reactivity scores
        # 4) [blend]      try to blend multiple items with attention scores 

        # Step-1) reactivity
        reactivity_scores = self._calculate_reactivity(query_vector, multiple_items) #쿼리와 아이템 합침
        #torch.Size([200, 12]) 
        
        # Step-2) masking
        # The mask marks valid positions so we invert it using `mask & 0`.
        # detail : check the masked_fill_() of pytorch 
        reactivity_scores.data.masked_fill_(mask == 0, -float('inf'))

        # Step-3) attention score
        attention_scores = F.softmax(reactivity_scores, dim=1) # over the item dimensions
        #torch.Size([200, 12]) #위에서 구한 반응성이 큰것이 높은 비율을 갖게됨 (모든 비율의 합은 1)
        
        # Step-4) blend multiple items
        # merge by weighted sum
        attention_scores = attention_scores.unsqueeze(1) # [B, 1, #_of_items] #torch.Size([200, 1, 12]) 

        # [B, 1, #_of_items] * [B, #_of_items, dim_of_item] --> [B, 1, dim_of_item]
        ##torch.Size([200, 1, 12]) matmul torch.Size([200, 12, 512])
        #matmul은 브로드캐스트
        blendded_vector = torch.bmm(attention_scores, multiple_items) #torch.Size([200, 1, 512])
        blendded_vector = blendded_vector.squeeze(1) # [B, dim_of_item]  #torch.Size([200, 512])

        return blendded_vector, attention_scores



## ------------------------------------------------------------------------ ##
## Training and Testing with toy dataset                                    ##
## ------------------------------------------------------------------------ ##
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np 

def load_data(fn):
    data = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            seq_str, query, y = line.split('\t')
            seqs = seq_str.split(',')
            data.append( (seqs, query, y) )
    return data

# you can define any type of dataset
# dataset : return an example for batch construction. 
class NumberDataset(Dataset):
    """Dataset."""

    def __init__(self, fn, input_vocab, output_vocab, max_seq_length):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.max_seq_length = max_seq_length 
        
        # load 
        self.data = load_data(fn)

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx): 
        seq, q, y = self.data[idx]

        # [ input ]
        seq_ids = [ self.input_vocab[t] for t in seq ]

        # <pad> processing
        pad_id = self.input_vocab['<pad>']
        num_to_fill = self.max_seq_length - len(seq)
        seq_ids = seq_ids + [pad_id]*num_to_fill

        # mask processing (1 for valid, 0 for invalid)
        weights = [1]*len(seq) + [0]*num_to_fill


        # ex) 
        # seq_ids : 6, 3, 5, 2, 4, _, _, _
        # weights : 1, 1, 1, 1, 1, 0, 0, 0

        # [ query ]
        # NOTE : we assume that query vocab space is same as input vocab space

        q_id = self.input_vocab[q] # enable valid query 
        #q_id = 0 # disable query -- to check query effect in attention mechanism

        # [ ouput ] 
        y_id = self.output_vocab[y]

        item = [
                    # input
                    np.array(seq_ids),
                    q_id,
                    np.array(weights),

                    # output
                    y_id
               ]
        return item 


class NumberDataModule(pl.LightningDataModule):
    def __init__(self, 
                 max_seq_length: int=12,
                 batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length 

        input_vocab, output_vocab = self.make_vocab('./data/numbers/train.txt') #값에 대한 아이디 딕셔너리
        self.input_vocab_size  = len( input_vocab )
        self.output_vocab_size = len( output_vocab )
        self.padding_idx = input_vocab['<pad>']

        self.all_train_dataset = NumberDataset('./data/numbers/train.txt', input_vocab, output_vocab, max_seq_length)
        self.test_dataset      = NumberDataset('./data/numbers/test.txt', input_vocab, output_vocab, max_seq_length)

        self.input_r_vocab  = { v:k for k,v in input_vocab.items() } #아이디에 대한 값 딕셔너리
        self.output_r_vocab = { v:k for k,v in output_vocab.items() }

        # random split train / valiid for early stopping
        N = len(self.all_train_dataset)
        tr = int(N*0.8) # 8 for the training
        va = N - tr     # 2 for the validation 
        self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.all_train_dataset, [tr, va])
  

    def make_vocab(self, fn):
        data = load_data(fn)
        
        input_tokens = sorted(list(set([x for seqs, query, y in data for x in seqs])))
        output_tokens = sorted(list(set([y for seqs, query, y in data])))
        # for seqs, query, y in data:
        #     for token in seqs:
        #         input_tokens.append(token)
        #     output_tokens.append(y)
        
        # input_tokens = list(set(input_tokens))
        # output_tokens = list(set(output_tokens)) 

        # input_tokens.sort()
        # output_tokens.sort()

        # [input vocab]
        # add <pad> symbol to input tokens as a first item
        input_tokens = ['<pad>'] + input_tokens 
        input_vocab = { str(token):index for index, token in enumerate(input_tokens) }

        # [output voab]
        output_vocab = { str(token):index for index, token in enumerate(output_tokens) }

        return input_vocab, output_vocab

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # NOTE : Shuffle

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

from torchmetrics import functional as FM

class Attention_Number_Finder(pl.LightningModule): 
    def __init__(self, 
                 # network setting
                 input_vocab_size,
                 output_vocab_size,
                 d_model,      # dim. in attemtion mechanism 
                 padding_idx,
                 # optiimzer setting
                 learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  


        # note 
        #   - the dimension for query and multi-items do not need to be same. 
        #   - for simplicity, we make all the dimensions as same. 

        # symbol_number_character to vector_number
        self.digit_emb = nn.Embedding(self.hparams.input_vocab_size, #num_embeddings : 임베딩을 할 단어들의 개수. 다시 말해 단어 집합의 크기입니다.
                                      self.hparams.d_model, #embedding_dim : 임베딩 할 벡터의 차원입니다. 사용자가 정해주는 하이퍼파라미터입니다. (단어당 벡터의 크기)
                                      padding_idx=self.hparams.padding_idx) #padding_idx : 선택적으로 사용하는 인자입니다. 패딩을 위한 토큰의 인덱스를 알려줍니다.
                                                                            #padding_idx do not contribute to the gradient,  the embedding vector at padding_idx is not updated during training

        
        #인코딩 시퀀스(전체 x입력)를 피처 벡터로 만드는 과정 인코딩
        # sequence encoder using RNN
        #모델은 거의 2층으로만 사용하는데, 3층 이상으로 레이어를 쌓을 수록 효과가 거의 미미
        self.encoder = nn.LSTM(d_model, int(self.hparams.d_model/2), #  두개의 히든벡터가 합쳐짐 why? since bidirectional LSTM
                            num_layers=2, 
                            bidirectional=True,
                            batch_first=True
                          )#인풋 벡터의 크기, 히든 벡터의 크기
        
        #LSTM OUPUT 결과는 HiddenState, 셀상태 Cellstate
        #LSTM의 핵심 Key는 diagram의 상단을 통과하는 수평선인 Cell State, 
        # Cell State는 일종의 컨베이어 벨트와 같다.
        # 전체 Chain을 따라 직진하며, 약간의 작은 선형 상호작용이 있을 뿐이다.
        # 이를 통해, 정보는 변하지 않으며, 그저 흘러가기가 매우 쉽다.
        # self.encoder = nn.LSTM(d_model, self.hparams.d_model, #  두개의 히든벡터가 합쳐짐 why? since bidirectional LSTM
        #             batch_first=True
        #             )#인풋 벡터의 크기, 히든 벡터의 크기
                
        # attention mechanism
        self.att = BahdanauAttention(item_dim=self.hparams.d_model,
                                     query_dim=self.hparams.d_model,
                                     attention_dim=self.hparams.d_model)

        # [to output]
        self.to_output = nn.Linear(self.hparams.d_model, self.hparams.output_vocab_size) # D -> a single number

        # loss
        #-log(exp(x[class])/SIGMA_j[exp(x_j)]) #x가 커지면 손실 0 x가 작아지면 손실 무한대로 발산
        self.criterion = nn.CrossEntropyLoss() 
        

    def forward(self, seq_ids, q_id, weight):
        # ------------------- ENCODING with ATTENTION -----------------#
        #B:배치사이즈, T:문장의 길이, D:단어의 사이즈
        # [ Digit Character Embedding ]
        # seq_ids : [B, max_seq_len] = 배치 사이즈 개수 만큼 [숫자 아이디가 있는 시퀀스, 길이 :max_seq_len]
        seq_embs = self.digit_emb(seq_ids.long()) # [B, max_seq_len, emb_dim] #emb_dim : 숫자하나당 벡터 크기
        #torch.Size([200, 12, 512])
        
        # [ Sequence of Numbers Encoding ]
        seq_encs, hidden = self.encoder(seq_embs) # [B, max_seq_len, enc_dim*2]  since we have 2 layers
        #seq_encs : LSTM 토근별 모든 아웃풋, torch.Size([200, 12, 512]) #
        #hidden[0] : LSTM 마지막 층의 히든층 값, torch.Size([4, 200, 256]) #4인 이유는 layer2개 및 양방향 2개 총 4개
        #seq_encs[0][-1][:256] 마지막 히든층값 == hidden[0][2][0], seq_encs[0][0][:256] 첫번째 히든층값 == hidden[0][3][0] *이해하기 어려움
        
        #<LSTM layer1이고, 양방향이 아닌 경우hidden[0][0][0]==seq_encs[0][-1]과 같음>
        #hidden[1] : LSTM 마지막 층의 CellState 값(이것을 통해 히든값 즉, 아웃풋을 만들어냄), torch.Size([4, 200, 256]) 
        
     
        # with query (context)
        query = self.digit_emb(q_id) # [B, query_dim]
        #q_id.shape == 200
        #query.shape == torch.Size([200, 512])
        
        # dynamic encoding-summarization (blending)
        multiple_items = seq_encs

        blendded_vector, attention_scores = self.att(query, multiple_items, mask=weight) # [B, #_of_items]
        # blendded_vector  : [B, dim_of_sequence_enc] torch.Size([200, 512]) => 12개의 글자를 어텐션 스코어를 통해 하나로 합침
        # attention_scores : [B, query_len, key_len]  #torch.Size([200, 1, 12])

        # To output
        logits = self.to_output(blendded_vector)
        #to_ouput == Linear(in_features=512, out_features=9, bias=True)
        #torch.Size([200, 9])
        
        return logits, attention_scores

    def training_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch 
        logits, _ = self(seq_ids, q_id, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long()) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # y_id[0]==tensor(7), 
        # logits[0]==tensor([ 0.0486, -0.0049, -0.0071,  0.0759,  0.0287, -0.0194,  0.0182,  0.0448, -0.0062],
        # # all logs are automatically stored for tensorboard
        return loss

    def validation_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch 

        logits, _ = self(seq_ids, q_id, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long()) 
        
        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def validation_step_end(self, val_step_outputs):
        val_acc  = val_step_outputs['val_acc'].cpu()
        val_loss = val_step_outputs['val_loss'].cpu()

        self.log('validation_acc',  val_acc, prog_bar=True)
        self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        seq_ids, q_id, weights, y_id = batch 

        logits, _ = self(seq_ids, q_id, weights)  # [B, output_vocab_size]
        loss = self.criterion(logits, y_id.long()) 
        
        ## get predicted result
        prob = F.softmax(logits, dim=-1)
        acc = FM.accuracy(prob, y_id)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics, on_epoch=True)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ATTENTION")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parent_parser

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = [10, 8]
def check_attention(model, ex, input_vocab, output_vocab):

    seq_ids, q_id, weights, y_id = ex
    seq_ids = seq_ids.to(model.device)
    q_id = q_id.to(model.device)
    weights = weights.to(model.device)

    import os 
    os.makedirs('./output_figs/Bahdanau', exist_ok=True)

    import pandas as pd
    # predictions
    with torch.no_grad():
        logits, att_scores = model(seq_ids, q_id, weights)  # [B, output_vocab_size]
    
        prob = F.softmax(logits, dim=-1)
        y_id_pred = prob.argmax(dim=-1)

        for idx, (a_seq_ids, a_q_id, a_weights, a_y_id, a_y_id_pred, a_att_scores) in enumerate( zip(seq_ids, q_id, weights, y_id, y_id_pred, att_scores) ):
            N =  a_weights.sum().item()

            input_sym = [ input_vocab[i.item()] for i in a_seq_ids[:N] ]
            q_sym = input_vocab[a_q_id.item()]

            ref_y_sym = output_vocab[a_y_id.item()]
            pred_y_sym = output_vocab[a_y_id_pred.item()]

            scores = a_att_scores.cpu().detach().numpy()[0][:N].tolist() 

            ## heatmap
            data = { 'scores':[] }
            for word, score in zip(input_sym, scores):
                data['scores'].append( score )
            df = pd.DataFrame(data)
            df.index = input_sym
  
            plt.figure()
            #sns.set(rc = {'figure.figsize':(2,8)})
            sns.heatmap(df, cmap='RdYlGn_r')
            plt.title(f'Finding the first larger value than query={q_sym}, ref={ref_y_sym}, pred={pred_y_sym}', fontsize=10)
            plt.savefig(os.path.join('./output_figs/Bahdanau', f'{idx}.png'))


from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--d_model',    default=512, type=int)  # dim. for attention model 

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Attention_Number_Finder.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = NumberDataModule.from_argparse_args(args)
    iter(dm.train_dataloader()).next() # <for testing 


    # ------------
    # model
    # ------------
    model = Attention_Number_Finder(dm.input_vocab_size,
                                    dm.output_vocab_size,
                                    args.d_model,       # dim. in attemtion mechanism 
                                    dm.padding_idx,
                                    args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
                            max_epochs=60, 
                            callbacks=[EarlyStopping(monitor='val_loss')],
                            gpus = 1 # if you have gpu -- set number, otherwise zero
                        )
    trainer.fit(model, datamodule=dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=dm.test_dataloader())
    print(result)

    #{'test_acc': 0.9039999842643738, 'test_loss': 0.2998247742652893}

    # ------------
    # Check the attention scores to attend on multiple items
    # ------------
    #model = Attention_Number_Finder.load_from_checkpoint('./lightning_logs/version_15/checkpoints/epoch=0-step=179.ckpt').to('cuda:0')
    ex_batch = iter(dm.test_dataloader()).next()
    check_attention(model, ex_batch, dm.input_r_vocab, dm.output_r_vocab)


if __name__ == '__main__':
    cli_main()