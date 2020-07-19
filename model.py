import torch.nn as nn
import torch.nn.functional as F
import torch


class WordAVGModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim,dropout=0.2, pad_idx=0):
        # 初始化参数，
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # embedded.shape = (batch_size,seq,embed_size)
        embedded = self.dropout(self.embedding(text))

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        # [batch size, embedding_dim] 把单词长度的维度压扁为1，并降维

        return self.fc(pooled)
        # （batch size,output_dim）

class LSTM(nn.Module):
    def __init__(self, vocab_list, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.2, pad_idx=0):
        super().__init__()
        self.num_input = len(vocab_list)
        
        self.ad_embedding = torch.nn.Embedding(num_embeddings=len(vocab_list[0].vocab), embedding_dim=300, padding_idx=0)
        self.advertiser_embedding = torch.nn.Embedding(num_embeddings=len(vocab_list[1].vocab), embedding_dim=300, padding_idx=0)
        self.creative_embedding = torch.nn.Embedding(num_embeddings=len(vocab_list[2].vocab), embedding_dim=300, padding_idx=0)
        self.industry_embedding = torch.nn.Embedding(num_embeddings=len(vocab_list[3].vocab), embedding_dim=300, padding_idx=0)
        self.product_embedding = torch.nn.Embedding(num_embeddings=len(vocab_list[4].vocab), embedding_dim=300, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim * self.num_input, hidden_dim, num_layers=n_layers,batch_first=True,
                           bidirectional=bidirectional)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # 这里hidden_dim乘以2是因为是双向，需要拼接两个方向，跟n_layers的层数无关。

        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, text):
        em_ad = self.ad_embedding(text[0])
        em_advertiser = self.advertiser_embedding(text[1])
        x = torch.cat([em_ad, em_advertiser], dim=2)
        em_creative = self.creative_embedding(text[2])
        x = torch.cat([x, em_creative], dim=2)
        em_industry = self.industry_embedding(text[3])
        x = torch.cat([x, em_industry], dim=2)
        em_product = self.product_embedding(text[4])
        x = torch.cat([x, em_product], dim=2)
        #print(x.shape)
        #x = torch.stack([em_ad, em_advertiser, em_creative, em_industry, em_product], dim=2)
        # text.shape=[seq_len, batch_size]
        embedded = self.dropout(x)
        # output: [batch,seq,2*hidden if bidirection else hidden]
        # hidden/cell: [bidirec * n_layers, batch, hidden]
        output, (hidden, cell) = self.lstm(embedded)

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]，

        return self.fc(hidden.squeeze(0))  # 在接一个全连接层，最终输出[batch size, output_dim]

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,batch_first=True,
                           bidirectional=bidirectional)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        # 这里hidden_dim乘以2是因为是双向，需要拼接两个方向，跟n_layers的层数无关。

        self.dropout = nn.Dropout(dropout)



    def forward(self, text):
        # text.shape=[seq_len, batch_size]
        embedded = self.dropout(self.embedding(text))
        # output: [batch,seq,2*hidden if bidirection else hidden]
        # hidden/cell: [bidirec * n_layers, batch, hidden]
        output, (hidden, cell) = self.rnn(embedded)

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]，

        return self.fc(hidden.squeeze(0))  # 在接一个全连接层，最终输出[batch size, output_dim]


class tCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filter,
                 filter_sizes, output_dim, dropout=0.2, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filter,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        # in_channels：输入的channel，文字都是1
        # out_channels：输出的channel维度
        # fs：每次滑动窗口计算用到几个单词,相当于n-gram中的n
        # for fs in filter_sizes用好几个卷积模型最后concate起来看效果。

        self.fc = nn.Linear(len(filter_sizes) * num_filter, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        # 升维是为了和nn.Conv2d的输入维度吻合，把channel列升维。
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved = [batch size, num_filter, sent len - filter_sizes+1]
        # 有几个filter_sizes就有几个conved

        pooled = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]  # [batch,num_filter]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, num_filter * len(filter_sizes)]
        # 把 len(filter_sizes)个卷积模型concate起来传到全连接层。

        return self.fc(cat)

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filter,
                 filter_sizes, output_dim, dropout=0.2, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filter,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        # in_channels：输入的channel，文字都是1
        # out_channels：输出的channel维度
        # fs：每次滑动窗口计算用到几个单词,相当于n-gram中的n
        # for fs in filter_sizes用好几个卷积模型最后concate起来看效果。

        self.fc = nn.Linear(len(filter_sizes) * num_filter, 64)
        self.fc1 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # [batch size, sent len, emb dim]
        embedded = embedded.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        # 升维是为了和nn.Conv2d的输入维度吻合，把channel列升维。
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conved = [batch size, num_filter, sent len - filter_sizes+1]
        # 有几个filter_sizes就有几个conved

        pooled = [F.max_pool1d(conv,conv.shape[2]).squeeze(2) for conv in conved]  # [batch,num_filter]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, num_filter * len(filter_sizes)]
        # 把 len(filter_sizes)个卷积模型concate起来传到全连接层。
        x = self.fc(cat)

        return x, self.fc1(x)