# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from params import NUM_WORDS, EMBEDDING_SIZE, ABLATION1, ABLATION3


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class CrossDecoderTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, att_dropout, ffn_dropout):
        super(CrossDecoderTransformerLayer, self).__init__()
        self.self_attn_text = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=att_dropout)
        self.self_attn_tabular = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=att_dropout)
        
        self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=att_dropout)
        self.cross_attn_tabular = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=att_dropout)
        
        self.ffn_text1 = nn.Linear(d_model, dim_feedforward)
        self.ffn_text2 = nn.Linear(dim_feedforward, d_model)
        
        self.ffn_tabular1 = nn.Linear(d_model, dim_feedforward)
        self.ffn_tabular2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1_text = nn.LayerNorm(d_model)
        self.norm1_tabular = nn.LayerNorm(d_model)
        
        self.norm2_text = nn.LayerNorm(d_model)
        self.norm2_tabular = nn.LayerNorm(d_model)
        
        self.norm3_text = nn.LayerNorm(d_model)
        self.norm3_tabular = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(ffn_dropout)
        
    def forward(self, text, tabular):
        if not ABLATION3:
            # Text Self Attention
            text = self.norm1_text(text + self.self_attn_text(text, text, text)[0])
            # Tabular Self Attention
            tabular = self.norm1_tabular(tabular + self.self_attn_tabular(tabular, tabular, tabular)[0])
        
        # Cross Attention: Text to Tabular
        text = self.norm2_text(text + self.cross_attn_text(text, tabular, tabular)[0])
        # Cross Attention: Tabular to Text
        tabular = self.norm2_tabular(tabular + self.cross_attn_tabular(tabular, text, text)[0])
        
        # Feed Forward Network for Text
        text1 = self.dropout(F.relu(self.ffn_text1(text)))
        text = self.norm3_text(text + self.ffn_text2(text1))
        # Feed Forward Network for Tabular
        tabular1 = self.dropout(F.relu(self.ffn_tabular1(tabular)))
        tabular = self.norm3_tabular(tabular + self.ffn_tabular2(tabular1))
        
        return text, tabular
    
    

# Text classifier based on a pytorch TransformerEncoder.
class CrossDecoderTransformer(nn.Module):
    def __init__(
            self,
            nhead=10,
            dim_feedforward=600,
            num_layers=1,
            att_dropout=0.1,
            ffn_dropout=0.1,
            num_class = 2,
            pe_dropout = 0.1
    ):
        super().__init__()

        vocab_size = NUM_WORDS + 2
        d_model = EMBEDDING_SIZE
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        # Embedding layer definition
        self.text_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.speaker_emb = nn.Embedding(3, d_model, padding_idx=0)
        self.id_emb = nn.Embedding(6212, d_model, padding_idx=0)
        self.department_emb = nn.Embedding(59, d_model, padding_idx=0)
        self.title_emb = nn.Embedding(7, d_model, padding_idx=0)
        self.linear_layer = nn.Linear(110, d_model)

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            dropout=pe_dropout,
            vocab_size=vocab_size
        )
        self.layers = nn.ModuleList([
            CrossDecoderTransformerLayer(d_model, nhead, dim_feedforward, att_dropout, ffn_dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(d_model, num_class)
        self.d_model = d_model

    def forward(self, X_text, X_cat, X_num, speakers):
        # text embedding and PE
        if not ABLATION1:
            X_text = (self.text_emb(X_text) + self.speaker_emb(speakers)) * math.sqrt(self.d_model)
        else:
            X_text = (self.text_emb(X_text)) * math.sqrt(self.d_model)
        X_text = self.pos_encoder(X_text)
        
        # tabular embedding
        id_embeddings = self.id_emb(X_cat[:, 0])  # (batch_size, d_model)
        department_embeddings = self.department_emb(X_cat[:, 1])  # (batch_size, d_model)
        title_embeddings = self.title_emb(X_cat[:, 2])  # (batch_size, d_model)
        X_num = X_num.float()
        num_embeddings = self.linear_layer(X_num)  # (batch_size, d_model)
        X_tab = torch.stack([id_embeddings, department_embeddings, title_embeddings, num_embeddings], dim=1)
        
        # cross-decoder layer
        for layer in self.layers:
            X_text, X_tab = layer(X_text, X_tab)
            
        X = torch.cat((X_text, X_tab), dim = 1)
        
        # final classifier
        X = X.mean(dim=1)
        X = self.classifier(X)

        return X
