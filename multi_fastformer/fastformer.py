import os
import logging
import torch
from torch import einsum
import torch.nn as nn
from einops import rearrange

from transformers.models.bert.modeling_bert import (
    BertSelfOutput,
    BertIntermediate,
    BertOutput,
)
from transformers import RobertaModel



class AttentionPooling(nn.Module):
    def __init__(self, config):
        self.config = config
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.att_fc2 = nn.Linear(config.hidden_size, config.num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))
        return x


class FastSelfAttention(nn.Module):
    def __init__(self, config):
        super(FastSelfAttention, self).__init__()
        self.config = config
        self.n_global = config.n_global
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        #         self.input_dim= config.hidden_size
        self.input_dim = config.input_dim

        self.query = nn.Linear(self.all_head_size, self.all_head_size)
        self.query_att = nn.Linear(
            self.all_head_size, self.num_attention_heads * self.n_global
        )
        self.key = nn.Linear(self.all_head_size, self.all_head_size)
        self.key_att = nn.Linear(
            self.all_head_size, self.num_attention_heads * self.n_global
        )
        self.value = nn.Linear(self.all_head_size, self.all_head_size)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        # batch_size, seq_len, num_head * head_dim, batch_size, seq_len
        batch_size, seq_len, _ = hidden_states.shape
        query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # batch_size, num_head, seq_len

        #### Query
        query_for_score = (
            rearrange(
                self.query_att(query_layer),
                "b n (h g) -> b h g n",
                g=self.n_global,
            )
            / self.attention_head_size**0.5
        )
        # add attention mask
        #         attention_mask = rearrange(attention_mask, 'b i w -> b i () w')
        query_for_score += attention_mask.unsqueeze(1)

        query_weight = self.softmax(query_for_score)
        mixed_query_layer = rearrange(
            query_layer, "b n (h d) -> b h n d", d=self.attention_head_size
        )

        global_q = einsum("b h g n, b h n d -> b h d g", query_weight, mixed_query_layer)
        global_q = rearrange(global_q, "b h d g -> b h () d g")

        #### Key
        # batch_size, num_head, seq_len
        mixed_key_layer = rearrange(
            mixed_key_layer, "b n (h d) -> b h n d ()", d=self.attention_head_size
        )
        # torch.Size([64, 16, 256, 32])
        mixed_query_key_layer = rearrange(
            mixed_key_layer * global_q, "b h n d g-> b n (h d) g"
        )

        # torch.Size([64, 256, 256])

        mixed_query_key_layer = torch.max(mixed_query_key_layer, axis=-1).values

        query_key_score = (
            rearrange(
                self.key_att(mixed_query_key_layer),
                "b n (h g) -> b h g n",
                g=self.n_global,
            )
            / self.attention_head_size**0.5
        )

        # add attention mask
        query_key_score += attention_mask.unsqueeze(1)

        mixed_key_layer = mixed_key_layer.squeeze()
        global_k = einsum("b h g n, b h n d -> b h d g", query_key_score, 
                          mixed_key_layer)
        global_k = rearrange(global_k, "b h d g -> b h () d g")

        # query = value

        mixed_value_layer = rearrange(
            mixed_value_layer, "b n (h d) -> b h n d ()", d=self.attention_head_size
        )
        u = (global_k * mixed_value_layer)
        u = torch.max(u, axis=-1).values
        u = rearrange(u, "b h n d -> b n (h d)")
        r = self.transform(u)
        output = r + query_layer

        return output


class FastAttention(nn.Module):
    def __init__(self, config):
        super(FastAttention, self).__init__()
        self.self = FastSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class FastformerLayer(nn.Module):
    def __init__(self, config):
        super(FastformerLayer, self).__init__()
        self.attention = FastAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class FastformerEncoder(nn.Module):
    def __init__(self, config, pooler_count=1):
        super(FastformerEncoder, self).__init__()
        self.config = config
        self.encoders = nn.ModuleList(
            [FastformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # support multiple different poolers with shared bert encoder.
        self.poolers = nn.ModuleList()
        if config.pooler_type == "weightpooler":
            for _ in range(pooler_count):
                self.poolers.append(AttentionPooling(config))
        logging.info(f"This model has {len(self.poolers)} poolers.")

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_embs, attention_mask, pooler_index=0):
        # input_embs: batch_size, seq_len, emb_dim
        # attention_mask: batch_size, seq_len, emb_dim

        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        batch_size, seq_length, emb_dim = input_embs.shape
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_embs.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_embs + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # print(embeddings.size())
        all_hidden_states = [embeddings]

        for i, layer_module in enumerate(self.encoders):
            layer_outputs = layer_module(all_hidden_states[-1], extended_attention_mask)
            all_hidden_states.append(layer_outputs)
        assert len(self.poolers) > pooler_index
        output = self.poolers[pooler_index](all_hidden_states[-1], attention_mask)

        return output


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.dense_linear = nn.Linear(config.hidden_size, 4)
        self.word_embedding = nn.Embedding(
            config.word_size, self.config.hidden_size, padding_idx=0
        )
        self.fastformer_model = FastformerEncoder(config)
        self.criterion = nn.CrossEntropyLoss()
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Embedding)) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, labels, **kwargs):
        mask = input_ids.bool().float()
        embds = self.word_embedding(input_ids)
        text_vec = self.fastformer_model(embds, mask)
        score = self.dense_linear(text_vec)
        loss = self.criterion(score, labels)
        return loss, score

