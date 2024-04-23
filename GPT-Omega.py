import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import spacy
from torch.utils.data import DataLoader, TensorDataset
import math
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nlp = spacy.load("en_core_web_lg")

with open('BotsQ&A.json', 'r', encoding='utf-8', errors='ignore') as file:
    data = json.load(file)

questions = [entry['question'] for entry in data]
answers = [entry['answer'] for entry in data]

questions = [token.text for question in questions for token in nlp(question)]
answers = [token.text for answer in answers for token in nlp(answer)]

word_to_idx = {}
idx_to_word = {}
for question in questions:
    for word in question.split():
        if word not in word_to_idx:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word

if '<eos>' not in word_to_idx:
    idx = len(word_to_idx)
    word_to_idx['<eos>'] = idx
    idx_to_word[idx] = '<eos>'

for answer in answers:
    for word in answer.split():
        if word not in word_to_idx:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word

X = []
y = []

max_length = max(len(seq) for seq in questions + answers)  

for question, answer in zip(questions, answers):
    question_indices = [word_to_idx[word] for word in question.split() if word in word_to_idx]
    answer_indices = [word_to_idx[word] for word in answer.split() if word in word_to_idx]
    padded_question = question_indices + [word_to_idx['<eos>']] + [0] * (max_length - len(question_indices) - 1)
    padded_answer = answer_indices + [word_to_idx['<eos>']] + [0] * (max_length - len(answer_indices) - 1)

    X.append(padded_question)
    y.append(padded_answer)

X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=512, shuffle=True)


class Lamb(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, adam=adam)
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                beta1_correction = 1 - beta1 ** state['step']
                beta2_correction = 1 - beta2 ** state['step']

                lr = group['lr'] * torch.sqrt(torch.tensor(beta2_correction)) / torch.tensor(beta1_correction)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = lr
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    

class DynamicActivationLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(DynamicActivationLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout) 
        
    def forward(self, queries):
        batch_size, seq_len, d_model = queries.size()
        query = self.query(queries).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(queries).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(queries).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)  
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.fc_out(output)
        output = self.layer_norm(queries + output)
        return output


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CausalSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
        scores = scores.masked_fill(mask == 1, float('-inf'))
        attention_probs = nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.fc_out(output)
        output = self.layer_norm(x + output)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        key = self.key(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        value = self.value(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_probs = nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc_out(context)
        output = self.layer_norm(query + output)
        output = self.ffn(output)
        return output
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.fc3 = nn.Linear(d_model, d_model * 2)
        self.fc4 = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=x.device)
        position_embeddings = self.position_embeddings(positions)
        position_embeddings = position_embeddings.unsqueeze(0).expand_as(x)
        x = x + self.pe[:seq_length] + position_embeddings
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        x = self.dropout(x)
        return x

    
class ResidualFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1, num_layers=90):
        super(ResidualFeedForward, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(d_model, hidden_dim * 8),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim * 8),
                nn.Linear(hidden_dim * 8, hidden_dim * 4),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, d_model * 8),
                nn.ReLU(),
                nn.LayerNorm(d_model * 8),
                nn.Linear(d_model * 8, d_model * 4),
                nn.ReLU(),
                nn.LayerNorm(d_model * 4),
                nn.Linear(d_model * 4, d_model * 2),
                nn.ReLU(),
                nn.LayerNorm(d_model * 2),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            ))
        
    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        
        return x + residual


class MaskedLanguageModelHead(nn.Module):
    def __init__(self, d_model, vocab_size, projection_dim=2048, dropout=0.1): 
        super(MaskedLanguageModelHead, self).__init__()
        self.layer_norm_before = nn.LayerNorm(d_model)  
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 8),
            nn.LayerNorm(d_model * 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 8, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, d_model * 4),
            nn.LayerNorm(d_model * 4),
            nn.ReLU()
        )
        self.classifier = nn.Linear(d_model * 4, vocab_size)

    def forward(self, x):
        x = self.layer_norm_before(x)  
        x = self.mlp(x)
        x = self.classifier(x)
        return x


class AxialAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(AxialAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_row = nn.Linear(d_model, d_model)
        self.query_col = nn.Linear(d_model, d_model)
        self.key_row = nn.Linear(d_model, d_model)
        self.key_col = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        query_row = self.query_row(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        query_col = self.query_col(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_row = self.key_row(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_col = self.key_col(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores_row = torch.matmul(query_row, key_row.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores_col = torch.matmul(query_col, key_col.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs_row = nn.functional.softmax(scores_row, dim=-1)
        attention_probs_col = nn.functional.softmax(scores_col, dim=-1)
        attention_probs_row = self.dropout(attention_probs_row)
        attention_probs_col = self.dropout(attention_probs_col)
        output_row = torch.matmul(attention_probs_row, value)
        output_col = torch.matmul(attention_probs_col.transpose(-2, -1), value)
        output_row = output_row.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output_col = output_col.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = output_row + output_col
        output = self.fc_out(output)
        output = self.layer_norm(x + output)
        return output


class LocalAttention(nn.Module):
    def __init__(self, d_model, window_size, num_heads, dropout=0.1):
        super(LocalAttention, self).__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.zeros(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_probs = nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.fc_out(output)
        output = self.layer_norm(x + output)
        return output


class SparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, sparsity=4, dropout=0.1, max_seq_len=1024):
        super(SparseAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.sparsity = sparsity
        
        # Precompute sparse mask
        self.register_buffer('sparse_mask', self._generate_sparse_mask(max_seq_len))
        
    def _generate_sparse_mask(self, seq_len):
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.sparsity // 2)
            end = min(seq_len, i + self.sparsity // 2 + 1)
            mask[i, start:end] = 1
        return mask.unsqueeze(0).unsqueeze(0)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        scores = scores.masked_fill(self.sparse_mask[:, :, :seq_len, :seq_len] == 0, float('-inf'))
        
        attention_probs = nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.fc_out(output)
        output = self.layer_norm(x + output)
        return output



class KernelAttention(nn.Module):
    def __init__(self, d_model, kernel_size=3, num_layers=90, dropout=0.1):
        super(KernelAttention, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        for layer in self.layers:
            x = F.relu(layer(x.transpose(1, 2)).transpose(1, 2))
            x = self.dropout(x)
        
        output = self.layer_norm(residual + x)
        return output



class ReformerAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(ReformerAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_probs = nn.functional.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        output = torch.matmul(attention_probs, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.fc_out(output)
        return output


class AdvancedTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(AdvancedTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.encoder_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, hidden_dim * 4, kernel_size=3, padding=1, dilation=1)
            for _ in range(num_layers)
        ])
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout),
            num_layers=num_layers
        )
        self.decoder_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, hidden_dim * 4, kernel_size=3, padding=1, dilation=1)
            for _ in range(num_layers)
        ])
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout),
            num_layers=num_layers
        )
        self.dynamic_activation = DynamicActivationLayer(embedding_dim, num_heads, dropout)
        self.causal_self_attention = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.axial_attention = AxialAttention(embedding_dim, num_heads, dropout)
        self.local_attention = LocalAttention(embedding_dim, window_size=3, num_heads=num_heads, dropout=dropout)
        self.kernel_attention = KernelAttention(embedding_dim, num_layers=3, dropout=dropout)
        self.mlm_head = MaskedLanguageModelHead(embedding_dim, vocab_size)
        self.multihead_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward1 = ResidualFeedForward(embedding_dim, hidden_dim, dropout)
        self.feed_forward2 = ResidualFeedForward(embedding_dim, hidden_dim, dropout)
        self.sparse_attention = SparseAttention(embedding_dim, num_heads, sparsity=4, dropout=dropout)
        self.reformer_attention = ReformerAttention(embedding_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)
        encoder_output = embedded.permute(0, 2, 1)  
        for layer in self.encoder_layers:
            encoder_output = F.relu(layer(encoder_output))
        encoder_output = encoder_output.permute(0, 2, 1)
        memory = self.encoder(encoder_output)
        decoder_output = embedded.permute(0, 2, 1)
        for layer in self.decoder_layers:
            decoder_output = F.relu(layer(decoder_output))
        decoder_output = decoder_output.permute(0, 2, 1)
        queries = keys = values = self.dropout(decoder_output)
        dynamic_output = self.dynamic_activation(queries)
        causal_output = self.causal_self_attention(decoder_output)
        axial_output = self.axial_attention(decoder_output)
        local_output = self.local_attention(decoder_output)
        kernel_output = self.kernel_attention(decoder_output)
        combined_output = dynamic_output + causal_output + axial_output + local_output + kernel_output
        sparse_output = self.sparse_attention(combined_output)
        transformer_output = self.decoder(sparse_output, memory)
        feed_forward_output1 = self.feed_forward1(transformer_output)
        feed_forward_output2 = self.feed_forward2(feed_forward_output1)
        mlm_output = self.mlm_head(feed_forward_output2)
        return mlm_output




vocab_size = len(word_to_idx)
embedding_dim = 960
hidden_dim = 1024*16
num_layers = 96
num_heads = 48
model = AdvancedTransformer(vocab_size, embedding_dim, hidden_dim, num_layers, num_heads).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Lamb(model.parameters(), lr=5e-6)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_parameters = count_parameters(model)
print(f'Total parameters in the model: {total_parameters}')

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            outputs = model(batch_X.to(device))
            loss = criterion(outputs.view(-1, vocab_size), batch_y.view(-1).to(device))
            total_loss += loss.item()
            total_samples += batch_y.size(0) * batch_y.size(1)

    average_loss = total_loss / len(dataloader)
    perplexity = math.exp(average_loss)

    return average_loss, perplexity

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X.to(device))
        loss = criterion(outputs.view(-1, vocab_size), batch_y.view(-1).to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 2)
        correct_predictions += (predicted == batch_y.to(device)).sum().item()
        total_samples += batch_y.size(0) * batch_y.size(1)

    train_accuracy = correct_predictions / total_samples
    train_average_loss = total_loss / len(dataloader)
    val_loss, val_perplexity = evaluate_model(model, dataloader, criterion)

    print(f'Epoch {epoch + 1}/{num_epochs}, '
          f'Train Loss: {train_average_loss:.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}')

#torch.save(model.state_dict(), 'advanced_transformer_model.pth')

def generate_text_with_transformer(question, model, word_to_idx, idx_to_word, max_length=10, temperature=0.7):
    with torch.no_grad():
        question_indices = [word_to_idx[word] for word in question.split() if word in word_to_idx]
        padded_question = question_indices + [word_to_idx['<eos>']] + [0] * (max_length - len(question_indices) - 1)
        input_tensor = torch.tensor([padded_question], dtype=torch.long).to(device)

        output_indices = []
        for _ in range(max_length):
            mask = (input_tensor != 0)
            output = model(input_tensor, mask=mask)
            output_probs = nn.functional.softmax(output[:, -1, :], dim=-1)
            output_probs = output_probs.to('cpu')
            output_probs = output_probs ** (1 / temperature)
            output_probs = output_probs / torch.sum(output_probs)
            predicted_index = torch.multinomial(output_probs, num_samples=1)

            if predicted_index.item() == word_to_idx['<eos>']: 
                break

            output_indices.append(predicted_index.item())
            input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_index.item()]], dtype=torch.long).to(device)], dim=1)

        generated_text = ' '.join([idx_to_word[idx] for idx in output_indices])
        return generated_text


while True:
    user_question = input("Ask me something (or 'exit' to quit): ")
    if user_question.lower() == 'exit':
        print("Goodbye!")
        break

    generated_answer = generate_text_with_transformer(user_question, model, word_to_idx, idx_to_word, temperature=0.7)
    print(f'AI: {generated_answer}')
