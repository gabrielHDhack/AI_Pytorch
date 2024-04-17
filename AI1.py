import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('data3.json', 'r', encoding='utf-8', errors='ignore') as file:
    data = json.load(file)

questions = [entry['question'] for entry in data]
answers = [entry['answer'] for entry in data]

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
        assert d_model % num_heads == 0, f"d_model deve ser divisível por num_heads. d_model: {d_model}, num_heads: {num_heads}"
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
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model * nhead, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, query, key, value, mask=None):
        query = self.dropout(query)
        key = self.dropout(key)
        value = self.dropout(value)
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=mask)
        attn_output = self.layer_norm(query + attn_output)
        output = self.ffn(attn_output)
        return self.fc_out(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    
class NormalizedFeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(NormalizedFeedForward, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, d_model * 2)
        self.fc3 = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.dropout(x)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.layer_norm(x + x) 
        return x


class AdvancedTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(AdvancedTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                    dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        decoder_layers = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                    dim_feedforward=hidden_dim, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        self.dynamic_activation = DynamicActivationLayer(embedding_dim, num_heads, dropout) 
        self.causal_self_attention = CausalSelfAttention(embedding_dim, num_heads, dropout)
        self.multihead_attention = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = NormalizedFeedForward(embedding_dim, hidden_dim, dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.positional_encoding(embedded)
        memory = self.encoder(embedded)

        queries = keys = values = self.dropout(embedded)
        dynamic_output = self.dynamic_activation(queries)

        causal_output = self.causal_self_attention(embedded)
        combined_output = dynamic_output + causal_output
        transformer_output = self.decoder(combined_output, memory)
        feed_forward_output = self.feed_forward(transformer_output)
        output = self.fc(feed_forward_output)
        return output
    
vocab_size = len(word_to_idx)
embedding_dim = 12*100
hidden_dim = 1024*16
num_layers = 4*10
num_heads = 4*10 
model = AdvancedTransformer(vocab_size, embedding_dim, hidden_dim, num_layers, num_heads).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-6)

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

def generate_text_with_transformer(question, model, word_to_idx, idx_to_word, max_length=50, temperature=0.2):
    with torch.no_grad():
        question_indices = [word_to_idx[word] for word in question.split() if word in word_to_idx]
        padded_question = question_indices + [word_to_idx['<eos>']] + [0] * (max_length - len(question_indices) - 1)
        input_tensor = torch.tensor([padded_question], dtype=torch.long).to(device)

        output_indices = []
        for _ in range(max_length):
            output = model(input_tensor)
            output_probs = nn.functional.softmax(output[:, -1, :], dim=-1)
            output_probs = output_probs.to('cpu')
            output_probs = output_probs ** (1 / temperature)
            output_probs = output_probs / torch.sum(output_probs)
            predicted_index = torch.multinomial(output_probs, num_samples=1)

            if predicted_index == word_to_idx['<eos>']:
                break

            output_indices.append(predicted_index.item())
            input_tensor = torch.tensor([[predicted_index.item()]], dtype=torch.long).to(device)

        generated_text = ' '.join([idx_to_word[idx] for idx in output_indices])
        return generated_text

while True:
    user_question = input("Ask me something (or 'exit' to quit): ")
    if user_question.lower() == 'exit':
        print("Goodbye!")
        break

    generated_answer = generate_text_with_transformer(user_question, model, word_to_idx, idx_to_word, temperature=0.7)
    print(f'AI: {generated_answer}')
