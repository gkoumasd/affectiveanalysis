import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn

class LFLSTM(nn.Module):
    # modality: sentences, visual, or acoustic
    def __init__(self, input_size, hidden_size, fc1_size, output_size, dropout_rate, len_word2id):
        super(LFLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.len_word2id = len_word2id
        
        
        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.embed = nn.Embedding(len_word2id, input_size[0])
        self.rnn1 = nn.LSTM(input_size[0], hidden_size[0], bidirectional=True)
        self.rnn2 = nn.LSTM(2*hidden_size[0], hidden_size[0], bidirectional=True)
        
        
        self.fc1 = nn.Linear(sum(hidden_size)*4, fc1_size)
        self.fc2 = nn.Linear(fc1_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm((hidden_size[0]*2,))
        self.bn = nn.BatchNorm1d(sum(hidden_size)*4)
        
        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        
        packed_sequence = pack_padded_sequence(sequence, lengths)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2
    
    
    # modality: sentences, visual, or acoustic
    def fusion(self, sentences,lengths):
        
        batch_size = lengths.size(0)
        
        sentences = self.embed(sentences)
        # extract features from text modality
        final_h1, final_h2 = self.extract_features(sentences, lengths, self.rnn2, self.rnn1, self.layer_norm)
        
        # simple late fusion -- concatenation + normalization
        #stack up lstm outputs
        h = torch.cat((final_h1, final_h2),dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        
        return self.bn(h)
    
    
    def forward(self, sentences, lengths):
        
        batch_size = lengths.size(0)
        
        h = self.fusion(sentences, lengths)
        h = self.fc1(h) 
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        
        return o