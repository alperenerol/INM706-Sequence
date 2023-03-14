import torch 
from torch import nn


class basic_model(nn.Module): 
# SOURCE: https://www.kdnuggets.com/2020/07/pytorch-lstm-text-generation-tutorial.html
    
    def __init__(self, dataset): 
        super(basic_model, self).__init__() 
        self.lstm_size = 128 # number of expected features in the input x
        self.embedding_dim = 128 # size of each embedding vector
        self.num_layers = 3 # number of stacked LSTM layers
        
        n_vocab = len(dataset.unique_words) 
        
        # Embedding layer converts word indexes to word vectors.
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab, 
            embedding_dim=self.embedding_dim, 
        ) 
        
        self.lstm = nn.LSTM(
            input_size=self.lstm_size, 
            hidden_size=self.lstm_size, 
            num_layers=self.num_layers, 
        )
        
        self.fc = nn.Linear(self.lstm_size, n_vocab) 
    
    def forward(self, x, prev_state): 
        embed = self.embedding(x) 
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output) 
        return logits, state 
    
    def init_state(self, sequence_length): 
    #function called at the start of every epoch to initialize the right shape of the state. 
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
        
        
        