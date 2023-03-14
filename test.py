import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np 

def predict(dataset, model, text, next_words=100):
    model.eval()
    end_sentence = '</s>'
    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])
    
    if words[-1] != end_sentence:
        words.append(end_sentence)
        
    text = ' '.join(words)

    return text