import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np 


# compute perplexity
def perplexity_val(probs, n):
    return -1/float(n) * np.log(probs)


def predict(dataset, model, text, next_words=100):
    model.eval()
    start_sentence = '<s>'
    end_sentence = '</s>'
    words = [start_sentence] + text.split(' ')  # splits the text into words
    state_h, state_c = model.init_state(len(words)) 
    perplexity = 1 # log cannot take 0 
    sent_perplexity = []
    words_in_sent = 0 
    
    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        # it makes raw predictions for the possible words based on the words that we already have
        last_word_logits = y_pred[0][-1]
        
        # the softmax function generates a vector of (normalized) probabilities with one value for each possible word.
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        #print('Number of possible words:', len(p))
        
        # it makes a weighted random choice from the possible words according to p 
        word_index = np.random.choice(len(last_word_logits), p=p)
        chosen_word = dataset.index_to_word[word_index]
        # adds the word to our sentence
        words.append(chosen_word)
        #print('Index of chosen word:', word_index)
        #print('Probability of chosen word:', p[word_index])
        #print('Chosen word:', dataset.index_to_word[word_index])
        #print(' ')
        
        words_in_sent += 1
        perplexity *= p[word_index]
        
        #print(perplexity)
        #print(words_in_sent)
        
        # if we are at the end of the sentence
        if chosen_word == end_sentence: # observed the end of sentence
            sent_perplexity.append(perplexity_val(perplexity,words_in_sent)) # compute the total perplexity of the generated text
            words_in_sent = 0 # reset the word count for the next sentence
            perplexity = 1 # reset perplexity for the next sentence     
    
    if words[-1] != end_sentence:
        sent_perplexity.append(perplexity_val(perplexity,words_in_sent)) # compute the total perplexity of the generated text
        words.append(end_sentence) # adds </s> at the end of the text 
        
    text = ' '.join(words)

    return text, sent_perplexity