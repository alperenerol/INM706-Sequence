
import torch 
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np 

def save_checkpoint(model_state, filename='checkpoints.tar'): 
    print("-> Saving checkpoint") 
    torch.save(model_state, filename)      
    
def load_checkpoint(checkpoint, model, optimizer): 
    print("-> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer']) 
    
    
def train(dataset, model, args, load_model=False): 
    model.train() 
    dataloader = DataLoader(dataset, batch_size=args.batch_size) 
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
        
    # WARNING: everytime we set load_model=False, it overwrites the previously saved file.
    if load_model: 
        load_checkpoint(torch.load('checkpoints.tar'), model, optimizer) 
        
    mean_loss = []
    loss_history = []
    perplexity_history = []
    
    for epoch in range(args.max_epochs): 
        
        # save checkpoint   
        if epoch % 10 == 0 and epoch != 0: 
            checkpoint = {
                'state_dict': model.state_dict(), 
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint) 
            
        loop = tqdm(dataloader, leave=True)
        state_h, state_c = model.init_state(args.sequence_length) 
    
        for batch, (x, y) in enumerate(loop): 
            optimizer.zero_grad() 
            
            y_pred, (state_h, state_c) = model(x, (state_h, state_c)) 
            loss = criterion(y_pred.transpose(1,2),y)
            mean_loss.append(loss.item())
            
            state_h = state_h.detach()
            state_c = state_c.detach() 
            
            loss.backward() 
            optimizer.step() 
            
            # update progress bar
            loop.set_postfix(loss=loss.item())
        
        avg_loss = sum(mean_loss)/len(mean_loss)
        perplexity = np.exp(avg_loss)
        loss_history.append(avg_loss)
        perplexity_history.append(perplexity) 
        
        print(f"\033[34m EPOCH {epoch + 1}: \033[0m Mean loss {avg_loss:.3f}, \033[0m Perplexity {perplexity:.3f}")
        
    return loss_history, perplexity_history