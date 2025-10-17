import numpy as np 
import torch
from model import Transformer, Transformerconfig
from model import NoamOpt
import os
import time
from transliteration_data.data import data_driven

out_dir= 'out'
eval_interval=2000
eval_iters=282
always_save_checkpoint=True
wandb_log=True
wandb_project= 'Transformer_from_scratch'
wandb_run_name= 'transformer_model'
bias= True
vocab_size= 194



#adamw optimizer
learning_rate= 1e-3
max_iters= 35000
no_of_epochs= 10
weight_decay= 1e-1
beta1=0.9
beta2= 0.98
grad_clip= 1.0

batch_size= 32
block_size= 256
n_layer= 6
n_head= 8
n_embd= 512
dropout= 0.0

warmup_iters= 4000
lr_decay_iters= 600000
decay_lr=True
min_lr= 6e-5
compile= True

device= 'cuda'
data_dir='/mnt/sparsh_transformers/transliteration_data'

device_type= 'cuda' if 'cuda' in device else 'cpu'

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys} 

encoder_train_data, encoder_test_data, decoder_train_data, decoder_test_data= data_driven()
tokens_per_iter= batch_size*block_size
print(f"Tokens per iteration: {tokens_per_iter}")


master_process= True    
seed_offset= 0

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)

device_type = 'cuda' if 'cuda' in device else 'cpu'
# note: float16 data type will automatically use a GradScaler
# ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

current_batch_index=0

def get_batch(split):
        global current_batch_index  
        if split == 'train':
                encoder_data= encoder_train_data
                decoder_data= decoder_train_data
        elif split == 'val':
                encoder_data= encoder_test_data
                decoder_data= decoder_test_data

        if current_batch_index + batch_size > len(encoder_data):
                current_batch_index=0
        x= torch.from_numpy(encoder_data[current_batch_index: current_batch_index + batch_size].astype(np.int64)).to(device)
        y= torch.from_numpy(decoder_data[current_batch_index: current_batch_index + batch_size].astype(np.int64)).to(device)
        current_batch_index += batch_size

        x_mask= x!=0
        y_mask= y!=0
        future_mask= (1 - torch.triu(torch.ones(1, block_size, block_size), diagonal=1)).bool()
        return x.to(device), y.to(device), x_mask.to(device), y_mask.to(device), future_mask.to(device)

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

iter_num = 0
best_val_loss = 1e9
model_args['vocab_size'] = vocab_size
transformer_config= Transformerconfig(**model_args)
model= Transformer(transformer_config).to(device)

optimizer= model.configure_optimizers(learning_rate, (beta1, beta2), device_type)

scheduler= NoamOpt(n_embd, 1, warmup_iters, optimizer)

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) 

@torch.no_grad()
def estimate_loss():
        out={}
        accuracy_dict={}
        model.eval()
        for split in ['train', 'val']:
                losses= torch.zeros(eval_iters, device=device)
                accuracy= torch.zeros(eval_iters, device=device)
                for k in range(eval_iters):
                        X, Y, X_mask, Y_mask, future_mask = get_batch(split)
                        logits, loss = model(X, Y, X_mask, Y_mask, future_mask,1)
                        batch_accuracy=(torch.sum(logits.argmax(dim=-1) == Y))/ torch.numel(Y)
                        accuracy[k]= batch_accuracy.item()
                        losses[k]= loss.item()
                accuracy_dict[split]= accuracy.mean()
                out[split]= losses.mean()
        model.train()
        return out,accuracy_dict


if wandb_log and master_process:
       import wandb
       wandb.login()
       wandb.init(project=wandb_project, name=wandb_run_name, config=config)


X,Y,X_mask,Y_mask,future_mask= get_batch('train')
t0= time.time()
local_iter_num = 0 
iter_num = 0
while True:

    # determine and set the learning rate for this iteration
    lr = optimizer.param_groups[0]['lr']
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses ,accuracy_dict = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print("\n**************************************************************************\n")
        print(f"step {iter_num}: train accuracy {accuracy_dict['train']:.4f}, val accuracy {accuracy_dict['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr
            })
        if losses['val'] < best_val_loss and always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
    logits, loss = model(X, Y, X_mask, Y_mask, future_mask,1)
                
    X,Y,X_mask,Y_mask,future_mask= get_batch('train')
    
    loss.backward()
    scheduler.step()
    scheduler.optimizer.zero_grad()

    iter_num += 1
    local_iter_num += 1


    if iter_num > max_iters:
        break