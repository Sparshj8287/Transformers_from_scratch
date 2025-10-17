import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from dataclasses import dataclass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class MultiheadAttention(nn.Module):
    """
    A module that implements the multi-head self-attention mechanism
    which is a key component of Transformer models.

    The module takes a configuration object upon initialization which
    should contain the necessary parameters such as the number of
    heads (n_head), the embedding dimension (n_embd), the dropout rate
    (dropout), and the use of bias.

    Attributes:
    c_attn (torch.nn.Module): A linear layer used to create query, key,
        and value vectors from the input.
    attn_dropout (torch.nn.Module): Dropout layer applied to attention weights.
    n_head (int): The number of attention heads.
    n_embd (int): The size of each embedding vector.
    dropout (float): The dropout rate.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd,bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,bias=False)

        #regularization
        self.attn_dropout= nn.Dropout(config.dropout)
        self.resid_dropout= nn.Dropout(config.dropout)
        self.n_head= config.n_head
        self.n_embd= config.n_embd
        self.dropout=config.dropout

    def forward(self, x,src_mask=None, mask=None):
        """
        Forward propagate the multi-head attention mechanism.

        Parameters:
        x (torch.Tensor): The input tensor of shape [batch size, sequence length, embedding size]
        mask (torch.Tensor, optional): An optional mask tensor for the attention mechanism.

        Returns:
        y (torch.Tensor): The output of the multi-head attention of shape [batch size, sequence length, embedding size]
        """

        B,T,C= x.size()

        q,k,v= self.c_attn(x).split(self.n_embd,dim=2)
        k= k.view(B,T, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)
        q= q.view(B,T, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)
        v= v.view(B,T, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)

        att= (q@ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        if mask!=None:
            att= att.masked_fill(mask==0, float('-inf'))

        if src_mask!=None:
            att= att.masked_fill(src_mask[:, None, None, :] == 0, float("-inf"))

        att= F.softmax(att,dim=-1)
        att=self.attn_dropout(att)

        y=att@v #(B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs)

        y=y.transpose(1,2).contiguous().view(B,T,C)

        y= self.resid_dropout(self.c_proj(y))

        return y
    






class CrossMultiheadAttention(nn.Module):
    """
    A module that implements the multi-head self-attention mechanism
    which is a key component of Transformer models.

    The module takes a configuration object upon initialization which
    should contain the necessary parameters such as the number of
    heads (n_head), the embedding dimension (n_embd), the dropout rate
    (dropout), and the use of bias.

    Attributes:
    c_attn (torch.nn.Module): A linear layer used to create query, key,
        and value vectors from the input.
    attn_dropout (torch.nn.Module): Dropout layer applied to attention weights.
    n_head (int): The number of attention heads.
    n_embd (int): The size of each embedding vector.
    dropout (float): The dropout rate.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        #Attention mechanism
        self.c_attn= nn.Linear(config.n_embd, config.n_embd,bias=False)
        self.c_attn2 = nn.Linear(config.n_embd, 2*config.n_embd,bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd,bias=False)

        #regularization
        self.attn_dropout= nn.Dropout(config.dropout)
        self.resid_dropout= nn.Dropout(config.dropout)
        self.n_head= config.n_head
        self.n_embd= config.n_embd
        self.dropout=config.dropout

    def forward(self, x, encoder_output, src_mask=None):
        """
        Forward propagate the multi-head attention mechanism.

        Parameters:
        x (torch.Tensor): The input tensor of shape [batch size, sequence length, embedding size]
        mask (torch.Tensor, optional): An optional mask tensor for the attention mechanism.

        Returns:
        y (torch.Tensor): The output of the multi-head attention of shape [batch size, sequence length, embedding size]
        """

        B,T,C= x.size()
        B,T_enc,C= encoder_output.size()

        q= self.c_attn(x)
        k,v= self.c_attn2(encoder_output).split(self.n_embd,dim=2)
        
        k= k.view(B,T_enc, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)
        v= v.view(B,T_enc, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)


        q= q.view(B,T, self.n_head, C//self.n_head).transpose(1,2) #(B, nh, T, hs)
        att= (q@ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))

        if src_mask!=None:
            att= att.masked_fill(src_mask[:, None, None, :] == 0, float("-inf"))

        att= F.softmax(att,dim=-1)
        att=self.attn_dropout(att)
        y=att@v #(B,nh,T,T) x (B,nh,T,hs) -> (B,nh,T,hs)
        y=y.transpose(1,2).contiguous().view(B,T,C)

        y= self.resid_dropout(self.c_proj(y))

        return y
    








class FeedForwardNeuralNetwork(nn.Module):
  """
  A simple Feed-Forward Neural Network (FFNN) module consisting of two linear layers
  with a GELU (Gaussian Error Linear Unit) activation function in between.

  Note:
  This FFNN is often used in the Transformer model after the Multihead Attention mechanism.

  Attributes:
  c_fc (torch.nn.Module): A linear layer used to transform the input data.
  gelu (torch.nn.Module): An activation layer using GELU.
  final_layer (torch.nn.Module): A final linear layer to transform
      the data to the original dimensionality.
  dropout (torch.nn.Module): Dropout layer applied after the final linear transformation.
  """

  def __init__(self, config):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
    self.gelu = nn.GELU()
    self.final_layer = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
    self.dropout = nn.Dropout(config.dropout)

  def forward(self, x):
    """
    Forward propagate the FFNN.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    x (torch.Tensor): The output of the FFNN.
    """
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.final_layer(x)
    x = self.dropout(x)
    return x






class PositionalEncoding(nn.Module):
  """
  A module to add positional encodings to the input sequences.

  The positional encoding uses sinusoidal functions of different frequencies to
  encode the position. It is added to the input embeddings to provide
  information about the relative position of the words in a sentence.

  Attributes:
  pe (torch.Tensor): The positional encodings for maximum sequence length.
  """

  def __init__(self, config, t=256):
    super().__init__()

    pe = torch.zeros(t, config.n_embd)
    position= torch.arange(0, t, dtype=torch.float).unsqueeze(1)
    div_term= torch.exp(torch.arange(0, config.n_embd, 2).float() * -(math.log(10000.0) / config.n_embd))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self,x):
    """
    Forward propagate the positional encodings.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    (torch.Tensor): The input tensor added with positional encodings.
    """
    return x + self.pe[:, :x.size(1)]
  







class LayerNorm(nn.Module):
  """
  A module for Layer Normalization: normalization is done across
  each feature in a training example. As compared to Batch
  Normalization, Layer Normalization is not sensitive to the batch
  size.

  Attributes:
  weight (torch.nn.Parameter): learnable scale factors. The number of
    these factors is equal to the input size.
  bias (torch.nn.Parameter): learnable shift factors. The number of
    these factors is equal to the input size.
  """
  def __init__(self, ndim, bias):
    super().__init__()
    # Initialize the learnable parameters weight and bias.
    self.weight = nn.Parameter(torch.ones(ndim))
    self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

  def forward(self, input):
    """
    Forward propagate the LayerNorm module.

    Parameters:
    input (torch.Tensor): The input tensor.

    Returns:
    (torch.Tensor): The output of the Layer Normalization.
    """
    return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
  









class Encoder(nn.Module):
  """
  A transformer encoder module. Consists of a Multihead Attention mechanism
  followed by positional feedforward neural network. Layer normalization is
  applied before the attention and feedforward network, respectively

  Attributes:
  ln_1 (LayerNorm): The first layernorm module, applied before attention mechanism
  attn (MultiheadAttention): The attention mechanism
  ln_2 (LayerNorm): The second layernorm module, applied before feedforward network
  FeedForwardNeuralNetwork (FeedForwardNeuralNetwork): The positional feedforward network
  """

  def __init__(self,config):
    super().__init__()
    self.ln_1= LayerNorm(config.n_embd,bias=config.bias)
    self.attn= MultiheadAttention(config)
    self.ln_2 = LayerNorm(config.n_embd, bias= config.bias)
    self.mlp= FeedForwardNeuralNetwork(config)

  def forward(self,x,src_mask):
    """
    Forward propagate the encoder module.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    x (torch.Tensor): The output of the encoder module.
    """
    x = x+self.attn(self.ln_1(x),src_mask)
    x= x+ self.mlp(self.ln_2(x))
    return x
  






@dataclass
class Transformerconfig:
  block_size: int =256
  vocab_size: int = 194
  n_layer: int= 6
  n_head: int =8
  n_embd: int= 512
  dropout: float=0.0
  bias: bool = True
  padding_idx: int = 0












class Decoder(nn.Module):
  """
  """

  def __init__(self,config):
    super().__init__()
    self.ln_1= LayerNorm(config.n_embd,bias=config.bias)
    self.attn_1= MultiheadAttention(config)
    self.ln_2 = LayerNorm(config.n_embd,bias= config.bias)
    self.attn_2= CrossMultiheadAttention(config)
    self.ln_3 = LayerNorm(config.n_embd, bias= config.bias)
    self.mlp= FeedForwardNeuralNetwork(config)

  def forward(self,x,src_mask,mask,encoder_output):
    """
    """
    x = self.attn_1(self.ln_1(x), src_mask, mask)
    x = x+ self.attn_2(self.ln_2(x),encoder_output,src_mask)
    x= x+ self.mlp(self.ln_3(x))
    return x
  


class Transformer(nn.Module):
  def __init__(self,config):
    super().__init__()
    assert config.vocab_size is not None
    assert config.block_size is not None
    self.config= config


    self.transformer_encoder= nn.ModuleDict(dict(
        wte= nn.Embedding(config.vocab_size,config.n_embd,padding_idx=config.padding_idx),
        wpe= nn.Embedding(config.block_size,config.n_embd,padding_idx=config.padding_idx),
        # wpe= PositionalEncoding(config),
        drop= nn.Dropout(config.dropout),
        h= nn.ModuleList([Encoder(config) for _ in range(config.n_layer)]),
        ln_f= LayerNorm(config.n_embd, bias=config.bias)
    ))

    self.transformer_decoder= nn.ModuleDict(dict(
        wte= nn.Embedding(config.vocab_size,config.n_embd,padding_idx=config.padding_idx),
        wpe= nn.Embedding(config.block_size,config.n_embd,padding_idx=config.padding_idx),
        # wpe= PositionalEncoding(config),
        drop= nn.Dropout(config.dropout),
        h= nn.ModuleList([Decoder(config) for _ in range(config.n_layer)]),
        ln_f= LayerNorm(config.n_embd, bias=config.bias)
    ))

    self.lm_head= nn.Linear(config.n_embd, config.vocab_size,bias=False)

    self.apply(self._init_weights)

    for pn,p in self.named_parameters():
        if pn.endswith('c_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*config.n_layer))


    print("number of parameters: %.4fM" % (self.get_num_params()/1e6,))

  
  
  def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer_encoder.wpe.weight.numel()+self.transformer_decoder.wpe.weight.numel()
        return n_params
  

  
  
  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
          if module.bias is not None:
              torch.nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
          torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def custom_cross_entropy(input, target, ignore_indices=[0,1]):
    log_softmax = F.log_softmax(input, dim=-1)
    losses = -torch.gather(log_softmax, dim=-1, index=target.unsqueeze(-1))
    mask = torch.ones_like(target, dtype=torch.bool)
    for idx in ignore_indices:
        mask &= target.ne(idx)
    losses *= mask.type_as(losses)
    return losses.sum() / max(mask.sum(), 1)
  
  
  def forward(self, idx,target_idx,src_mask=None,tgt_mask=None, mask=None, targets=None):

    device= idx.device

    # targets= target_idx

    b,t= idx.size()
    # self.config.block_size= t

    assert t<= self.config.block_size,f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
    pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
    #Encoder part Starts
    tok_emb= self.transformer_encoder.wte(idx)
    pos_emb = self.transformer_encoder.wpe(pos) 
    x = self.transformer_encoder.drop(tok_emb + pos_emb)
    for block in self.transformer_encoder.h:
      encoder_output = block(x,src_mask)
    encoder_output= self.transformer_encoder.ln_f(encoder_output)


    b,t= target_idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

    #Decoder part Starts
    out_tok_emb= self.transformer_decoder.wte(target_idx)
    pos_emb = self.transformer_decoder.wpe(pos)
    x_decoder = self.transformer_decoder.drop(out_tok_emb + pos_emb)
    for block in self.transformer_decoder.h:
      x_decoder= block(x_decoder,tgt_mask,mask,encoder_output)
    x_decoder= self.transformer_decoder.ln_f(x_decoder)
  
    
    if targets is not None:
            targets= target_idx
            logits = self.lm_head(x_decoder)
            loss = custom_cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    else:
            logits = self.lm_head(x_decoder[:, [-1], :]) 
            loss = None

    return logits, loss
  

  def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(lr=learning_rate, betas=betas)
        return optimizer
  
  


  @torch.no_grad()
  def generate(self, idx, start_token, eos_token, temperature=1.0, device=None):
    """
    Generate sequences from the model.

    Parameters:
    idx (torch.Tensor): The input tensor.
    start_token (int): The token to start the generation with.
    eos_token (int): The token representing the end of the sequence.
    temperature (float): A temperature value to apply to the logits before sampling.
    device (torch.device): The device to run the model on.

    Returns:
    (torch.Tensor): The generated sequence.
    """
    self.eval()

    src_mask= idx!=0

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx = idx.to(device)
    _, t = idx.size()
    for i in range(256):
        logits, _ = self(idx,start_token,src_mask)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
        # print(i,next_token)

        if (next_token == eos_token).any():
            break 

        start_token = torch.cat((start_token, next_token), dim=-1)
        # print(start_token)


    return start_token
  

model= Transformer(Transformerconfig())


class NoamOpt:

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """
        Update parameters and rate"
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `lrate` above
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
