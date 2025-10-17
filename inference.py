from model import Transformer, Transformerconfig
import torch
from transliteration_data.data import decode_data,encode_data,data_preparation
import collections
import numpy as np

max_seq_len= 256
input_file= "/mnt/sparsh_transformers/transliteration_data/hi.hindi_original.txt"
output_file= "/mnt/sparsh_transformers/transliteration_data/hi_hindi_transliteration.txt"


# Model parameters
n_layer= 6
n_head= 8
n_embd= 512
block_size= 256
bias= True
vocab_size= 194
dropout= 0.0
device= 'cuda' if torch.cuda.is_available() else 'cpu'

def model_parameters():

    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout)


    model_args['vocab_size'] = vocab_size
    transformer_config= Transformerconfig(**model_args)
    model= Transformer(transformer_config).to(device)
    return model

char_to_int, int_to_char, vocab_size= data_preparation(input_file, output_file)


def rename_state_dict_keys(state_dict, prefix='_orig_mod.'):
    new_state_dict = collections.OrderedDict()  
    for key, value in state_dict.items():
        new_key = key.replace(prefix, '')  
        new_state_dict[new_key] = value
    return new_state_dict


def data_preprocessing():   
    sentence="मैं घर जा रही हु"
    encoded_data= [[1]+encode_data((sentence),char_to_int)+[2]]
    padded_encoded_data= np.array([seq + [0]*(max_seq_len-len(seq)) for seq in encoded_data], dtype= np.uint16)
    return padded_encoded_data

def data_postprocessing(output_data):
    padded_output_data= output_data
@torch.no_grad()
def main():
    # Load the model
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint= torch.load('/mnt/sparsh_transformers/out/ckpt.pt')
    model= model_parameters()
    model.load_state_dict(rename_state_dict_keys(checkpoint['model']))
                           
    model.eval()

    # Load the data
    data= data_preprocessing()
    input_tensor=torch.from_numpy(data.astype(np.int64)).to(device)

    output2 = model.generate(input_tensor,torch.tensor([[1]]).to(device), torch.tensor([2]).to(device))
    print(output2)
    # output_data = decode_data(output2.cpu().numpy().tolist(), int_to_char)

    # print(output_data)

if __name__ == "__main__":
    main()