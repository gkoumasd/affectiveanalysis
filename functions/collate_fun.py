import torch
from torch.nn.utils.rnn import pad_sequence

#Collate functions are functions used by PyTorch dataloader to gather batched data from dataset. 
# It loads multiple data points from an iterable dataset object and put them in a certain format. 
#Here we just use the lists we've constructed as the dataset and assume PyTorch dataloader will operate on that.
def multi_collate(batch):
    
    PAD = 1
    
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    #torch.from_numpy: Creates a torch tensor from numpy
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    #padding sequence of variable length
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    #visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    #acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    
    return sentences, labels, lengths #visual, acoustic,