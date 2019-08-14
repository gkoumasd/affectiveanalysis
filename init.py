from functions.data import *
from functions.collate_fun import *
from functions.word2vec import *
from models.LFLSTM import *

from torch.optim import Adam, SGD



#Specify dataset
DATASET = md.cmu_mosi

# define your different modalities - refer to the filenames of the CSD files
visual_field = 'CMU_MOSI_VisualFacet_4.1'
acoustic_field = 'CMU_MOSI_COVAREP'
text_field = 'CMU_MOSI_ModifiedTimestampedWords'

label_field = 'CMU_MOSI_Opinion_Labels'

text_size = 300
visual_size = 47
acoustic_size = 74


torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
CUDA = torch.cuda.is_available()


#Hyper-parameters
batch_sz = 56
MAX_EPOCH = 2

#Download data
download_data(DATASET)

#Load data
dataset = load_dataset(visual_field,acoustic_field,text_field)

#Alignment process
# first we align to words with averaging, collapse_function receives a list of functions
dataset.align(text_field, collapse_functions=[avg])
# we add and align to lables to obtain labeled segments
# this time we don't apply collapse functions so that the temporal sequences are preserved
label_recipe = {label_field: os.path.join(DATA_PATH, label_field + '.csd')}
dataset.add_computational_sequences(label_recipe, destination=None)
dataset.align(label_field)

#Obtain train/dev/test splits
train_split, dev_split, test_split = split_dataset(DATASET)

#data processing
train, dev, test, word2id = data_processing(dataset,text_field,visual_field,acoustic_field,label_field,train_split,dev_split,test_split)


# construct dataloaders, dev and test could use around ~X3 times batch size since no_grad is used during eval
train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate)
dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate)


temp_loader = iter(DataLoader(test, shuffle=True, batch_size=8, collate_fn=multi_collate))
batch = next(temp_loader)

print(batch[0].shape) # word vectors, padded to maxlen
print(batch[1]) # labels
print(batch[2]) # lengths




#training

# define some model settings and hyper-parameters
input_sizes = [text_size]
hidden_sizes = [int(text_size * 1.5)]
fc1_size = sum(hidden_sizes) // 2
dropout = 0.25
output_size = 1
curr_patience = patience = 8
num_trials = 3
grad_clip_value = 1.0
weight_decay = 0.1


if os.path.exists(CACHE_PATH):
    pretrained_emb, word2id = torch.load(CACHE_PATH)
elif WORD_EMB_PATH is not None:
    pretrained_emb = load_emb(word2id, WORD_EMB_PATH)
    torch.save((pretrained_emb, word2id), CACHE_PATH)
else:
    pretrained_emb = None
    
    
#modality: sentences, visual, or acoustic    
model = LFLSTM(input_sizes, hidden_sizes, fc1_size, output_size, dropout, len(word2id)) 

if pretrained_emb is not None:
    model.embed.weight.data = pretrained_emb   
    
model.embed.requires_grad = False

optimizer = Adam([param for param in model.parameters() if param.requires_grad], weight_decay=weight_decay)    


if CUDA:
    model.cuda()
    
criterion = nn.L1Loss(reduction='sum')    
criterion_test = nn.L1Loss(reduction='sum')
best_valid_loss = float('inf')

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
lr_scheduler.step() # for some reason it seems the StepLR needs to be stepped once first

train_losses = []
valid_losses = []

for e in range(MAX_EPOCH):
    model.train()
    train_iter = tqdm_notebook(train_loader)
    train_loss = 0.0
    for batch in train_iter:
        model.zero_grad()
        t, y, l = batch
        batch_size = t.size(0)
      
        if CUDA:
            t = t.cuda()
            y = y.cuda()
            l = l.cuda()
            
        #It calls forward function    
        y_tilde = model(t, l)
        
        loss = criterion(y_tilde, y)
        loss.backward()
        
        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
        optimizer.step()
        train_iter.set_description(f"Epoch {e}/{MAX_EPOCH}, current batch loss: {round(loss.item()/batch_size, 4)}")
        train_loss += loss.item()
        
    train_loss = train_loss / len(train)
    train_losses.append(train_loss) 

    print(f"Training loss: {round(train_loss, 4)}")    