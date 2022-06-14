import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import math


class EEGCogNet(Dataset):
  def __init__(self, npdata):
    self.len = npdata.shape[0]

    self.trainX = torch.from_numpy(npdata[:,:-1])
    self.trainY = torch.from_numpy(npdata[:,-1])
    for label in self.trainY:
      label -= 1

  def __getitem__(self, index):
      return self.trainX[index], self.trainY[index]

  def __len__(self):
      return self.len

#def train, test
def train_loop(epoch, train_loader, model, criterion, optimizer, device):
    """
    Performs one epoch of training the model through the dataset stored in dataloader
    Optimizer default: Adam
    Loss default: Cross Entropy
    Returns training metrics (loss for regressions, loss and accuracy for classification) of the epoch 
    """

    training_loss = 0.0
    for batch, data in enumerate(train_loader):
   
      X, y = data
      X, y = X.to(device), y.to(device)
        
      optimizer.zero_grad()

      # Compute prediction and loss
      #Add parameter to automize this 
      model.double()
      #For LSTM and CNN use the following line
      #output = model(X) 
      #For EEG_Transformer use the following line
      output, _ = model(X)
      loss = criterion(output, y.long())

      # Backpropagation and optimization 
      loss.backward()
      optimizer.step()

      # Add up metrics
      training_loss += loss.item()
        
      if batch % 50 == 49:
        print("Epoch: %d | Batch Index: %d | Running loss: %.3f" % (epoch + 1, batch + 1, training_loss))
        training_loss = 0.0

          

def test_loop(test_loader, model):
    """
    gives total correct predictions and evaluate the model
    """

    val_loss = correct = total = 0
    with torch.no_grad():
      for X, y in test_loader:

        # Move tensors to GPU if CUDA available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)

        output, _ = model(X)
        _, prediction = torch.max(output.data, dim = 1)

        total += y.size(0)
        correct += (prediction == y).sum().item()
      
    print("Accuracy on test set is %.3f" % (correct / total))

    return correct/total

class DL_Model():

  def __init__(self, model, batch_size, epoch = 50, lr = 0.1):

    self.epoch = epoch
    self.lr = lr
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size

    if model == "cnn":
      self.net = CNN(self.batch_size)
    elif model == "lstm":
      self.net = LSTM()
    elif model == "transformer":
      self.net = EEG_Transformer()
    else:
      raise ValueError("Choose a valid model")




  def fit(self, train_loader):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.net.parameters(), self.lr)

    epoch = self.epoch

    for i in range(epoch):
      print("------------------------")
      print("Epoch:", i)

      train_loop(i, train_loader, self.net, criterion, optimizer, device)
      



  



class CNN(torch.nn.Module):
  
  def __init__(self, batch_size, kernel_size = 3, in_channel = 1, out_channel = 3, pool_size = 2, padding = 1):
    
    super(CNN, self).__init__()
    self.conv = torch.nn.Conv1d(in_channel, out_channel, kernel_size = kernel_size, padding = padding)
    self.maxPool = torch.nn.MaxPool1d(pool_size)
    self.batchNorm = torch.nn.BatchNorm1d(num_features = out_channel)
    self.fc = torch.nn.Linear(30, 5)
    self.batch_size = batch_size


  def forward(self, x):
    
    #print(x.size(), "before")
    x = torch.unsqueeze(x, 1)
    #print(x.size(), "before")
    x = self.conv(x.float())
    #print(x.size(), "before")
    x = self.batchNorm(x)
    #print(x.size(), "before")
    x = self.maxPool(x)
    #print(x.size(), "before")
    x = F.relu(x)
    #print(x.size(), "before")
    x = x.view(self.batch_size, -1)
    #print(x.size(), "after")
    x = self.fc(x)
    return x


class LSTM(torch.nn.Module):

  def __init__(self, hidden_size = 64, input_size = 1, num_layers = 1, batch_first = True, bidirectional = False, drop_out = 0.2):

    super(LSTM, self).__init__()
    self.num_layers = num_layers
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.lstm = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = batch_first,\
                              bidirectional = bidirectional, dropout = drop_out)
    self.fc = torch.nn.Linear(hidden_size, 5)

    
  def forward(self, x):
    #x.double()
    x = torch.unsqueeze(x, 2)
    #init_hidden = torch.zeros(self.num_layers, BATCH_SIZE, self.hidden_size)
    #print("got to here", x.shape, init_hidden.shape)
    out,_ = self.lstm(x)
    #print("out size", out.shape)
    #print("hidden", hidden[0].shape)
    final_hidden = out[:, -1, :]
    final_hidden = self.fc(final_hidden)
    return final_hidden


#1 encoding in positonal infomratoin 
class Positional_Encoding(nn.Module):

  def __init__(self, seq_length = 20, embedding_size = 72, drop_out = 0.1):

    super(Positional_Encoding, self).__init__()
    #here plus 1 because need additional dimension for clf token
    self.conv = torch.nn.Conv1d(in_channels = 1, out_channels = embedding_size, kernel_size = 3, padding = 1)
    self.positional_embedding = torch.zeros(seq_length, embedding_size)
    
    #Using Google's sinusoidal positional encoding formula here
    positions = torch.arange(0., seq_length).unsqueeze(1)
    div = torch.exp(torch.arange(0., embedding_size, 2) * -(math.log(10000.0)/embedding_size))

    self.positional_embedding[:, 0::2] = torch.sin(positions*div)
    self.positional_embedding[:, 1::2] = torch.cos(positions*div)
    token_position = torch.zeros(1, embedding_size)
    self.positional_embedding = torch.cat((token_position, self.positional_embedding))
    self.positional_embedding = self.positional_embedding.unsqueeze(0)

    self.classifier_token = torch.zeros(1, 1, embedding_size)
    self.dropout = nn.Dropout(drop_out)

  def forward(self, x):
    x = torch.unsqueeze(x, 2)
    clf_token = self.classifier_token.expand(x.size(dim = 0), -1, -1)
    #(bs, 1, embedding)
    x = x.transpose(-1, -2)
    x = self.conv(x)
    x = x.transpose(-1, -2)
    x = torch.cat((clf_token, x), dim = 1) #(bs, 21, 1)
    embeddings = x + self.positional_embedding
    embeddings = self.dropout(embeddings)
    return embeddings

#self attention making qkv

class Attention(nn.Module):
  
  def __init__(self, embedding_size = 72, num_heads = 6,drop_out = 0.1):
      super(Attention,self).__init__()
      self.num_attention_heads= num_heads
      self.attention_head_size = int(embedding_size/ self.num_attention_heads)  
      self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

      self.query = nn.Linear(embedding_size, self.all_head_size) #wm,768->768，Wq（768,768）
      self.key = nn.Linear(embedding_size, self.all_head_size) #wm,768->768,Wk（768,768）
      self.value = nn.Linear(embedding_size, self.all_head_size) #wm,768->768,Wv（768,768）
      self.out = nn.Linear(embedding_size, embedding_size)  # wm,768->768
      self.attn_dropout = nn.Dropout(drop_out) # 0.1
      self.proj_dropout = nn.Dropout(drop_out) # seperated for readibility also 0.1

      self.softmax = nn.Softmax(dim=-1)

  def transpose_for_scores(self, x):
      new_x_shape = x.size()[:-1] + (
      self.num_attention_heads, self.attention_head_size)  # wm,(bs,21)+(12,64)=(bs,21,12,64)
      x = x.view(*new_x_shape)
      return x.permute(0, 2, 1, 3)  # wm, (bs,12,21,64)

  def forward(self, hidden_states):
      # hidden_states：(bs,21,768)
      mixed_query_layer = self.query(hidden_states)#wm,768->768
      mixed_key_layer = self.key(hidden_states)#wm,768->768
      mixed_value_layer = self.value(hidden_states)#wm,768->768

      query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,12,21,64)
      key_layer = self.transpose_for_scores(mixed_key_layer)
      value_layer = self.transpose_for_scores(mixed_value_layer)

      attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) #q*k^t（bs,12,21,64)
      attention_scores = attention_scores / math.sqrt(self.attention_head_size)#regularize
      attention_probs = self.softmax(attention_scores)# softmax to calculate prob
      weights = attention_probs # weights
      attention_probs = self.attn_dropout(attention_probs)

      context_layer = torch.matmul(attention_probs, value_layer)# multiply v and prob
      context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
      new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)#wm,(bs,21)+(768,)=(bs,21,768)
      context_layer = context_layer.view(*new_context_layer_shape)
      attention_output = self.out(context_layer)
      attention_output = self.proj_dropout(attention_output)
      return attention_output, weights #wm,(bs,21,768),(bs,21,21)


#3 multi layer perception feed forward 

class MLP(nn.Module):

  def __init__(self, embedding_size = 72, mlp_dim = 288, drop_out = 0.1):
      
      super(MLP, self).__init__()
      self.fc1 = nn.Linear(embedding_size, mlp_dim)
      self.fc2 = nn.Linear(mlp_dim, embedding_size)
      self.act_fn = F.gelu#wm activation funciton
      self.dropout = nn.Dropout(drop_out)
      
      self._init_weights()
    
  def _init_weights(self):
    #See for this paper “Understanding the difficulty of training deep feedforward neural networks” 
    #Basically intializing weight used in this mlp training
      nn.init.xavier_uniform_(self.fc1.weight)
      nn.init.xavier_uniform_(self.fc2.weight)
      #Fills the given 2-dimensional matrix with values drawn from a normal distribution parameterized by mean and std
      nn.init.normal_(self.fc1.bias, std=1e-6)
      nn.init.normal_(self.fc2.bias, std=1e-6)


  def forward(self, x):

      x = self.fc1(x)#wm,786->3072
      x = self.act_fn(x) # non linear transformation
      x = self.dropout(x)
      x = self.fc2(x) #wm3072->786
      x = self.dropout(x)
      return x

#4.Reusable blocks consisting of mlp and self-atttention

class Block(nn.Module):
    def __init__(self, embedding_size = 72):

        super(Block, self).__init__()
        self.hidden_size = embedding_size #wm,768
        self.attention_norm = nn.LayerNorm(embedding_size)#wm，layer normalization 
        self.ffn_norm = nn.LayerNorm(embedding_size)
        
        self.ffn = MLP()
        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h #res block to prevent degration

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h #res block
        return x, weights

#5.Encoder implementation (N*Block)

class Encoder(nn.Module):
    def __init__(self, embedding_size = 72, num_layers = 6):
        super(Encoder, self).__init__()

        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embedding_size, eps=1e-6)

        for _ in range(num_layers):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []

        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)

        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

#6 Complete Transformer w/ Encoder blocks

class Transformer(nn.Module):

    def __init__(self, embedding_size = 72):
        super(Transformer, self).__init__()

        self.embeddings = Positional_Encoding()
        self.encoder = Encoder()

    def forward(self, input_ids):

        embedding_output = self.embeddings(input_ids)#wm,（bs,20,768)
        encoded, attn_weights = self.encoder(embedding_output)#wm,（bs,20,768)
        return encoded, attn_weights#（bs,21,768）

#7 Transformer used for EEG Classification

class EEG_Transformer(nn.Module):
    def __init__(self):
        super(EEG_Transformer, self).__init__()

        num_classes = 5
        embedding_size = 72

        self.transformer = Transformer()
        self.head = nn.Linear(embedding_size, num_classes) #wm,768-->10

    def forward(self, x, labels=None):

        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        return logits, attn_weights

'''
#Add following code to the four .py files:
            #define BATCH_SIZE = 64 at the top of each file
            tcr = EEGCogNet(name_of_data_for_training)
            train_loader = DataLoader(dataset = tcr, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

            #hack: change the variable name from 'cnn' to 'model' later (more meaningul when including other DL models)
            model = DL_Model(name, BATCH_SIZE)
            model.fit(train_loader)
            
            #This is the part for testing
            data_test = folds[-1]
            testdata = np.array(data_test)
            tcr_test = TCR(testdata)
            test_loader = DataLoader(dataset = tcr_test, batch_size = BATCH_SIZE, shuffle = False, num_workers = NUM_PROCESS)
            cur_accuracy = test_loop(test_loader, model.net)
            
            ####################################### Abdel Note: Training and testing end here ############################################################
            print("executed", execute_counter, "times")
            accuracy = accuracy + cur_accuracy
            execute_counter += 1
            accs.append(cur_accuracy)
'''