

class Model(nn.Module):
    def __init__( self, SOURCE_OBJ1, SOURCE_OBJ2 ):
      

        super().__init__()
        self.__dict__.update(SOURCE_OBJ1.__dict__)
        self.__dict__.update(SOURCE_OBJ2.__dict__)
        self.TRAIN_DL = SOURCE_OBJ2.TRAIN_DL
        self.VAL_DL = SOURCE_OBJ2.VAL_DL
        self.TEST_DL = SOURCE_OBJ2.TEST_DL
        ind_branch = 0
        ind_block = 0
        self.epoch = 500
        self.lrate = 0.001
        self.Loss_FUNC = F.mse_loss
        self.layers = nn.ModuleList()


        print('\n feature size: {} \n ALL BLOCKS: {} \n  '.format(self.feature_size, self.ALL_BLOCKS))
        for i in range(self.num_of_branches + self.num_of_blocks):

            if self.order[i] == 'branch':
                ind_branch = ind_branch + 1
                self.layers.append( Branch( self.ALL_BRANCHES[ str(ind_branch )]))

            elif self.order[i] == 'block':
                ind_block = ind_block + 1
                self.layers.append( Block( self.ALL_BLOCKS[ str(ind_block )]))


    def forward(self,x):
        for LAYER in self.layers:
            x = LAYER(x)
        return x

    def loss_batch(self, TR_INP, TR_OUT, opt=None):
        model_out = self(TR_INP)
        loss = self.Loss_FUNC(model_out, TR_OUT)
        if opt is not None:
            loss.backward(retain_graph=True)
            opt.step()
            opt.zero_grad()
        return loss.item(), len(TR_INP)


    def fit(self):
        self.hist = list()
        self.hist_valid = list()

        for epoch in range(self.epoch):
            batch_loss = 0
            count = 0 
            self.train()            
            for TR_INP, TR_OUT in self.TRAIN_DL:
                losses, nums = self.loss_batch(TR_INP, TR_OUT, opt = self.optimizer)
                batch_loss = batch_loss + losses
                count = count + 1
            train_loss = batch_loss / count
            self.hist.append(train_loss)
            

            self.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self.loss_batch(TR_INP, TR_OUT) for TR_INP, TR_OUT in self.VAL_DL]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            print('Train Loss:  {}  and Validation Loss:  {}'.format(train_loss, val_loss))

            self.hist_valid.append(val_loss)

            print('{}:   {}'.format(epoch,val_loss))
        




class LST(nn.Module):
    def __init(self,param):
        super().__init__()
        
        BSIZEOLSUN = param[-1]
        self.init_hidden_states(param)

    def init_hidden_states(self,param):
        if param[5] == 2:
            NUM_OF_DIRECTIONS = 2
        else:
            NUM_OF_DIRECTIONS = 1
        
        self.hidden = (torch.randn(param[2] * NUM_OF_DIRECTIONS, self.batch_size, param[1]),
                       torch.randn(param[2] * NUM_OF_DIRECTIONS, self.batch_size, param[1]))      

    def forward(x):
        batchsize , features , windowlength = x.shape
        x = x.reshape(batchsize, windowlength, features)
        x, self.hidden = layer(x,self.hidden)
        batchsize, windowlength, features = x.shape
        x = x.reshape(batchsize, features, windowlength)
        return x

    def splitter(self,key_list, sep = '-'):
        key_list = copy.deepcopy(key_list)
        new_key_list = list()
        num_of_keys = len(key_list)
        for key in range(num_of_keys):
            new_key_list.append(key_list[key].split(sep))
        return new_key_list


class Block(nn.Module):
    def __init__(self,BLOCK):
        super().__init__()
        ENTITY_LIST = list(BLOCK.keys())

        self.layers = nn.ModuleList()
        KEY_LIST = self.splitter(ENTITY_LIST) 

        for i, [ ENTITY, num ] in enumerate(KEY_LIST):
            ENTITY_KEY =  ENTITY + '-' + num
            
            if ENTITY == 'block':
                self.layers.append( Block(BLOCK[ ENTITY_KEY ]))
            elif ENTITY == 'branch':
                self.layers.append( Branch(BLOCK[ ENTITY_KEY ]))
            elif ENTITY == 'layer':
                TYPE = BLOCK[ ENTITY_KEY ][0]
                ARGS = BLOCK[ ENTITY_KEY ][1]
                self.layers = self.layer_add(self.layers,TYPE,*ARGS)

    def forward(self,x):
        for LAYER in self.layers:
            x = LAYER(x)
        return x

                    
    def layer_add(self,submodule,key,*args):
        submodule.append(self.layer_set(key,*args))
        return submodule


    def layer_set(self,key,*args):
        ## push args into key layer type, return it
        ## push args into key layer type, return it
        if key == 'conv1d':
            return nn.Conv1d(*args)
        elif key == 'LSTM':
            return LST(*args[:-1])
        elif key == 'Linear':
            return nn.Linear(*args)
        elif key == 'Dropout':
            return nn.Dropout(*args)
        elif key == 'BatchNorm1d':
            return nn.BatchNorm1d(*args)
        elif key == 'Flatten':
            return nn.Flatten()
        elif key == 'ReLU':
            return nn.ReLU()
        elif key == 'MaxPool1d':
            return nn.MaxPool1d(*args)

    def splitter(self,key_list, sep = '-'):
        key_list = copy.deepcopy(key_list)
        new_key_list = list()
        num_of_keys = len(key_list)
        for key in range(num_of_keys):
            new_key_list.append(key_list[key].split(sep))
        return new_key_list


class Branch(nn.Module):
    def __init__(self,BRANCH):
        super().__init__()
        the_list = list(BRANCH.keys())
        self.BB = nn.ModuleDict()
        KEY_LIST = self.splitter(the_list) 
        self.forward_keys = list()
        for [ TYPE, NUM ] in KEY_LIST:
            the_key = TYPE + '-' + NUM


            if TYPE == 'block':
                self.BB[ the_key ] = Block(BRANCH[ the_key ])
                self.forward_keys.append( the_key )
            elif TYPE == 'branch':
                self.BB[ the_key ] = Branch(BRANCH[ the_key ])
                self.forward_keys.append( the_key )
            print('FORWARD KEYS ARE: {}'.format(self.forward_keys))
    def forward(self,x):
        for i,  TYPE in enumerate(self.forward_keys):
            
            if i == 0:
                branch_concat_out = torch.Tensor( self.BB[ TYPE ](x) )
            else:
                branch_concat_out = torch.cat( [ branch_concat_out, self.BB[ TYPE ](x) ], dim = 1)

        return branch_concat_out



    def splitter(self, KEYS, sep = '-'):

        key_list = copy.deepcopy(KEYS)
        num_of_keys = len(key_list)
        new_key_list = list()
        for key in range(num_of_keys):
            new_key_list.append(key_list[key].split(sep))
        return new_key_list


    def read_data(self):

        with open("DATA/in_use_info.txt") as file:
            data_info = file.readlines() 

        
        data_root_name, feature_size, seq_len = self.split_data_info(data_info)
        train = torch.load(data_root_name + '_TRAIN')
        val = torch.load(data_root_name + '_VAL')
        test = torch.load(data_root_name + '_TEST')
        return train, val, test



    def split_data_info(self,data_info):
        print(data_info)
        data_root_name = data_info[0]
        
        feature_size_info = data_info[1].split('=')
        seq_len_info = data_info[2].split('=')

        feature_size = int(feature_size_info[1])
        seq_len = int(seq_len_info[1])

        return data_root_name[:-2], feature_size, seq_len
