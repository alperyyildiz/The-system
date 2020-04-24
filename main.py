from collections import OrderedDict 
import copy
import numpy as np
import torch 
import torch.nn as nn


class MAIN_OBJ():
    def __init__(self):
        super().__init__()
        self.init_params()
        
    def init_params(self):
        #default input types from pytorch documentation. 
        #User input will be processed by these functions and parameters

        self.input_types = {'conv1d': [int,int,int,int,int,int,int,self.Bool_Rework,str],
                            'conv2d': [int,int,int,int,int,int,int,self.Bool_Rework,str],
                            'conv1dTranspose': [int,int,int,int,int,int,int,self.Bool_Rework,int,str],
                            'conv2dTranspose': [int,int,int,int,int,int,int,self.Bool_Rework,int,str],
                            'LSTM': [int,int,int,self.Bool_Rework,self.Bool_Rework,int,self.Bool_Rework,int],
                            'Linear': [int,int,self.Bool_Rework],
                            'MaxPool1d': [int,int,int,int,self.Bool_Rework,self.Bool_Rework],
                            'MaxPool2d': [int,int,int,int,self.Bool_Rework,self.Bool_Rework],
                            'BatchNorm1d': [int, float, float, self.Bool_Rework, self.Bool_Rework],
                            'BatchNorm2d': [int, float, float, self.Bool_Rework, self.Bool_Rework],
                            'Dropout': [float,self.Bool_Rework],
                            'Flatten': [int,int]
                            }
        #Default layer parameters from pytorch docs.
        self.Def_Params = {'conv1d': {'in_channels': None,
                                      'out_channels':None,
                                      'kernel_size':None,
                                      'stride':1,
                                      'padding':0,
                                      'dilation':1,
                                      'groups':1, 
                                      'bias':True,
                                      'padding_mode': 'zeros'},
                           'conv1dTranspose':{'in_channels':None,
                                              'out_channels':None,
                                              'kernel_size':None, 
                                              'stride':1,
                                              'padding':0, 
                                              'output_padding':0,
                                              'groups':1, 
                                              'bias':True,
                                              'dilation':1, 
                                              'padding_mode':'zeros'},
                           'LSTM': {'input_size': None,
                                    'hidden_size': None,
                                    'num_layers': 1,
                                    'bias': True,
                                    'batch_first': True,
                                    'dropout': 0,
                                    'bidirectional': False,
                                    'train_batch_size': None},


                           'conv2d': {'in_channels':None,
                                      'out_channels':None,
                                      'kernel_size':None,
                                      'stride':1,
                                      'padding':0, 
                                      'dilation':1, 
                                      'groups':1, 
                                      'bias':True, 
                                      'padding_mode':'zeros'},
                           
                           
                           'conv2dTranspose': {'in_channels':None,
                                               'out_channels':None,
                                               'kernel_size':None,
                                               'stride':1,
                                               'padding':0,
                                               'output_padding':0, 
                                               'groups':1,
                                               'bias':True,
                                               'dilation':1,
                                               'padding_mode':'zeros'},
                           
                           'Linear': {'in_features':None,
                                      'out_features':None,
                                      'bias':True},
                           
                           'MaxPool1d': {'kernel_size':None, 
                                         'stride':1,
                                         'padding':0,
                                         'dilation':1,
                                         'return_indices':False,
                                         'ceil_mode':False},
                           'MaxPool2d': {'kernel_size':None, 
                                         'stride':1,
                                         'padding':0,
                                         'dilation':1, 
                                         'return_indices':False,
                                         'ceil_mode':False},
                           
                           'Flatten': {'start_dim':1, 
                                       'end_dim':-1},
                           
                           'BatchNorm1d': {'num_features':None, 
                                           'eps':1e-05, 
                                           'momentum':0.1, 
                                           'affine':True,
                                           'track_running_stats':True},
                           'BatchNorm2d': {'num_features':None,
                                           'eps':1e-05,
                                           'momentum':0.1,
                                           'affine':True,
                                           'track_running_stats':True},
                           'Dropout': {'p':0.5,
                                       'inplace': False}
                           }


#============================================================================INPUT REWORKS=========================================================================================

    def input_reworker(self,type, input_params):
        #Split inputs given by user
        #Set defaults if remain
        #Change dtype from str to corresponding types
        #Built in functions used which are initialized as self.input_types
        input_params = input_params.split(",")
        type_def_params = list(self.Def_Params[type].values())

        for i in range(len(input_params)):
            if input_params[i] == '-':
                if type == 'MaxPool1d' and i == 1:
                    input_params[i] = input_params[i - 1]
                else:
                    input_params[i] = type_def_params[i]

        data_type_functions = self.input_types[type]
        for i in range(len(input_params)):
            try: 
                input_params[i] = data_type_functions[i](input_params[i])
            except:
                print('Built in Function cannot be applied to param {}'.format(i))
        return input_params
        
    def Bool_Rework(self,x):
        #Convert user input from string to Boolean
        if x in ['T','True','Yes','Y','t','y'] or x is True:
            return True
        elif x in ['F','False','No','N','f','n'] or x is False:
            return False
        else:
            print('WE HAVE A PROBLEM IN BOOL_REWORK')


#============================================================================INPUT REWORKS=========================================================================================
#============================================================================TRACK SEQ AND CH=========================================================================================

    def track_seq_len_branch(self,Branch, seq_len_list):
        BRANCH = copy.deepcopy(Branch)
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)
        seq_len_list = list()
        Flat = self.check_flatten_branch(BRANCH)
        for bb_number,[TYPE,NUM] in enumerate(BB_keys):

            BB = BRANCH[TYPE + '-' + NUM]

            if TYPE == 'block':              
                seq_len_list.append(self.track_seq_len_block(BB,seq_len_list(bb_number)))
            elif TYPE == 'branch':
                sel_len_list.append(self.track_seq_len_branch(BB,seq_len_list(bb_number)))
        
        
        All_Same = all(x == seq_len_list[0] for x in seq_len_list)
        Flat = self.check_flatten_branch(BRANCH)

        if ALL_Same == False and Flat == False:
            raise Exception('You Will Have Dimension Problems \n Sequences are not equal \n Flatten is Not Used')

        return seq_len
                
    def track_seq_len_block(self, Block, seq_len):
        BLOCK = copy.deepcopy(Block)
        block_layers = list(BLOCK.keys())
        for layer in block_layers:
            seq_len = self.track_seq_len(BLOCK[layer])
        return seq_len

    def track_seq_len(self,layer,seq_length):
        #Method to track sequence length of forward input
        #Used to calculate input channels for the layer after flatten
        #Also will be used for debugging in the future

        params = block[layer][1]
        if block[layer][0] == 'conv1d':
            seq_length = int(np.floor((seq_length + 2*params[4] - params[5]*(params[2]-1)-1)/params[3] + 1))
        elif block[layer][0] == 'MaxPool1d':
            seq_length = int(np.floor((seq_length + 2*params[2] - params[3]*(params[0]-1)-1)/params[1] + 1))
        elif block[layer][0] == 'Flatten':
            seq_length =  1

        return seq_length


    def track_out_channels_branch(self,branch,out_channels,seq_len_list):
        BRANCH = copy.deepcopy(Branch)
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)
        out_channels_list = list()
        Flat = self.check_flatten_branch(BRANCH)

        for bb_num, [TYPE,NUM] in enumerate(BB_keys):
            BB = BRANCH[TYPE + '-' + NUM]
            if TYPE == 'block':
                out_channels_list.append(self.track_out_channels_block(BB,out_channels,seq_len_list[bb_num]))
            elif TYPE == 'branch':
                out_channels_list.append(self.track_out_channels_branch(BB,out_channels,seq_len_list[bb_num]))
        
        out_channels = 0
        for elem in out_channels_list:
            out_channels = out_channels + elem 
        
        return out_channels
            


    def track_out_channels_block(self, block, out_channels, seq_len):
        BLOCK = copy.deepcopy(Block)
        block_layers = list(BLOCK.keys())
        for layer in block_layers:
            out_channels = self.track_out_channels(BLOCK[layer],out_channels)
        return out_channels
            
    def track_out_channels(self,layer,out_channels, seq_len):
        
        PARAMS = layer[1]
        TYPE = layer[0]
        if TYPE in ['LSTM','conv1d']:
            out_channels = PARAMS[1]
        elif TYPE == 'Flatten':
            out_channels = seq_len * out_channels
        return out_channels

#============================================================================TRACK SEQ AND CH=========================================================================================

    def Create_Block_v2(self):
        #Creates an ordered dictionary
        #Saves layer type with parameters specified by user
        NAME = input('Enter Block Name: ')
        count =0
        Add = True
        BLOCK = OrderedDict()

        #While loop until user says stop
        while Add:
            count = count + 1
            TYPE = input('enter layer type: ')
            print('\n Seperate input with comma, write - for defalt value\n')
            print(self.Def_Params[TYPE])
            print('\n')
            input_params = input()
            input_params = self.input_reworker(TYPE, input_params)
            layer_num = str(count)
            BLOCK[layer_num] = [TYPE, input_params]
            add_value = input('You want to add more?? \n')
            Add = self.Bool_Rework(add_value)


        layer_list_str = list(Dict_Block.keys())
        out_channels = Dict_Block[layer_list_str[0]][1][1]
        #HARD CODING 2nd if
        #HARD CODING 2nd if
        #HARD CODING 2nd if
        for i,layer in enumerate(layer_list_str):
            if i is not 0:
              print('layer number is: {}'.format(layer))
              if Dict_Block[layer][0] in ['Linear','conv1d','LSTM']:
                  Dict_Block[layer][1][0] = out_channels
                  out_channels =  Dict_Block[layer][1][1]
              elif Dict_Block[layer][0] == 'Flatten':
                  out_channels =  None
   

        self.Blocks[NAME] = BLOCK


    
    def Append_Block_to_Network_v2(self, Block):
        #Append the block to the Network
        block = copy.deepcopy(Block)
        
        network_layers = list(self.network.keys())
        block_layers = list(block.keys())
        
        if self.First == True:
            self.Branch_First = False
            self.num_of_blocks = self.num_of_blocks + 1
            self.order.append('block')
        
        if self.First == False:
            #Find last layer contains out_channels as parameter
            block[block_layers[0]][1][0] = self.out_channels

            if len(network_layers) == 0:
                count = 0
            else:
                count = int(network_layers[-1])
            
            #Append block layers to network
            for lay in block_layers:
                
                count = count + 1
                self.network[str(count)] = block[lay]
                self.seq_len_processor(block[lay])
                


        #If this is the first block to be added
        #Attend feature size as in_channels
        else:
            block[block_layers[0]][1][0] = self.feature_size
            count = 0
            for lay in block_layers:
                count = count + 1
                self.network[str(count)] = block[lay]
                self.seq_len_processor(block[lay])

        self.First = False

   
    def channels_processor(self,layer,out_channels):
        params = layer[1]
        if layer[0] in ['conv1d']:
            pass



    def seq_and_ch_processor(self,layer, out_channels, seq_len):
        out_channels = self.out_channels_processor(layer, out_channels, seq_len)
        seq_len = self.seq_len_processor_v2(layer,seq_len)
        return out_channels, seq_len


    def seq_len_processor(self,layer):
        PARAMS = layer[1]
        if layer[0] == 'conv1d':
            self.seq_len = int(np.floor((self.seq_len + 2*PARAMS[4] - PARAMS[5]*(PARAMS[2]-1)-1)/PARAMS[3] + 1))

        elif layer[0] in ['MaxPool1d', 'AVERAGEPOOL']:
            self.seq_len = int(np.floor((self.seq_len + 2*PARAMS[2] - PARAMS[3]*(PARAMS[0]-1)-1)/PARAMS[1] + 1))
        if self.seq_len < 0:
            raise Exception('Seq length becomes negative')
  

    def check_flatten_block(self,block):
        Flatten = False
        for layer in list(block.keys()):
            if block[layer][0] == 'Flatten':
                Flatten = True
        return Flatten

        
    def check_flatten_branch(self,Branch):
        flat_list = list()
        BRANCH = copy.deepcopy(Branch)
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)

        for [TYPE,NUM] in BB_keys:
            BB = BRANCH[TYPE + '-' + NUM]
            if TYPE == 'block':
                flat_list.append(self.check_flatten_block(BB))
            elif TYPE == 'branch':
                flat_list.append(self.check_flatten(Branch))


        ALL_SAME = all(x == flat_list[0] for x in flat_list)
        ALL_FLAT = all(x == True for x in flat_list)

        if ALL_SAME == True:
            if ALL_FLAT == True:
                return True
            elif ALL_FLAT == False:
                return False
        elif ALL_SAME == False:
            raise Exception('Some are Flat, Some are Not')
            




    def Create_Block(self):
        #Creates an ordered dictionary
        #Saves layer type with parameters specified by user
        NAME = input('Enter Block Name: ')
        count =0
        Add = True
        Dict_Block = OrderedDict()

        #While loop until user says stop
        while Add:
            count = count + 1
            TYPE = input('enter layer type: ')
            print('\n Seperate input with comma, write - for defalt value\n')
            print(self.Def_Params[TYPE])
            print('\n')
            input_params = input()
            input_params = self.input_reworker(TYPE, input_params)
            layer_num = str(count)
            Dict_Block[layer_num] = [TYPE, input_params]
            add_value = input('You want to add more?? \n')
            Add = self.Bool_Rework(add_value)

        #Set parameters of consecutive layers
        #Notice we do not initialize layers
        #just setting parameters for initialization
        #MaxPool1d is a special case as flatten

        layer_list_str = list(Dict_Block.keys())
        out_channels = Dict_Block[layer_list_str[0]][1][1]
        #HARD CODING 2nd if
        #HARD CODING 2nd if
        #HARD CODING 2nd if
        for i,layer in enumerate(layer_list_str):
            if i is not 0:
              print('layer number is: {}'.format(layer))
              if Dict_Block[layer][0] in ['Linear','conv1d','LSTM']:
                  Dict_Block[layer][1][0] = out_channels
                  out_channels =  Dict_Block[layer][1][1]
              
   
        self.Blocks[NAME] = Dict_Block



    def Create_Branch(self):
        #Creates an ordered dictionary
        #Saves layer type with parameters specified by user
        NAME = input('Enter Branch Name: ')
        Add_layers = True
        Add_branches = True
        Dict_Branch = OrderedDict()

        count_block = 0
        #While loop until user says stop adding branches
        while Add_branches:
            count_block = count_block + 1
            print(self.Blocks)
            block_name  = input('Which block you want to add as a branch \n')
            Dict_Branch['block-' + str(count_block)] = self.Blocks[block_name]

            add_value = input('You want to add more branches?? \n')
            Add_branches = self.Bool_Rework(add_value)


        self.Branches_Created[NAME] = Dict_Branch

        
        
class NET(MAIN_OBJ):
    def __init__(self):
        super().__init__()
        self.seq_len = 250
        self.feature_size = 50

        self.Blocks = OrderedDict()
        self.Branches_Created = OrderedDict()

        self.network = OrderedDict()

        self.ALL_BLOCKS = OrderedDict()
        self.ALL_BRANCHES = OrderedDict()
        
        self.num_of_branches = 0
        self.num_of_blocks  = 0
        self.First = True

        self.order = list()



    def Append_Block_to_Network_v2(self, Block):
        #Append the block to the Network

        #Copy network to eliminate overwriting
        BLOCK = copy.deepcopy(Block)
        
        #Get the list of block layers to be added
        block_layers = list(BLOCK.keys())
        
        #if this is the first part to be added
        #set count to 0 and in_channels to feature_size
        if self.First == True:
            self.Branch_First = False
            self.num_of_blocks = self.num_of_blocks + 1
            BLOCK[block_layers[0]][1][0] = self.feature_size
            count = 0
        #if network initialized before
        #set in_channels and set count 
        elif self.First == False:
            BLOCK[block_layers[0]][1][0] = self.out_channels
            

        #track out channels and seqeunce length  
        self.out_channels, self.seq_len = self.track_seq_and_ch_block(BLOCK,self.out_channels,self.seq_len)


        #Append block layers to network
        self.num_of_blocks = self.num_of_blocks + 1 
        self.ALL_BLOCKS[str(self.num_of_blocks )] = BLOCK
        self.order.append('block')





    def Append_Branch_to_Network_v2(self,Branch,All_Flatten):
        #Append the branch to the network
        #Flatten all branches if needed 

        #Get block list in the branch
        BRANCH = copy.deepcopy(Branch)

        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)
        
        num_of_bb = len(list(BRANCH.keys()))

        #Check if the branch in the beginning
        #Provide sequence lengths if it is
        #create init_seq_len_list 
        if self.First is False:
            init_seq_len_list = [self.seq_len for x in range(num_of_bb)] 
        else:
        #
            self.Branch_First = True
            seq_list = input('Provide input sequences, separate them with comma(,) ')
            init_seq_len_list = seq_list.split(",")
        #Check if all seq_lengths are the same
        #Check if all blocks have flatten layer

        #Set In_Channels
        for [TYPE,NUM] in BB_keys:
            if self.First == False:
                BRANCH[BLOCK]['1'][1][0] = self.out_channels
            elif self.First == True: 
                BRANCH[BLOCK]['1'][1][0] = self.feature_size


        self.out_channels, self.seq_len = self.track_seq_and_ch_branch(BRANCH,init_seq_len_list)


        self.num_of_branches = self.num_of_branches + 1 
        self.ALL_BRANCHES[str(self.num_of_branches)] = BRANCH

        self.order.append('branch') 
        self.First = False

   
    def check_flatten_block(self,block):
        Flatten = False
        for layer in list(block.keys()):
            if block[layer][0] == 'Flatten':
                Flatten = True
        return Flatten

        
    def check_flatten_branch(self,Branch):
        flat_list = list()
        BRANCH = copy.deepcopy(Branch)
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)

        for [TYPE,NUM] in BB_keys:
            BB = BRANCH[TYPE + '-' + NUM]
            if TYPE == 'block':
                flat_list.append(self.check_flatten_block(BB))
            elif TYPE == 'branch':
                flat_list.append(self.check_flatten(Branch))


        ALL_SAME = all(x == flat_list[0] for x in flat_list)
        ALL_FLAT = all(x == True for x in flat_list)

        if ALL_SAME == True:
            if ALL_FLAT == True:
                return True
            elif ALL_FLAT == False:
                return False
        elif ALL_SAME == False:
            raise Exception('Some are Flat, Some are Not')
            
  
    def track_seq_and_ch(self,Layer,seq_len,out_channels):
        out_channels = self.track_out_channels(Layer,out_channels,seq_len)
        seq_len = self.track_seq_len(Layer,seq_len)
        return out_channels, seq_len

    def track_seq_and_ch_block(self, Block, seq_len, out_channels):
        BLOCK = copy.deepcopy(Block)
        out_channels = self.track_out_channels_block(BLOCK, out_channels, seq_len)
        seq_len = self.track_seq_len_block(BLOCK,seq_len)
        return out_channels, seq_len

    def track_seq_and_ch_branch(self, Branch, seq_len_list, out_channels):
        BRANCH = copy.deepcopy(Branch)
        out_channels = self.track_out_channels_branch(BRANCH, out_channels,seq_len_list)
        seq_len = self.track_seq_len_branch(BRANCH,seq_len_list)
        return out_channels, seq_len

class Model(nn.Module):
    def __init__(self, SOURCE_OBJ):
        super().__init__()
        self.play_list = nn.ModuleDict()
        self.__dict__.update(SOURCE_OBJ.__dict__)

        ind_branch = 0
        ind_block = 0
        self.layers = nn.ModuleList()

        for i in range(self.num_of_branches + self.num_of_blocks):

            if self.order[i] == 'branch':
                ind_branch = ind_branch + 1
                self.layers.append(Branch(self.ALL_BRANCHES[str(ind_branch)]))

            elif self.order[i] == 'block':
                ind_block = ind_block + 1
                self.layers.append(Block(self.ALL_BLOCKS[str(ind_block)]))


    def forward(self,x):
        for LAYER in self.layers:
            x = LAYER(x)
        return x
          
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
        
        self.hidden = (torch.randn(param[2] * NUM_OF_DIRECTIONS, self.DICT['OTHERS']['1']['batchsize'], param[1]),
                  torch.randn(param[2] * NUM_OF_DIRECTIONS, self.DICT['OTHERS']['1']['batchsize'], param[1]))


                                 

    def forward(x):
        batchsize , features , windowlength = x.shape
        x = x.reshape(batchsize, windowlength, features)
        x, self.hidden = layer(x,self.hidden)
        batchsize, windowlength, features = x.shape
        x = x.reshape(batchsize, features, windowlength)

        return x

class Block(nn.Module):
    def __init__(self,BLOCK):
        super().__init__()
        LAYER_LIST = list(BLOCK.keys())

        self.layers = nn.ModuleList()
        for LAYER in LAYER_LIST:
            TYPE = BLOCK[LAYER][0]
            ARGS = BLOCK[LAYER][1]
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


class Branch(nn.Module):
    def __init__(self,BRANCH):
        super().__init__()

        the_list = list(BRANCH.keys())
        self.BB = nn.ModuleDict()
        KEY_LIST = self.splitter(the_list) 

        for [B_or_B,num] in KEY_LIST:
            if B_or_B == 'block':
                self.BB[B_or_B + '=' + num] = Block(BRANCH[B_or_B + '-' + num])
            elif B_or_B == 'branch':
                self.BB[B_or_B + '=' + num] = Branch(BRANCH[B_or_B + '-' + num])
                

    def forward(self,x):
        block_key_list = list(self.BB.keys())
        block_key_list = self.splitter(block_key_list,'=')

        for i, [TYPE ,NUM] in enumerate(block_key_list):
            if i == 0:
                branch_concat_out = torch.Tensor(self.BB[TYPE + '=' +NUM](x))
            else:
                branch_concat_out = torch.cat([branch_concat_out,self.BB[TYPE + '=' + NUM](x)],dim = 1)
        return branch_concat_out

    def splitter(self,key_list, sep = '-'):
        num_of_keys = len(key_list)
        for key in range(num_of_keys):
            key_list[key] = key_list[key].split(sep)
        return key_list



