from collections import OrderedDict 
import copy
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from itertools import compress 


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

    def track_seq_len(self,layer,seq_len):
        #Method to track sequence length of forward input
        #Used to calculate input channels for the layer after flatten
        #Also will be used for debugging in the future

        params = layer[1]
        if layer[0] == 'conv1d':
            seq_len = int(np.ceil((seq_len + 2*params[4] - params[5]*(params[2]-1)-1)/params[3] + 1))
        elif layer[0] == 'MaxPool1d':
            seq_len = int(np.ceil((seq_len + 2*params[2] - params[3]*(params[0]-1)-1)/params[1] + 1))
        elif layer[0] == 'Flatten':
            seq_len =  1
        return seq_len


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
            branch_or_block = input('You want to add block or branch \n Type 1 or 2')

            if branch_or_block == '1':
                print(self.Blocks.keys())
                block_name  = input('Which block you want to add as a branch \n')
                Dict_Branch['block-' + str(count_block)] = self.Blocks[block_name]

            elif branch_or_block == '2':
                print(self.Branches_Created.keys())
                branch_name  = input('Which block you want to add as a branch \n')
                Dict_Branch['branch-' + str(count_block)] = self.Branches_Created[branch_name]

            add_value = input('You want to add more branches?? \n')
            Add_branches = self.Bool_Rework(add_value)


        self.Branches_Created[NAME] = Dict_Branch

#============================================================================ NET =========================================================================================
#============================================================================ NET =========================================================================================
#============================================================================ NET =========================================================================================


class NET(MAIN_OBJ):
    def __init__(self, INP_OBJ):
        super().__init__()


        self.seq_len = INP_OBJ.window_len
        self.feature_size = INP_OBJ.feature_size
        self.batch_size = INP_OBJ.batch_size
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
            self.out_channels = self.feature_size

            count = 0

        BLOCK = self.push_ch_to_block(BLOCK,self.out_channels)
        
        #track out channels and seqeunce length  
        self.out_channels, self.seq_len = self.track_seq_and_ch_block(BLOCK,self.out_channels,self.seq_len)

        #Append block layers to network
        self.num_of_blocks = self.num_of_blocks + 1 
        self.ALL_BLOCKS[str(self.num_of_blocks )] = BLOCK
        self.order.append('block')
        self.First = False

    def push_ch_to_block(self,BLOCK,out_channels):
        BLOCK = copy.deepcopy(BLOCK)

        BLOCK['1'][1][0] = out_channels
        return BLOCK

    def push_ch_to_branch(self,BRANCH,out_channels):
        BRANCH = copy.deepcopy(BRANCH)
        keys = list(BRANCH.keys())
        keys = self.splitter(keys)
        for [TYPE,NUM] in keys:
            if TYPE == 'block':
                BRANCH[TYPE + '-' + NUM] = self.push_ch_to_block( BRANCH[TYPE + '-' + NUM], out_channels )
            elif TYPE == 'branch':
                BRANCH[TYPE + '-' + NUM] = self.push_ch_to_branch( BRANCH[TYPE + '-' + NUM] , out_channels )
        return BRANCH


   
    def check_flatten_block(self,block):
        BLOCK = copy.deepcopy(block)
        Flatten = False
        for layer in list(block.keys()):
            if BLOCK[layer][0] == 'Flatten':
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
                flat_list.append(self.check_flatten_branch(BB))


        ALL_SAME = all(x == flat_list[0] for x in flat_list)
        ALL_FLAT = all(x == True for x in flat_list)

        if ALL_SAME == True:
            if ALL_FLAT == True:
                return True
            elif ALL_FLAT == False:
                return False
        elif ALL_SAME == False:
            raise Exception('Some are Flat, Some are Not')
            
  
    def track_seq_and_ch(self,Layer,out_channels,seq_len):
        LAYER = copy.deepcopy(Layer)
        out_channels = self.track_out_channels(LAYER,out_channels,seq_len)
        seq_len = self.track_seq_len(LAYER,seq_len)
        print(seq_len)
        return out_channels, seq_len

    def track_seq_and_ch_block(self, BLOCK, out_channels, seq_len):
        BLOCK = copy.deepcopy(BLOCK)

        block_layers = list(BLOCK.keys())
        for layer in block_layers:
            out_channels, seq_len = self.track_seq_and_ch(BLOCK[layer], out_channels, seq_len)
        return out_channels, seq_len


    def track_seq_and_ch_branch(self, BRANCH, out_channels, seq_len_list):
        BRANCH = copy.deepcopy(BRANCH)

        #List TYPE-NUM in branch items as BB_keys
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)
        final_seq_len_list = list()
        out_channels_list = list()
        for i, [TYPE,NUM] in enumerate(BB_keys):
            BB = OrderedDict(BRANCH[TYPE + '-' + NUM])
            seq_len = seq_len_list[i]

            if TYPE == 'block':
                out_channels, seq_len = self.track_seq_and_ch_block(BB, out_channels, seq_len)
                final_seq_len_list.append(seq_len)
                out_channels_list.append(out_channels)

            elif TYPE == 'branch':
                temp_branch_sll = [seq_len for _ in range(len(BB.keys()))]
                out_channels, seq_len = self.track_seq_and_ch_branch(BB,out_channels,temp_branch_sll)
                final_seq_len_list.append(seq_len)
                out_channels_list.append(out_channels)


        All_Same = all(x == final_seq_len_list[0] for x in final_seq_len_list)
        Flat = self.check_flatten_branch(BRANCH)

        if Flat == False:
            if All_Same == False:
                raise Exception('You Will Have Dimension Problems \n Sequences are not equal \n Flatten is Not Used')
            elif All_same == True:
                seq_len = final_seq_len_list[0]
        elif Flat == True:
            seq_len = 1
            out_channels = 0
            for t, item in enumerate(out_channels_list):
                out_channels = out_channels + item

        return out_channels, seq_len

    def splitter(self,key_list, sep = '-'):
        num_of_keys = len(key_list)
        for key in range(num_of_keys):
            key_list[key] = key_list[key].split(sep)
        return key_list



    def Append_Branch_to_Network_v2(self,BRANCH):

        #Append the branch to the network
        #Flatten all branches if needed 
        BRANCH = copy.deepcopy(BRANCH)

        #Get block list in the branch
        
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)
        
        num_of_bb = len(list(BRANCH.keys()))

        #Check if the branch in the beginning
        #Provide sequence lengths if it is
        #create init_seq_len_list 
        if self.First is False:
            seq_len = self.seq_len
            init_seq_len_list = [seq_len for x in range(num_of_bb)] 
        else:
            self.out_channels = self.feature_size
            self.Branch_First = True
            seq_list = input('Provide input sequences, separate them with comma(,)')
            init_seq_len_list = seq_list.split(",")

        #Check if all seq_lengths are the same
        #Check if all blocks have flatten layer


        #Set In_Channels
        BRANCH = self.push_ch_to_branch(BRANCH,self.out_channels)

        self.out_channels, self.seq_len = self.track_seq_and_ch_branch(BRANCH, self.out_channels, init_seq_len_list)



        self.num_of_branches = self.num_of_branches + 1 
        self.ALL_BRANCHES[str(self.num_of_branches)] = BRANCH

        self.order.append('branch') 
        self.First = False
#============================================================================ NET =========================================================================================
#============================================================================ NET =========================================================================================
#============================================================================ NET =========================================================================================







#============================================================================ DATA =========================================================================================
#============================================================================ DATA =========================================================================================
#============================================================================ DATA =========================================================================================

class Data():

    def __init__(self, batch_size = 32, window_len = 24, out_size = 4 ):      
        self.batch_size = batch_size
        self.window_len = window_len
        self.out_size = out_size

    def sliding_window_df( self, series, OUTPUT = False):
        TS = copy.deepcopy(series)
        window_len = self.window_len
        out_size = self.out_size

        num_of_windows = len(TS) - window_len - out_size + 1
        if OUTPUT == False:
            window_list = [ TS[ i : i + window_len ] for i in range(num_of_windows)]
        elif OUTPUT == True:
            window_list = [ TS[ i + window_len: i + window_len + out_size ] for i in range(num_of_windows)]
        return window_list
    



    def Preprocess( self, data, date_col_name = 'Date', out_col_name = 'Close', feature_dim = 1,  fft_degrees = [ 3, 6, 9] ):
        batch_size = self.batch_size
        window_len = self.window_len
        out_size = self.out_size
        DATA = copy.deepcopy(data)
        if feature_dim == 0:
            DATA = DATA.transpose()
        DATA = DATA.dropna( axis = 'columns', how = 'all' ).ffill()
        DATA = DATA.dropna( how = 'any' )

        out_col = list(compress(range(len(DATA.keys() == 'Close')), DATA.keys() == 'Close'))[0]


        #APPEND FOURIER AS FEATURE 
        DATA = self.add_fft_to_features(DATA, out_col_name, fft_degrees)

        #SPLIT DATA WITH SPECIFIED PERCENTAGES
        percentages  = {'train': 0.7, 'val': 0.2, 'test': 0.1 }
        TRAIN, VAL, TEST = self.train_val_test_split( DATA, percentages, output_size = out_size , window_len = window_len )

        #SAVE DATES 
        #SAVE DATES 
        self.date_TRAIN, self.date_VAL, self.date_TEST = TRAIN[date_col_name], VAL[date_col_name], TEST[date_col_name]

        #DROP DATES
        #DROP DATES 
        TRAIN, VAL, TEST = TRAIN.drop( [date_col_name], axis = 1 ), VAL.drop( [date_col_name], axis = 1 ), TEST.drop( [date_col_name], axis = 1 )

        #NORMALIZE DATA
        #NORMALIZE DATA
        TRAIN, STATS = self.Normalize_Data_Part( TRAIN )
        VAL = self.Normalize_Data_Part( VAL, feature_stats_list = STATS )
        TEST = self.Normalize_Data_Part( TEST, feature_stats_list = STATS )
        
        #SAVE OUTS
        #SAVE OUTS
        TRAIN_OUT, VAL_OUT, TEST_OUT = TRAIN[out_col], VAL[out_col], TEST[out_col]

        #DROP OUTS
        #DROP OUTS 
        TRAIN, VAL, TEST = np.delete(TRAIN, out_col, axis = 1 ), np.delete(VAL, out_col, axis = 1 ), np.delete(TEST, out_col, axis = 1 )
        

        #SLIDING WINDOW
        #SLIDING WINDOW  
        self.date_TRAIN, self.date_VAL, self.date_TEST = self.sliding_window_df(self.date_TRAIN), self.sliding_window_df(self.date_VAL), self.sliding_window_df(self.date_TEST)
        TRAIN, VAL, TEST =  np.array(self.sliding_window_df(TRAIN)).swapaxes(1,2),  np.array(self.sliding_window_df(VAL)).swapaxes(1,2),  np.array(self.sliding_window_df(TEST)).swapaxes(1,2)
        TRAIN_OUT, VAL_OUT, TEST_OUT =  np.array( self.sliding_window_df( TRAIN_OUT, OUTPUT = True ) ),  np.array( self.sliding_window_df( VAL_OUT, OUTPUT = True) ),  np.array( self.sliding_window_df( TEST_OUT, OUTPUT = True ) )
        
        #CONVERT DATA TO TENSOR
        #CONVERT DATA TO TENSOR
        total_nan = list( [np.count_nonzero(~np.isnan(TRAIN)),
                         np.count_nonzero(~np.isnan(VAL)),
                         np.count_nonzero(~np.isnan(TEST)),
                         np.count_nonzero(~np.isnan(TRAIN_OUT)),
                         np.count_nonzero(~np.isnan(VAL_OUT)),
                         np.count_nonzero(~np.isnan(TEST_OUT))] ) 
        TRAIN, VAL, TEST = torch.Tensor( TRAIN ), torch.Tensor( VAL ), torch.Tensor( TEST )
        TRAIN_OUT, VAL_OUT, TEST_OUT = torch.Tensor( TRAIN_OUT  ), torch.Tensor(VAL_OUT ), torch.Tensor( TEST_OUT )
        self.feature_size = TRAIN.shape[1]
        #COMBINE INPS WITH OUTS
        #COMBINE INPS WITH OUTS
        TRAIN_DS, VAL_DS, TEST_DS = TensorDataset( TRAIN, TRAIN_OUT ), TensorDataset( VAL, VAL_OUT ), TensorDataset( TEST, TEST_OUT )

        #BATCH ALL DATA
        #BATCH ALL DATA
        self.TRAIN_DL, self.VAL_DL, self.TEST_DL = DataLoader( TRAIN_DS, batch_size ), DataLoader( VAL_DS, batch_size ), DataLoader( TEST_DS, batch_size )


    def train_val_test_split( self, series, percentages, output_size,  window_len = 20,  batch_size = 8 ):
        TS = copy.deepcopy(series)
        length = len(TS)
        out_size = output_size
        val_len = int(self.round_data_size( length * percentages['val'] - window_len - out_size, batch_size ) + window_len + out_size)
        test_len = int(self.round_data_size( length * percentages['test'] - window_len - out_size, batch_size ) + window_len + out_size)
        train_len = int(self.round_data_size(length - test_len - val_len - window_len - out_size, batch_size ) + window_len + out_size)
        
        train_data = TS.iloc[ - val_len - test_len - train_len : - val_len - test_len , : ]
        val_data = TS.iloc[- val_len - test_len : -test_len , : ]
        test_data = TS.iloc[ - test_len : -1 , : ] 

        return train_data, val_data, test_data

    def add_fft_to_features( self, data, col_name, degrees ):
        DATA = copy.deepcopy(data)
        close_fft = np.fft.fft(np.asarray(DATA[col_name].tolist()))
        fft_df = pd.DataFrame( { 'fft' : close_fft } )
        fft_df[ 'absolute' ] = fft_df[ 'fft' ].apply(lambda x: np.abs(x))
        fft_df[ 'angle' ] = fft_df[ 'fft' ].apply(lambda x: np.angle(x))
        fft_list = np.asarray(fft_df['fft'].tolist())
        for degree in degrees:
            fft_list_m10= np.copy(fft_list)
            fft_list_m10[ degree : -degree ] = 0
            DATA['fft-' + str(degree)] = np.real(np.fft.ifft( fft_list_m10 ))
        return DATA


    def standard_normalization( self, series, stats = None):
        TS = copy.deepcopy(series)
        if stats is not None:
            mean = stats[0]
            std = stats[1]
            series = list(map( lambda x: ( x - mean ) / std, series ))
            return series
        else:
            mean = np.mean( np.array( series ) )
            std = np.std( np.array( series ) )
            stats = [ mean, std ]
            series = list(map( lambda x: ( x - mean ) / std, series ))
            return series, stats


    def tanh_normalization( self, series, stat=False ):
        TS = copy.deepcopy(series)
        if stats is not None:
            mean = stats[0]
            std = stats[1]
            series = list(map( lambda x: 0.5*( math.tanh( 0.01 * ( x - mean) / std)+1), series ))
            return series
        else:
            mean = np.mean( np.array( series ) )
            std = np.std( np.array( series ) )
            stats = [ mean, std ]
            series = list(map( lambda x: 0.5*( math.tanh( 0.01 * ( x - mean) / std)+1), series ))
            return series, stats


    def Normalize_Data_Part( self, data, method = 'standard', feature_stats_list = None ):
        DATA = copy.deepcopy(data)
        NUM_OF_FEATURES = DATA.shape[1]
    
        DATA_CREATED = list()
    
        if feature_stats_list == None:
            feature_stats_list = list()
            on_train_process = True
        else:
            on_train_process = False

            
        for feature in range(NUM_OF_FEATURES):
            FEATURE_VALUES = DATA.iloc[feature]
            if on_train_process == True:
                NEW_FEATURE_VALUES, stats = self.Normalize_Series( FEATURE_VALUES )
                feature_stats_list.append(stats)
                DATA_CREATED.append(NEW_FEATURE_VALUES)

            else:
                NEW_FEATURE_VALUES = self.Normalize_Series( FEATURE_VALUES , stats = feature_stats_list[feature])
                DATA_CREATED.append(NEW_FEATURE_VALUES)
        if on_train_process == True:
            return np.array(DATA_CREATED), feature_stats_list
        elif on_train_process == False:
            return np.array(DATA_CREATED)

    def Normalize_Series( self, series, method = 'standard', stats= None ):
        TS = copy.deepcopy(series)
        
        if method == 'standard':
            if stats is not None:
                TS = self.standard_normalization(TS, stats )
                return TS
            else:
                TS, stats = self.standard_normalization( TS )
                return TS, stats


    def round_data_size(self, series_len, batch_size, threshold=2):
        how_far = series_len % batch_size
        how_many = max( threshold, np.round(series_len / batch_size) )
        return how_many * batch_size



#============================================================================ DATA =========================================================================================
#============================================================================ DATA =========================================================================================
#============================================================================ DATA =========================================================================================







#============================================================================ MODEL =========================================================================================
#============================================================================ MODEL =========================================================================================
#============================================================================ MODEL =========================================================================================


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
                self.BB[B_or_B + '-' + num] = Block(BRANCH[B_or_B + '-' + num])
            elif B_or_B == 'branch':
                self.BB[B_or_B + '-' + num] = Branch(BRANCH[B_or_B + '-' + num])
                

    def forward(self,x):
        block_key_list = list(self.BB.keys())
        block_key_list = self.splitter(block_key_list,'-')
        
        for i, [TYPE ,NUM] in enumerate(block_key_list):

            if i == 0:
                branch_concat_out = torch.Tensor(self.BB[TYPE + '-' +NUM](x))
            else:
                branch_concat_out = torch.cat([branch_concat_out,self.BB[TYPE + '-' + NUM](x)],dim = 1)

        return branch_concat_out

    def splitter(self,key_list, sep = '-'):
        num_of_keys = len(key_list)
        for key in range(num_of_keys):
            key_list[key] = key_list[key].split(sep)
        return key_list

#============================================================================ MODEL =========================================================================================
#============================================================================ MODEL =========================================================================================
#============================================================================ MODEL =========================================================================================
