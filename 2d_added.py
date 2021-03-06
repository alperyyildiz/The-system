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
        self.COLLECT_OUT_CHANNELS_FROM = ['conv1d', 'conv2d', 'conv1dTranspose', 'conv2dTranspose', 'Linear'] #out channels reworked, first param will be skipped
        self.PUSH_OUT_CHANNELS_TO = ['conv1d', 'conv2d', 'conv1dTranspose', 'conv2dTranspose', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'Linear' ] 

        self.Only_Parameter_Layers = ['Dropout', 'Dropout2d', 'Dropout3d', 'AlphaDropout' ,
                                      'ReflectionPad1d', 'ReflectionPad2d', 
                                      'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d', 
                                      'Hardshrink', 'Softshrink' ]                                      
        self.Direct_Pass_Layers = ['ReLU', 'ReLU6', 'ELU', 'SELU', 'Flatten' ]
        self.Layers_1d = ['conv1d', 'conv1dTranspose', 'BatchNorm1d','Linear']
        self.Layers_2d = ['conv2d', 'conv2dTranspose','BatchNorm2d' ]

        self.Activation_Layers = ['relu', 'sigmoid', 'tanh', ]
        self.Def_Params = {'conv1d': {'in_channels': None,
                                      'out_channels':None,
                                      'kernel_size':None,
                                      'stride':1,
                                      'padding':0,
                                      'dilation':1,
                                      'groups':1, 
                                      'bias':True,
                                      'padding_mode': 'zeros'},
                           'conv2d': {'in_channels':None,
                                      'out_channels':None,
                                      'kernel_size':None,
                                      'stride':1,
                                      'padding':0, 
                                      'dilation':1, 
                                      'groups':1, 
                                      'bias':True, 
                                      'padding_mode':'zeros'},
                           'conv3d': {'in_channels': None,
                                      'out_channels':None,
                                      'kernel_size':None, 
                                      'stride':1,
                                      'padding':0,
                                      'dilation':1,
                                      'groups':1,
                                      'bias':True,
                                      'padding_mode':'zeros'},
                           
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
                           'conv3dTranspose': {'in_channels':None,
                                               'out_channels':None,
                                               'kernel_size':None,
                                               'stride':1,
                                               'padding':0,
                                               'output_padding':0, 
                                               'groups':1,
                                               'bias':True,
                                               'dilation':1,
                                               'padding_mode':'zeros'},
                           'GRU': {'input_size': None,
                                    'hidden_size': None,
                                    'num_layers': 1,
                                    'bias': True,
                                    'batch_first': True,
                                    'dropout': 0,
                                    'bidirectional': False,
                                    'train_batch_size': None},
                           'LSTM': {'input_size': None,
                                    'hidden_size': None,
                                    'num_layers': 1,
                                    'bias': True,
                                    'batch_first': True,
                                    'dropout': 0,
                                    'bidirectional': False,
                                    'train_batch_size': None},
                           
                           'Linear': {'in_features':None,
                                      'out_features':None,
                                      'bias':True},
                           'Bilinear':{'in1_features':None, 'in2_features':None, 'out_features':None, 'bias':True},

                           'Embedding': {'num_embeddings':None, 'embedding_dim':None, 'padding_idx':None, 'max_norm':None, 'norm_type':2.0, 'scale_grad_by_freq':False, 'sparse':False, '_weight':None},
                           'EmbeddingBag': {'num_embeddings':None, 'embedding_dim':None, 'padding_idx':None, 'max_norm':None, 'norm_type':2.0, 'scale_grad_by_freq':False, 'sparse':False, '_weight':None},
                           #===================   POOL POOL POOL ===================#
                           'AvgPool1d': {'kernel_size':None, 'stride':1,'padding':0,'ceil_mode':False,'count_include_pad':True},
                           'AvgPool2d': {'kernel_size':None, 'stride':1,'padding':0,'ceil_mode':False,'count_include_pad':True,'divisor_override':None},
                           'AvgPool3d': {'kernel_size':None, 'stride':1,'padding':0,'ceil_mode':False,'count_include_pad':True,'divisor_override':None},
                           'MaxPool1d': {'kernel_size':None, 'stride':1,'padding':0,'dilation':1,'return_indices':False,'ceil_mode':False},
                           'MaxPool2d': {'kernel_size':None, 'stride':1,'padding':0,'dilation':1, 'return_indices':False,'ceil_mode':False},
                           'MaxPool3d': {'kernel_size':None, 'stride':1,'padding':0,'dilation':1, 'return_indices':False,'ceil_mode':False},
                           'FractionalMaxPool2d': {'kernel_size':None,'output_size':None,'output_ratio':None,'return_indices':False,'_random_samples':None},
                           'LPPool1d':{'norm_type':None, 'kernel_size':None,'stride':None,'ceil_mode':False},
                           'LPPool2d':{'norm_type':None,'kernel_size':None,'stride':None,'ceil_mode':False},
                           'AdaptiveMaxPool1d': {'output_size':None,'return_indices':False},
                           'AdaptiveMaxPool2d': {'output_size':None,'return_indices':False},
                           'AdaptiveMaxPool3d': {'output_size':None,'return_indices':False},
                           'AdaptiveAvgPool1d':{'output_size':None},
                           'AdaptiveAvgPool2d':{'output_size':None},
                           'AdaptiveAvgPool3d':{'output_size':None},
                           #===================   POOL POOL POOL ===================#


                           #===================   PAD PAD PAD ===================#
                           'ReflectionPad1d':{'padding':None},
                           'ReflectionPad2d':{'padding':None},
                           'ReplicationPad1d':{'padding':None},
                           'ReplicationPad2d':{'padding':None},
                           'ReplicationPad3d':{'padding':None},
                           'ZeroPad2d':{'padding':None},
                           'ConstantPad1d':{'padding':None,'value':None},
                           'ConstantPad2d':{'padding':None,'value':None},
                           'ConstantPad3d':{'padding':None,'value':None},
                           #===================   PAD PAD PAD ===================#


                           #===================   UNPOOL UNPOOL ===================#
                           'MaxUnpool1d':{'kernel_size': None,'stride': None,'padding': 0},
                           'MaxUnpool2d':{'kernel_size': None,'stride': None,'padding': 0},
                           'MaxUnpool3d':{'kernel_size': None,'stride': None,'padding': 0},
                           #===================   UNPOOL UNPOOL ===================#



                           #=================== BATCHNORM ===================#
                           'BatchNorm1d': {'num_features': None  ,'eps':1e-05, 'momentum':0.1, 'affine':True,'track_running_stats':True},
                           'BatchNorm2d': {'num_features':None,'eps':1e-05,'momentum':0.1,'affine':True,'track_running_stats':True},
                           'BatchNorm3d': {'num_features':None,'eps':1e-05,'momentum':0.1,'affine':True,'track_running_stats':True},
                           'SyncBatchNorm': {'num_features':None,'eps':1e-05,'momentum':0.1,'affine':True,'track_running_stats':True},
                           'InstanceNorm1d': {'num_features':None,'eps':1e-05,'momentum':0.1,'affine':True,'track_running_stats':True},
                           'InstanceNorm2d': {'num_features':None,'eps':1e-05,'momentum':0.1,'affine':True,'track_running_stats':True},
                           'InstanceNorm3d': {'num_features':None,'eps':1e-05,'momentum':0.1,'affine':True,'track_running_stats':True},
                           'LayerNorm':{'normalized_shape':None, 'eps':1e-05,'elementwise_affline':True},
                           'LocalResponseNorm': {'size':None, 'alpha':0.0001,'beta':0.75,'k':1.0},
                           #===================  BATCHNORM ===================#

                           #===================  ACTIVATIONS ===================#
                           'ELU':{'alpha':1.0, 'inplace':False},
                           'Hardshrink': {'lambd':0.5},
                           'Hardtanh':{'min_val':-1.0, 'max_val':1.0, 'inplace':False, 'min_value':None, 'max_value':None},
                           'LeakyReLU':{'negative_slope': 0.01, 'inplace' :False},
                           'MultiheadAttention': {'embed_dim':None, 'num_heads':None,'dropout':0.0,'bias':True,'add_bias_kv':False,'add_zero_attn':False, 'kdim':None, 'vdim':None},
                           'PReLU': {'num_parameters': 1, 'init': 0.25},
                           'ReLU':{'inplace':False},
                           'ReLU6':{'inplace':False},
                           'RReLU':{'lower':0.125, 'upper':0.3333333333333333, 'inplace':False},
                           'SELU':{'inplace':False},
                           'CELU':{'alpha': 1.0, 'inplace':False},
                           'GELU':{},
                           'Sigmoid': {},
                           'Softplus':{'beta':1,'threshold':20},
                           'Softshrink':{'lambd':0.5},
                           'Softsign':{},
                           'Tanh': {},
                           'Tanshrink':{},
                           'Threshold':{'threshold':None,'value':None, 'inplace':False},
                           'Softmin':{'dim':None},
                           'Softmax':{'dim':None},
                           'Softmax':{},
                           'LogSoftmax':{'dim':None},
                           'AdaptiveLogSoftmaxWithLoss': {'in_features':None, 'n_classes':None, 'cutoffs':None, 'div_value':4.0, 'head_bias':False},
                           #===================  ACTIVATIONS ===================#
                           

                           #===================  TRANSFORMERS ===================#
                           'Transformer':{'d_model':512, 'nhead':8, 'num_encoder_layers':6, 'num_decoder_layers':6, 'dim_feedforward':2048,'dropout':0.1, 'activation':'relu', 'custom_encoder':None, 'custom_decoder':None},
                           'TransformerEncoder':{'encoder_layer':None, 'num_layers':None, 'norm':None},
                           'TransformerDecoder':{'decoder_layer':None, 'num_layers':None, 'norm':None},
                           'TransformerEncoderLayer':{'d_model':None, 'nhead':None, 'dim_feedforward':2048, 'dropout':0.1, 'activation':'relu'},
                           'TransformerDecoderLayer': {'d_model':None, 'nhead':None, 'dim_feedforward':2048, 'dropout':0.1, 'activation':'relu'},
                           #===================  TRANSFORMERS ===================#



                           #===================  DIST FUNC ===================#
                           'CosineSimilarity': {'dim':1,'eps':1e-08},
                           'PairwiseDistance': {'p':2.0, 'eps':1e-06, 'keepdim':False},
                           #===================  DIST FUNC ===================#


                           #===================  LOSS FUNC ===================#
                           'L1Loss': {'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'MSELoss': {'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'CrossEntropyLoss':{'weight':None, 'size_average':None, 'ignore_index':-100, 'reduce':None, 'reduction':'mean'},
                           'CTCLoss': {'blank':0, 'reduction':'mean', 'zero_infinity':False},
                           'NLLLoss':{'weight':None, 'size_average':None, 'ignore_index':-100, 'reduce':None, 'reduction':'mean'},
                           'PoissonNLLLoss':{'log_input':True, 'full':False, 'size_average':None, 'eps':1e-08, 'reduce':None, 'reduction':'mean'},
                           'KLDivLoss': {'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'BCELoss': {'weight':None,'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'BCEWithLogitsLoss': {'weight':None,'size_average':None, 'reduce':None, 'reduction':'mean','pos_weight':None},
                           'MarginRankingLoss': {'margin':0.0, 'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'HingeEmbeddingLoss':{'margin':0.0, 'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'MultiLabelMarginLoss':{'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'SmoothL1Loss': {'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'SoftMarginLoss':{'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'MultiLabelSoftMarginLoss':{'weight':None,'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'CosineEmbeddingLoss': {'margin':0.0, 'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'MultiMarginLoss': {'p':1, 'margin':1.0, 'weight':None, 'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'TripletMarginLoss': {'margin':1.0, 'p':2.0, 'eps':1e-06, 'swap':False, 'size_average':None, 'reduce':None, 'reduction':'mean'},
                           'Upsample': {'size':None, 'scale_factor':None, 'mode':'nearest', 'align_corners':None},
                           'UpsamplingNearest2d': {'size':None,'scale_factor':None},
                           'UpsamplingBilinear2d':{'size':None, 'scale_factor':None},
                           #===================  LOSS FUNC ===================#




                           'Unfold': {'kernel_size':None, 'dilation':1,'padding':0,'stride':1},
                           'Unfold': {'output_size':None,'kernel_size':None, 'dilation':1,'padding':0,'stride':1},

                           'Flatten': {'start_dim':1, 'end_dim':-1},
                           'Dropout': {'p':0.5, 'inplace': False},
                           'Dropout2d': {'p':0.5, 'inplace': False},
                           'Dropout3d': {'p':0.5, 'inplace': False},
                           'AlphaDropout': {'p':0.5, 'inplace': False}}


#============================================================================INPUT REWORKS=========================================================================================

    def input_reworker(self,typez, input_params):
        #Split inputs given by user
        #Set defaults if remain
        #Change dtype from str to corresponding types
        #Built in functions used which are initialized as self.input_types
        type_def_params = list(self.Def_Params[typez].values())

        for i in range(len(input_params)):
            if input_params[i] == '-':
                if typez == 'MaxPool1d' and i == 1:
                    input_params[i] = input_params[i - 1]
                else:
                    input_params[i] = type_def_params[i]

        data_type_functions = self.input_types[typez]
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

    def track_dim(self,layer,dim, num_of_dim):
        #Method to track sequence length of forward input
        #Used to calculate input channels for the layer after flatten
        #Also will be used for debugging in the future
        params = self.input_reworker(layer[0], layer[1])        
        if num_of_dim == 1:

            if layer[0] == 'conv1d':
                dim = int(np.ceil((dim + 2*params[4] - params[5]*(params[2]-1)-1)/params[3] + 1))
            elif layer[0] == 'MaxPool1d':
                dim = int(np.ceil((dim + 2*params[2] - params[3]*(params[0]-1)-1)/params[1] + 1))
            elif layer[0] == 'Flatten':
                dim =  1
        
        elif num_of_dim == 2:
            if layer[0] == 'conv2d':
                dim[0] = int(np.ceil((dim[0] + 2*params[4][0] - params[5][0]*(params[2][0]-1)-1)/params[3][0] + 1))
                dim[1] = int(np.ceil((dim[1] + 2*params[4][1] - params[5][1]*(params[2][1]-1)-1)/params[3][1] + 1))
            elif layer[0] == 'MaxPool2d':
                dim[0] = int(np.ceil((dim[0] + 2*params[2][0] - params[3][0]*(params[0][0]-1)-1)/params[3][0] + 1))
                dim[1] = int(np.ceil((dim[1] + 2*params[2][1] - params[3][1]*(params[0][1]-1)-1)/params[3][1] + 1))
            elif layer[0] == 'Flatten':
                dim[0], dim[1] = 1 , 1
        
        return dim

                


    def track_out_channels(self,layer,out_channels, dim, num_of_dim):
        PARAMS = layer[1]
        TYPE = layer[0]
        if TYPE in self.COLLECT_OUT_CHANNELS_FROM:
            out_channels = PARAMS[1]
        elif TYPE == 'Flatten':
            if num_of_dim == 1:
                out_channels = dim * out_channels
            elif num_of_dim == 2:
                out_channels = dim[0] * dim[1] * out_channels
        return out_channels

#============================================================================TRACK SEQ AND CH=========================================================================================

    def get_lay_params(self):
        TYPE = input('enter layer type: ')

        key_value_list = list(self.Def_Params[TYPE].items() )
        values_list = [x for y,x in key_value_list]
        
        if TYPE in self.Direct_Pass_Layers:
            input_params = values_list
        elif TYPE in self.Only_Parameter_Layers:
            print(key_value_list)
            input_params = input()
        
        elif TYPE in self.PUSH_OUT_CHANNELS_TO:
            print(key_value_list[1:])
            input_params = input()
            input_params = input_params.split(',')
            input_params.insert(0,None)
            if  input_params[-1] == '...':
                input_params = input_params[:-1]
                input_params.extend( values_list[ len(input_params) :  ]) 


            if len( input_params ) != len( values_list ):
                raise Exception('lengths are not consistent')

        return TYPE, input_params

'''
        else:
            print('\n Seperate input with comma, write - for defalt value, ... for default rest \n')
            print(self.Def_Params[TYPE])
            print('\n')
            input_params = input()
            
            if  input_params[-1] == '...':
                input_params = input_params[:-1]
                input_params.extend(values_list[ len(input_params) :  ]) 
                if len( input_params ) != len( values_list ):
                    raise Exception('lengths are not consistent')
            input_params = self.input_reworker(TYPE, input_params)
'''
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


#============================================================================ NET =========================================================================================
#============================================================================ NET =========================================================================================
#============================================================================ NET =========================================================================================


class NET(MAIN_OBJ):
    def __init__(self, INP_OBJ):
        super().__init__()


        self.dim = INP_OBJ.window_len
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
        self.out_channels, self.dim = self.track_dim_and_ch_block(BLOCK,self.out_channels,self.dim)

        #Append block layers to network
        self.num_of_blocks = self.num_of_blocks + 1 
        self.ALL_BLOCKS[str(self.num_of_blocks )] = BLOCK
        self.order.append('block')
        self.First = False

    def check_dim_branch(self, Branch):
        BRANCH = copy.deepcopy(Branch)

        entity_keys = list(BRANCH.keys())
        entity_types = self.splitter(entity_keys)
        dim_list = list()
        for i, ENTITY in enumerate(entity_keys):
            if entity_types[ i ][0] == 'block':
                dim = self.check_dim( BRANCH[ ENTITY ] )
            elif entity_types[ i ][0] == 'branch':
                dim = self.check_dim_branch(BRANCH[ ENTITY ])

            dim_list.append(dim)

        ALL_SAME = all(x == dim_list[0] for x in dim_list)
        
        if ALL_SAME == False:
            raise Exception('Dimension Error will occur')
        elif ALL_SAME == True:
            return dim_list[0]



    def check_dim(self,Block):
        BLOCK = copy.deepcopy(Block)
        entity_keys = list(BLOCK.keys())
        entity_types = self.splitter(entity_keys)

        for i, ENTITY in enumerate(entity_keys):
            if entity_types[ i ][0] == 'layer':
                if BLOCK[ ENTITY ][0] in self.Layers_1d:
                    num_of_dim = 1
                    return num_of_dim
                elif BLOCK[ ENTITY ][0] in self.Layers_2d:
                    num_of_dim = 2

                    return num_of_dim 

            elif entity_types[ i ][0] == 'block':
                return self.check_dim(BLOCK[ ENTITY ])

            elif entity_types[ i ][0] == 'branch':
                return self.check_dim_branch(BLOCK[ ENTITY ])

    def push_ch_to_block(self,BLOCK,out_channels):

        BLOCK = copy.deepcopy(BLOCK)
        entity_keys = list(BLOCK.keys())
        entity_types = self.splitter(entity_keys)
        if entity_types[0][0] == 'layer':
            BLOCK['layer-1'] = self.push_ch_to_layer(BLOCK['layer-1'], out_channels)
        elif entity_types[0][0] == 'block':
            BLOCK['block-1'] = self.push_ch_to_block( BLOCK[ 'block-1' ], out_channels )
        elif entity_types[0][0] == 'branch':
            BLOCK['branch-1'] = self.push_ch_to_branch( BLOCK[ 'branch-1' ], out_channels )

        return BLOCK


    def push_ch_to_branch(self,BRANCH,out_channels):
        BRANCH = copy.deepcopy(BRANCH)
        entity_keys = list(BRANCH.keys())
        entity_types = self.splitter(entity_keys)
        
        for i, ENTITY_KEY in enumerate(entity_keys): 
            if entity_types[i][0]  == 'block':
                BRANCH[ ENTITY_KEY ] = self.push_ch_to_block( BRANCH[ ENTITY_KEY ], out_channels )
            elif entity_types[i][0] == 'branch':
                BRANCH[ ENTITY_KEY ] = self.push_ch_to_branch( BRANCH[ ENTITY_KEY ], out_channels )
        return BRANCH


    def push_ch_to_layer(self,Layer, out_channels):
        LAYER = copy.deepcopy(Layer)
        TYPE = LAYER[0]
      
        if TYPE in self.PUSH_OUT_CHANNELS_TO:
            LAYER[1][0] = out_channels


        return LAYER

    def check_flatten_layer(self,Layer, Flat):
        LAYER = copy.deepcopy(Layer)
        if LAYER[0] == 'Flatten':
            Flat = True
        return Flat

    def check_flatten_block(self,block):
        BLOCK = copy.deepcopy(block)
        Flatten = False
        entity_keys = list(BLOCK.keys())
        entity_types = self.splitter(entity_keys)
        for i, ENTITY in enumerate(entity_keys):
            if entity_types[i][0] == 'layer':
                Flat = self.check_flatten_layer( BLOCK[ ENTITY ], Flatten ) 
                if Flat == True:
                    break
            elif entity_types[i][0] == 'block':
                Flat = self.check_flatten_block( BLOCK[ ENTITY ] ) 
                if Flat == True:
                    break
            elif entity_types[i][0] == 'branch':
                Flat = self.check_flatten_branch( BLOCK[ ENTITY ] ) 
                if Flat == True:
                    break
        return Flat

        
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
            
  
    def track_dim_and_ch(self,Layer,out_channels,dim, num_of_dim ):
        LAYER = copy.deepcopy( Layer )
        out_channels = self.track_out_channels( LAYER, out_channels, dim, num_of_dim)
        dim = self.track_dim( LAYER, dim, num_of_dim)
        return out_channels, dim




    def check_dim_branch(self, Branch):
        BRANCH = copy.deepcopy(Branch)

        entity_keys = list(BRANCH.keys())
        entity_types = self.splitter(entity_keys)
        dim_list = list()
        for i, ENTITY in enumerate(entity_keys):
            if entity_types[ i ][0] == 'block':
                dim = self.check_dim( BRANCH[ ENTITY ] )
            elif entity_types[ i ][0] == 'branch':
                dim = self.check_dim_branch(BRANCH[ ENTITY ])

            dim_list.append(dim)

        ALL_SAME = all(x == dim_list[0] for x in dim_list)
        
        if ALL_SAME == False:
            raise Exception('Dimension Error will occur')
        elif ALL_SAME == True:
            return dim_list[0]



    def check_dim(self,Block):
        BLOCK = copy.deepcopy(Block)
        entity_keys = list(BLOCK.keys())
        entity_types = self.splitter(entity_keys)

        for i, ENTITY in enumerate(entity_keys):
            if entity_types[ i ][0] == 'layer':
                if BLOCK[ ENTITY ][0] in self.Layers_1d:
                    num_of_dim = 1
                    return num_of_dim
                elif BLOCK[ ENTITY ][0] in self.Layers_2d:
                    num_of_dim = 2

                    return num_of_dim 

            elif entity_types[ i ][0] == 'block':
                return self.check_dim(BLOCK[ ENTITY ])

            elif entity_types[ i ][0] == 'branch':
                return self.check_dim_branch(BLOCK[ ENTITY ])


    def track_dim_and_ch_block(self, BLOCK, out_channels, dim):
        BLOCK = copy.deepcopy(BLOCK)
        num_of_dim = self.check_dim(BLOCK)
        block_keys = list(BLOCK.keys())
        entity_type = [x for x,y in self.splitter(block_keys)]

        for i, entity in enumerate(block_keys):
            
            if entity_type[i] == 'layer':
                out_channels, dim = self.track_dim_and_ch(BLOCK[ entity ], out_channels, dim, num_of_dim)
            elif entity_type[i] == 'block':
                out_channels, dim = self.track_dim_and_ch_block(BLOCK[ entity ], out_channels, dim)
            elif entity_type[i] == 'branch':
                dim_list = [dim for x in range(len(list( BLOCK[ entity ] ))) ]
                out_channels, dim = self.track_dim_and_ch_branch(BLOCK[ entity ], out_channels, dim_list)
        return out_channels, dim


    def track_dim_and_ch_branch(self, BRANCH, out_channels, dim_list):
        BRANCH = copy.deepcopy(BRANCH)

        #List TYPE-NUM in branch items as BB_keys
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)
        final_dim_list = list()
        out_channels_list = list()
        for i, [TYPE,NUM] in enumerate(BB_keys):
            BB = BRANCH[TYPE + '-' + NUM]
            dim = dim_list[i]

            if TYPE == 'block':
                out_channels, dim = self.track_dim_and_ch_block(BB, out_channels, dim)
                final_dim_list.append(dim)
                out_channels_list.append(out_channels)

            elif TYPE == 'branch':
                temp_branch_sll = [dim for _ in range(len(BB.keys()))]
                out_channels, dim = self.track_dim_and_ch_branch(BB,out_channels,temp_branch_sll)
                final_dim_list.append(dim)
                out_channels_list.append(out_channels)
        if len(np.array(final_dim_list).shape) > 3:
            Dim1_Same = all(x[1] ==final_dim_list[0][1] for x in final_dim_list)
            Dim2_Same = all(x[2] ==final_dim_list[0][2] for x in final_dim_list)
            if Dim1_Same == True and Dim2_Same == True:
                All_Same = True
            else:
                All_Same = False
        else:
            All_Same = all(x == final_dim_list[0] for x in final_dim_list)
        
        Flat = self.check_flatten_branch(BRANCH)

        if Flat == False:
            if All_Same == False:
                raise Exception('You Will Have Dimension Problems \n Sequences are not equal \n Flatten is Not Used')
            elif All_Same == True:
                dim = final_dim_list[0]

            for t, item in enumerate(out_channels_list):
                out_channels = out_channels + item

        elif Flat == True:
            dim = 1
            out_channels = 0
            for t, item in enumerate(out_channels_list):
                out_channels = out_channels + item

        return out_channels, dim

    def splitter(self,key_list, sep = '-'):
        key_list = copy.deepcopy(key_list)
        new_key_list = list()
        num_of_keys = len(key_list)
        for key in range(num_of_keys):
            new_key_list.append(key_list[key].split(sep))
        return new_key_list



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
        #create init_dim_list 
        if self.First is False:
            dim = self.dim
            init_dim_list = [dim for x in range(num_of_bb)] 
        else:
            self.out_channels = self.feature_size
            self.Branch_First = True
            seq_list = input('Provide input sequences, separate them with comma(,)')
            init_dim_list = seq_list.split(",")

        #Check if all seq_lengths are the same
        #Check if all blocks have flatten layer


        #Set In_Channels
        BRANCH = self.push_ch_to_branch(BRANCH,self.out_channels)

        self.out_channels, self.dim = self.track_dim_and_ch_branch(BRANCH, self.out_channels, init_dim_list)



        self.num_of_branches = self.num_of_branches + 1 
        self.ALL_BRANCHES[str(self.num_of_branches)] = BRANCH

        self.order.append('branch') 
        self.First = False


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
                print(list(self.Blocks.keys()))

                block_name  = input('Which block you want to add as a branch \n')
                Dict_Branch['block-' + str(count_block)] = self.Blocks[block_name]

            elif branch_or_block == '2':
                print(list(self.Branches_Created.keys()))

                branch_name  = input('Which block you want to add as a branch \n')
                Dict_Branch['branch-' + str(count_block)] = self.Branches_Created[branch_name]

            add_value = input('You want to add more branches?? \n')
            Add_branches = self.Bool_Rework(add_value)


        self.Branches_Created[NAME] = Dict_Branch

    def Create_Block(self):
        #Creates an ordered dictionary
        #Saves layer type with parameters specified by user
        NAME = input('Enter Block Name: ')
        count = 0
        Add = True


        MAIN_DICT = OrderedDict()
        out_channels = None
        
        #While loop until user says stop
        while Add:
            count = count + 1
            TYPE = input('enter entity type \n 1 --> Layer \n 2 --> Block \n 3 --> Branch ')
            
            if TYPE == '1':
                entity_key = 'layer-' + str(count)
                LAYER_TYPE , PARAM = self.get_lay_params()
                if LAYER_TYPE == 'Flatten':
                    Add = False
                else:
                    add_value = input('You want to add more?')
                    Add = self.Bool_Rework(add_value)

                MAIN_DICT[ entity_key ] =  [LAYER_TYPE, self.input_reworker( LAYER_TYPE, PARAM )]
            
            elif TYPE == '2':
                print(list(self.Blocks.keys()))
                entity_key = 'block-' + str(count)
                input_block = input()
                
                INP_BLOCK =  self.Blocks[ input_block ]

                MAIN_DICT[ entity_key ] = INP_BLOCK

                Flat = self.check_flatten_block( INP_BLOCK )
                if Flat == False:
                    add_value = input('You want to add more?? \n')
                    Add = self.Bool_Rework(add_value)

                else:
                    Add = False
            
            elif TYPE == '3':
                print(list(self.Branches_Created.keys()))
                entity_key = 'branch-' + str(count)
                input_branch = input()
                MAIN_DICT[ entity_key ] = self.Branches_Created[ input_branch ]

                Flat = self.check_flatten_branch(  MAIN_DICT[ entity_key ]  )
                if Flat == False:
                    add_value = input('You want to add more?? \n')
                    Add = self.Bool_Rework(add_value)
                else:
                    Add = False
      
        count = count + 1
        entity_list_str = list(MAIN_DICT.keys())
        entity_type_str = [x for [x,y] in self.splitter(entity_list_str)]
        for i, ENTITY in enumerate(entity_list_str):

            #IF ENTITY IS BLOCK
            if entity_type_str[i] == 'layer':
                MAIN_DICT[ ENTITY ] = self.push_ch_to_layer( MAIN_DICT[ ENTITY ]  , out_channels)
                out_channels = self.track_out_channels( MAIN_DICT[ ENTITY ] , out_channels, 10000, 1)

            if entity_type_str[i] == 'block':

                MAIN_DICT[ ENTITY ] = self.push_ch_to_block(MAIN_DICT [ ENTITY ] , out_channels )


                #IF BLOCK IS NOT FLATTENED
                #Flat = self.check_flatten_block(MAIN_DICT [ ENTITY ])

                #IF THIS IS NOT LAST PART
                if ENTITY != entity_list_str[-1] :

                    #IF DIM IS 1
                    if self.check_dim( MAIN_DICT[ ENTITY ] ) == 1:
                        out_channels, _ = self.track_dim_and_ch_block( MAIN_DICT [ ENTITY ], out_channels, dim=10000 )
                    #IF DIM IS 2
                    elif self.check_dim( MAIN_DICT[ ENTITY ] ) == 2:
                        out_channels, _ = self.track_dim_and_ch_block( MAIN_DICT [ ENTITY ], out_channels, dim=(10000,10000) )

            elif entity_type_str[i] == 'branch':
                MAIN_DICT[ ENTITY ] = self.push_ch_to_branch( MAIN_DICT [ ENTITY ] , out_channels)

                dim = (10000)
                dim_2d = (10000,10000)
                dim_list_1d = [dim for x in range(len(list(MAIN_DICT [ ENTITY ].keys())))]
                dim_list_2d = [dim_2d for x in range(len(list(MAIN_DICT [ ENTITY ].keys())))]
                Flat = self.check_flatten_branch(MAIN_DICT [ ENTITY ])
                #IF BLOCK IS NOT FLATTENED
                #IF THIS IS NOT LAST PART
                if ENTITY != entity_list_str[-1] :
                    #IF DIM IS 1
                    if self.check_dim( MAIN_DICT[ ENTITY ] ) == 1:
                        out_channels, _ = self.track_dim_and_ch_branch( MAIN_DICT [ ENTITY ], out_channels, dim_list_1d )
                    #IF DIM IS 2
                    elif self.check_dim( MAIN_DICT[ ENTITY ] ) == 2:
                        out_channels, _ = self.track_dim_and_ch_branch( MAIN_DICT [ ENTITY ], out_channels, dim_list_2d )
   
        self.Blocks[NAME] = MAIN_DICT


#============================================================================ NET =========================================================================================
#============================================================================ NET =========================================================================================
#============================================================================ NET =========================================================================================




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

#============================================================================ MODEL =========================================================================================
#============================================================================ MODEL =========================================================================================
#============================================================================ MODEL =========================================================================================






