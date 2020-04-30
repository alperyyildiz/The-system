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
import os
import pickle

class Data():

    def __init__(self ):  
        pass

    def sliding_window_df( self, series, window_len, out_size, OUTPUT = False):
        
        TS = copy.deepcopy(series)
        num_of_windows = len(TS) - window_len - out_size + 1
        if OUTPUT == False:
            window_list = [ TS[ i : i + window_len ] for i in range(num_of_windows)]
        elif OUTPUT == True:
            window_list = [ TS[ i + window_len: i + window_len + out_size ] for i in range(num_of_windows)]
        return window_list
    



    def Preprocess( self, data, date_col_name = 'Date', out_col_name = 'Close', feature_dim = 1,  fft_degrees = [ 3, 6, 9] ):
        
        
        bs = input('Provide batch size')
        wlen = input('Provide sequence length')
        o_size = input('Provide output length')  
        dm = input('Provide dimensions of the train data') 
        norm = input('Provide normalization technique') 
        
        batch_size = int(bs) 
        window_len = int(wlen)
        out_size = int(o_size)
        dim = int(dm)

        save_name = 'Dim-' + dm + '_' + 'out_size-' + o_size + '_' + 'Wlen-' + wlen + '_' + 'BS-' + bs + '_' + norm + '_' 

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
        date_TRAIN, date_VAL, date_TEST = TRAIN[date_col_name], VAL[date_col_name], TEST[date_col_name]

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
        date_TRAIN, date_VAL, date_TEST = self.sliding_window_df( date_TRAIN, window_len, out_size ), self.sliding_window_df( date_VAL, window_len, out_size ), self.sliding_window_df( date_TEST, window_len, out_size )
        TRAIN, VAL, TEST =  np.array(self.sliding_window_df(TRAIN, window_len, out_size)).swapaxes(1,2),  np.array(self.sliding_window_df(VAL, window_len, out_size)).swapaxes(1,2),  np.array(self.sliding_window_df(TEST, window_len, out_size)).swapaxes(1,2)
        TRAIN_OUT, VAL_OUT, TEST_OUT =  np.array( self.sliding_window_df( TRAIN_OUT, window_len, out_size, OUTPUT = True ) ),  np.array( self.sliding_window_df( VAL_OUT, window_len, out_size, OUTPUT = True) ),  np.array( self.sliding_window_df( TEST_OUT, window_len, out_size, OUTPUT = True ) )
        
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
        feature_size = TRAIN.shape[1]

        save_name = save_name + 'fsize-' + str(feature_size) + '_'

        DATA_LIST = [TRAIN, TRAIN_OUT, VAL, VAL_OUT, TEST, TEST_OUT]
        TRAIN_DL, VAL_DL, TEST_DL = self.make_tensor_and_batch( DATA_LIST, batch_size )
        NEW_DATA_LIST = [ TRAIN_DL, VAL_DL, TEST_DL ]
        self.Save_Processed_Data( NEW_DATA_LIST, save_name )

    def Save_Processed_Data(self, NEW_DATA_LIST, save_name):
        try:
            os.mkdir('DATA')
            os.mkdir('DATA/XU100-29022020')
        except:
            pass

        TRAIN_DL, VAL_DL, TEST_DL = NEW_DATA_LIST
        torch.save( TRAIN_DL, 'DATA/XU100-29022020/' + save_name + 'TRAIN')
        torch.save( VAL_DL,'DATA/XU100-29022020/' + save_name + 'VAL')
        torch.save( TEST_DL,'DATA/XU100-29022020/' + save_name + 'TEST')


        

    def make_tensor_and_batch(self, DATA_LIST, batch_size):
        TRAIN, TRAIN_OUT, VAL, VAL_OUT, TEST, TEST_OUT = DATA_LIST 
        
        #COMBINE INPS WITH OUTS
        #COMBINE INPS WITH OUTS
        TRAIN_DS, VAL_DS, TEST_DS = TensorDataset( TRAIN, TRAIN_OUT ), TensorDataset( VAL, VAL_OUT ), TensorDataset( TEST, TEST_OUT )

        #BATCH ALL DATA
        #BATCH ALL DATA
        TRAIN_DL, VAL_DL, TEST_DL = DataLoader( TRAIN_DS, batch_size ), DataLoader( VAL_DS, batch_size ), DataLoader( TEST_DS, batch_size )
        return TRAIN_DL, VAL_DL, TEST_DL


    def generate_fake_data(self,dims):
        [train_dim, train_out_dim, val_dim, val_out_dim, test_dim, test_out_dim ] = dims
        TRAIN = torch.randn( train_dim )
        TRAIN_OUT = torch.randn( train_out_dim )
        VAL = torch.randn( val_dim )
        VAL_OUT = torch.randn( val_out_dim )
        TEST = torch.randn( test_dim )
        TEST_OUT = torch.randn( test_out_dim )
        TRAIN_DS, VAL_DS, TEST_DS = TensorDataset( TRAIN, TRAIN_OUT ), TensorDataset( VAL, VAL_OUT ), TensorDataset( TEST, TEST_OUT )
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


