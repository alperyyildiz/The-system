from __future__ import division, print_function
import numpy as np
import pylab as plt
from scipy.spatial.distance import pdist, squareform
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
from os import listdir
from os.path import isfile, join
import math
import numpy as np
import pylab as plt

def rec_plot(s, eps=0.10, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return Z

class Data():

    def __init__(self ):  
        pass

    def sliding_window_df( self, series, max_window_len, window_len, out_size, OUTPUT = False):
        
        TS = copy.deepcopy(series)
        num_of_windows = len(TS) - max_window_len - out_size + 1
        base_start = max_window_len - window_len

        if OUTPUT == False:
            window_list = [ TS[ i + base_start : i + base_start + window_len ] for i in range(num_of_windows)]
        elif OUTPUT == True:
            window_list = [ TS[ i + base_start + window_len: i + base_start + window_len + out_size ] for i in range(num_of_windows)]
            print(np.array(window_list).shape)
        return window_list
    



    def Convert_to_Tensor(self, DATA_LIST, save_name):
        DATA_LIST = copy.deepcopy( DATA_LIST )
        TRAIN, TRAIN_OUT, VAL, VAL_OUT, TEST, TEST_OUT = DATA_LIST
        #CONVERT DATA TO TENSOR
        #CONVERT DATA TO TENSOR
        TRAIN, VAL, TEST = torch.Tensor( TRAIN ), torch.Tensor( VAL ), torch.Tensor( TEST )
        TRAIN_OUT, VAL_OUT, TEST_OUT = torch.Tensor( TRAIN_OUT  ), torch.Tensor(VAL_OUT ), torch.Tensor( TEST_OUT )

        return [ TRAIN, VAL, TEST, TRAIN_OUT, VAL_OUT, TEST_OUT ]
    




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
            FEATURE_VALUES = DATA.iloc[ : , feature]

            if on_train_process == True:
                NEW_FEATURE_VALUES, stats = self.Normalize_Series( FEATURE_VALUES )
                feature_stats_list.append(stats)
                DATA_CREATED.append(NEW_FEATURE_VALUES)

            else:
                NEW_FEATURE_VALUES = self.Normalize_Series( FEATURE_VALUES , stats = feature_stats_list[feature])
                DATA_CREATED.append(NEW_FEATURE_VALUES)

        if on_train_process == True:
            return np.array(DATA_CREATED).swapaxes( 0, 1 ), feature_stats_list
        elif on_train_process == False:
            return np.array(DATA_CREATED).swapaxes( 0, 1 )

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

    def join(self, name_parts, seperator):
        last = ''
        for i, part in enumerate(name_parts):
            if i == 0:
                last = last + part
            else:
                last = last + seperator + part
        return last




    def transform( self, serie ):
        """Compute the Gramian Angular Field of an image"""
        # Min-Max scaling
        min_ = np.amin(serie)
        max_ = np.amax(serie)
        scaled_serie = (2*serie - max_ - min_)/(max_ - min_)

        # Floating point inaccuracy!
        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

        # Polar encoding
        phi = np.arccos(scaled_serie)

        # GAF Computation (every term of the matrix)
        gaf = tabulate(phi, phi, cos_sum)
        return gaf, phi, scaled_serie

        
    def Preprocess( self, data, date_col_name = 'Date', out_col_name = 'Close', feature_dim = 1,  fft_degrees = [ 3, 6, 9] ):
        
        dt = input('Provide initial input dims \n Seperate with - if multiple') 

        dtype_new = dt.split('-')

        for elem in dtype_new:
            elem = int(elem)

        if len(dtype_new) > 1: 
            dtype = 'Multiple'
        else:
            dtype = 'Single'
        

        bs = input('\n Provide batch size \n ')
        wlen = input('\n Provide sequence length \n seperate with - if multiple \n ')
        o_size = input('\n Provide output length \n ')  
        norm = input('Provide normalization technique \n') 
        if dtype == 'Single':
            window_len = int(wlen)
            max_window_len = window_len

        elif dtype == 'Multiple': 
            window_len = wlen.split('-')
            window_len = list(map( int, window_len ))
            max_window_len = max(window_len)
        
        batch_size = int(bs) 
        out_size = int(o_size)
            
        save_name = 'type-' + dt + '_' + 'out_size-' + o_size + '_' + 'Wlen-' + wlen + '_' + 'BS-' + bs + '_' + norm 

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
        TRAIN, VAL, TEST = self.train_val_test_split( DATA, percentages, output_size = out_size , window_len = max_window_len )

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
        TRAIN_OUT, VAL_OUT, TEST_OUT = TRAIN[ :, out_col ], VAL[ :, out_col ], TEST[ :, out_col ]

        #DROP OUTS
        #DROP OUTS 
        TRAIN, VAL, TEST = np.delete(TRAIN, out_col, axis = 1 ), np.delete(VAL, out_col, axis = 1 ), np.delete(TEST, out_col, axis = 1 )



        if dtype == 'Single':
            #SLIDING WINDOW for wlen = window_len
            #SLIDING WINDOW for wlen = window_len
            date_TRAIN, date_VAL, date_TEST = self.sliding_window_df( date_TRAIN, max_window_len, window_len, out_size ), self.sliding_window_df( date_VAL, max_window_len, window_len, out_size ), self.sliding_window_df( date_TEST, max_window_len, window_len, out_size )
            TRAIN, VAL, TEST =  np.array(self.sliding_window_df(TRAIN, max_window_len, window_len, out_size)).swapaxes(1,2),  np.array(self.sliding_window_df(VAL, max_window_len, window_len, out_size)).swapaxes(1,2),  np.array(self.sliding_window_df(TEST, max_window_len, window_len, out_size)).swapaxes(1,2)
            TRAIN_OUT, VAL_OUT, TEST_OUT =  np.array( self.sliding_window_df( TRAIN_OUT, max_window_len, window_len, out_size, OUTPUT = True ) ),  np.array( self.sliding_window_df( VAL_OUT, max_window_len, window_len, out_size, OUTPUT = True) ),  np.array( self.sliding_window_df( TEST_OUT, max_window_len, window_len, out_size, OUTPUT = True ) )
            

            DATA_LIST = [TRAIN, TRAIN_OUT, VAL, VAL_OUT, TEST, TEST_OUT]
            FS = DATA_LIST[0].shape[1]

            DATA_LIST = self.Convert_to_Tensor( DATA_LIST, save_name )
            DL_LIST = self.make_batch_single( DATA_LIST, batch_size )
            self.Save_Processed_Data( DL_LIST,
                                     save_name,
                                     feature_size = FS,
                                     window_len = window_len,
                                     dtype = dtype )

        elif dtype == 'Multiple':
            ALL_DATA_LIST = list()
            FS_LIST = list()
            for i, dim in enumerate(dtype_new):
                new_date_TRAIN = copy.deepcopy( date_TRAIN )
                new_date_VAL = copy.deepcopy( date_TRAIN )
                new_date_TEST = copy.deepcopy( date_TRAIN )
                new_TRAIN = copy.deepcopy( TRAIN )
                new_VAL = copy.deepcopy( VAL )
                new_TEST = copy.deepcopy( TEST )
                new_TRAIN_OUT = copy.deepcopy( TRAIN_OUT )
                new_VAL_OUT = copy.deepcopy( VAL_OUT )
                new_TEST_OUT = copy.deepcopy( TEST_OUT )

                new_date_TRAIN = self.sliding_window_df( new_date_TRAIN, max_window_len, window_len[i], out_size )
                new_date_VAL   = self.sliding_window_df( new_date_VAL, max_window_len, window_len[i], out_size )
                new_date_TEST  = self.sliding_window_df( new_date_TEST, max_window_len, window_len[i], out_size )

                new_TRAIN =  np.array(self.sliding_window_df( new_TRAIN, max_window_len, window_len[ i ], out_size))
                new_VAL   =  np.array(self.sliding_window_df( new_VAL, max_window_len, window_len[ i ], out_size))
                new_TEST  =  np.array(self.sliding_window_df( new_TEST, max_window_len, window_len[ i ], out_size))

                new_TRAIN_OUT = np.array( self.sliding_window_df( new_TRAIN_OUT, max_window_len, window_len[ i ], out_size, OUTPUT = True ) )
                new_VAL_OUT = np.array( self.sliding_window_df( new_VAL_OUT, max_window_len, window_len[ i ], out_size, OUTPUT = True) )
                new_TEST_OUT  = np.array( self.sliding_window_df( new_TEST_OUT, max_window_len, window_len[ i ], out_size, OUTPUT = True ) )
                
                DATA_LIST = [ new_TRAIN, new_TRAIN_OUT, new_VAL, new_VAL_OUT, new_TEST, new_TEST_OUT ]



                FS = np.array( TRAIN ).shape[ 1 ]
                FS_LIST.append( FS )

                dim = int(dim)
                if dim == 1:
                    ALL_DATA_LIST.append( DATA_LIST )

                elif dim == 2:

                    DATA_LIST = self.Recurrence_Plot_DATA_LIST( DATA_LIST )
                    ALL_DATA_LIST.append( DATA_LIST )

            TRAIN_LIST = list()
            VAL_LIST = list()
            TEST_LIST = list()

            TRAIN_OUT_LIST = list()
            VAL_OUT_LIST = list()
            TEST_OUT_LIST = list()


            for BRANCH in range( np.array( ALL_DATA_LIST ).shape[0] ):
                [ TEMP_TRAIN, TEMP_TRAIN_OUT, TEMP_VAL, TEMP_VAL_OUT, TEMP_TEST, TEMP_TEST_OUT ] = ALL_DATA_LIST[ BRANCH ]

                TRAIN_LIST.append( TEMP_TRAIN )
                VAL_LIST.append( TEMP_VAL )
                TEST_LIST.append( TEMP_TEST )

                TRAIN_OUT_LIST.append( TEMP_TRAIN_OUT )
                VAL_OUT_LIST.append( TEMP_VAL_OUT )
                TEST_OUT_LIST.append( TEMP_TEST_OUT )

            LIST_OF_LISTS = [ TRAIN_LIST, TRAIN_OUT_LIST, VAL_LIST, VAL_OUT_LIST, TEST_LIST, TEST_OUT_LIST ]
            
        DL_LIST = self.make_batch_multiple( LIST_OF_LISTS, batch_size )
        
        self.Save_Processed_Data( DL_LIST,
                                save_name,
                                feature_size = FS,
                                window_len = window_len,
                                dtype = dtype )



    def make_batch_single(self, DATA_LIST, batch_size):
        TRAIN, TRAIN_OUT, VAL, VAL_OUT, TEST, TEST_OUT = DATA_LIST 
        
        #COMBINE INPS WITH OUTS
        #COMBINE INPS WITH OUTS
        TRAIN_DS = TensorDataset( TRAIN, TRAIN_OUT )
        VAL_DS = TensorDataset( VAL, VAL_OUT )
        TEST_DS = TensorDataset( TEST, TEST_OUT )

        #BATCH ALL DATA
        #BATCH ALL DATA
        TRAIN_DL, VAL_DL, TEST_DL = DataLoader( TRAIN_DS, batch_size ), DataLoader( VAL_DS, batch_size ), DataLoader( TEST_DS, batch_size )
        return [TRAIN_DL, VAL_DL, TEST_DL]

     
    def Recurrence_Plot_ONE_DATASET( self, DATA ):
        FS = DATA.shape[ 2 ]
        SAMPLES = DATA.shape[ 0 ]
        ALL_PLOTS = list()

        for SMP in range( SAMPLES ):
            SAMPLE_PLOTS = list()
            TEMP_SAMPLE = DATA[ SMP ]

            for FEAT in range( FS ):
                FEAT_VALUES = TEMP_SAMPLE[ :, FEAT ] 
                SAMPLE_PLOTS.append( rec_plot( FEAT_VALUES ) )
            ALL_PLOTS.append( SAMPLE_PLOTS )
        return np.array( ALL_PLOTS )




    def Recurrence_Plot_DATA_LIST( self, DATA_LIST ):
        [ TRAIN, TRAIN_OUT, VAL, VAL_OUT, TEST, TEST_OUT ] = DATA_LIST


        TRAIN_PLOTS = self.Recurrence_Plot_ONE_DATASET( TRAIN )
        VAL_PLOTS = self.Recurrence_Plot_ONE_DATASET( VAL )
        TEST_PLOTS = self.Recurrence_Plot_ONE_DATASET( TEST )

        return [ TRAIN_PLOTS, TRAIN_OUT, VAL_PLOTS, VAL_OUT, TEST_PLOTS, TEST_OUT ] 



    def Save_Processed_Data(self, NEW_DATA_LIST, save_name,  feature_size, window_len, dtype = 'Single'):
        try:
            os.mkdir('DATA')
        except:
            pass
        try:
            os.mkdir('DATA/XU100-29022020')
        except:
            pass
        try:
            os.mkdir('DATA/XU100-29022020/Single')
        except:
            pass

        try:
            os.mkdir('DATA/XU100-29022020/Multiple')
        except:
            pass

        FS_string = str(feature_size)

        if dtype == 'Single':
            window_len_string = str(window_len)
        elif dtype == 'Multiple':
            window_len_string = str( window_len[ 0 ] )  + 'x' + str( window_len[ 1 ] ) 

        TRAIN_DL, VAL_DL, TEST_DL = NEW_DATA_LIST
        torch.save( TRAIN_DL, 'DATA/XU100-29022020/' + dtype + '/' +  save_name + '_TRAIN')
        torch.save( VAL_DL,'DATA/XU100-29022020/' + dtype + '/' + save_name + '_VAL')
        torch.save( TEST_DL,'DATA/XU100-29022020/' + dtype + '/' + save_name + '_TEST')

        with open( 'DATA/XU100-29022020/' + dtype + '/' +  save_name + ".txt", "w") as file:
            file.write('DATA/XU100-29022020/' + dtype + '/' + save_name)

            file.write('\n')
            file.write('feature_size=' + FS_string)
            file.write('\n')
            file.write('INPUT_DIMS=' + window_len_string)


        with open("DATA/in_use_info.txt", "w") as file:
            file.write('DATA/XU100-29022020/' + dtype + '/' + save_name)
            file.write('\n')
            file.write('feature_size=' + FS_string)
            file.write('\n')
            file.write('INPUT_DIMS=' + window_len_string)




    def make_batch_multiple(self, LIST_OF_LISTS, batch_size ):
        [ TRAIN_LIST, TRAIN_OUT_LIST, VAL_LIST, VAL_OUT_LIST, TEST_LIST, TEST_OUT_LIST ] = LIST_OF_LISTS

        SAME = True
        for x in TRAIN_OUT_LIST:
            comparison = x == TRAIN_OUT_LIST[ 0 ]
            equal = comparison.all()
            if equal == False:
                SAME = False
        train_all_same = SAME        
        
        SAME = True
        for x in VAL_OUT_LIST:
            comparison = x == VAL_OUT_LIST[ 0 ]
            equal = comparison.all()
            if equal == False:
                SAME = False
        val_all_same = SAME        


        SAME = True
        for x in VAL_OUT_LIST:
            comparison = x == VAL_OUT_LIST[ 0 ]
            equal = comparison.all()
            if equal == False:
                SAME = False
        test_all_same = SAME        



        if train_all_same == False or val_all_same == False or test_all_same == False:
            raise Exception('DATA shifted???????????????????????????')

        TRAIN_LIST = list( map( torch.Tensor, TRAIN_LIST ) )
        TRAIN_OUT_LIST = list( map( torch.Tensor, TRAIN_OUT_LIST ) )

        VAL_LIST = list( map( torch.Tensor, VAL_LIST ) )
        VAL_OUT_LIST = list( map( torch.Tensor, VAL_OUT_LIST ) )

        TEST_LIST = list( map( torch.Tensor, TEST_LIST ) )
        TEST_OUT_LIST = list( map( torch.Tensor, TEST_OUT_LIST ) )

        TRAIN_DS = TensorDataset( *TRAIN_LIST, TRAIN_OUT_LIST[ 0 ] )
        VAL_DS = TensorDataset( *VAL_LIST, VAL_OUT_LIST[ 0 ] )
        TEST_DS = TensorDataset( *TEST_LIST, TEST_OUT_LIST[ 0 ] )

        TRAIN_DL, VAL_DL, TEST_DL = DataLoader( TRAIN_DS, batch_size ), DataLoader( VAL_DS, batch_size ), DataLoader( TEST_DS, batch_size )
        return [TRAIN_DL, VAL_DL, TEST_DL]

