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
                            'Dropout': [float,self.Bool_Rework]
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
            self.num_of_networks = self.num_of_networks + 1
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

    def track_out_channels_branch(self,branch,out_channels,seq_len):
        BRANCH = copy.deepcopy(Branch)
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)
        out_channels_list = list()
        for [TYPE,NUM] in BB_keys:
            BB = BRANCH[TYPE + '-' + NUM]
            out_channels_list

            
    def track_out_channels_block(self, block, out_channels, seq_len):
        BLOCK = copy.deepcopy(Block)
        block_layers = list(BLOCK.keys())
        for layer in block_layers:
            out_channels = self.out_channels_processor(BLOCK[layer])
        return out_channels
            
    def track_out_channels(self,layer,out_channels, seq_len):
        
        PARAMS = layer[1]
        TYPE = layer[0]
        if TYPE in ['LSTM','conv1d']:
            out_channels = PARAMS[1]
        elif TYPE == 'Flatten':
            out_channels = seq_len * out_channels
        return out_channels


    def seq_len_processor(self,layer):
        PARAMS = layer[1]
        if layer[0] == 'conv1d':
            self.seq_len = int(np.floor((self.seq_len + 2*PARAMS[4] - PARAMS[5]*(PARAMS[2]-1)-1)/PARAMS[3] + 1))

        elif layer[0] in ['MaxPool1d', 'AVERAGEPOOL']:
            self.seq_len = int(np.floor((self.seq_len + 2*PARAMS[2] - PARAMS[3]*(PARAMS[0]-1)-1)/PARAMS[1] + 1))
        if self.seq_len < 0:
            raise Exception('Seq length becomes negative')
    
    def track_seq_len_branch(self,Branch, seq_len):
        BRANCH = copy.deepcopy(Branch)
        BB_keys = list(BRANCH.keys())
        BB_keys = self.splitter(BB_keys)
        seq_len_list = list()
        Flat = self.check_flatten_branch(BRANCH)
        for bb_number,[TYPE,NUM] in enumerate(BB_keys):

            BB = BRANCH[TYPE + '-' + NUM],seq_len

            if TYPE == 'block':              
                seq_len_list.append(self.track_seq_len_block(BB,seq_len))
            elif TYPE = 'branch':
                sel_len_list.append(self.track_seq_len_branch(BB,seq_len))
        
        
        All_Same = all(x == seq_len_list[0] for x in seq_len_list)


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
        elif ALL_SAME = False:
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

    def last_normal_layer(self,block):
        layers = list(block.keys())
        normal_layer_need = True
        count = 0
        while normal_layer_need:
            count = count - 1
            if block[layers[count]][0] not in ['MaxPool1d','Dropout','Flatten']:
                the_layer = count
                normal_layer_need = False
                print(block[layers[count]][0])
        return the_layer

    def Check_Flatten(self,block):
        flatten = False
        layerlist = list(block.keys())
        for layer in layerlist:
            if block[layer][0] == 'Flatten':
                flatten = True
        return flatten

    def Check_Flatten_for_Branch(self,Branch):
        branch = copy.deepcopy(Branch)

        All_Flatten = False
        blocks = list(branch.keys())
        for BLOCK in blocks:
            flatten = False
            for LAYER in list(branch[BLOCK].keys()):
                if branch[BLOCK][LAYER][0] is 'Flatten':
                    flatten = True
                    cc = cc + 1
                    break
            
        if cc == len(blocks):
            All_Flatten = True
        return All_Flatten


    def Calculate_Output_Seq_Len_of_Branch_V2(self,Branch,seq_len_list):
        branch = copy.deepcopy(Branch)
        BB_keys = list(branch.keys())
        BB_keys = self.splitter(BB_keys)
        for bb_number,[TYPE,NUM] in enumerate(BB_keys):
            if TYPE == 'block':
                seq_len_list[bb_number] = int(np.floor(self.track_seq_len(branch[BLOCK],seq_len_list[bb_number])))
                if seq_len_list[bb_number] < 0:
                    raise Exception('seq_len become negative in branch {}'.format(BLOCK))
            elif TYPE == 'branch':
                pass
        return seq_len_list


    def Calculate_Output_Seq_Len_of_Branch(self,Branch,seq_len_list):
        branch = copy.deepcopy(Branch)
        blocks = list(branch.keys())

        for block_number,BLOCK in enumerate(blocks):
            seq_len_list[block_number] = int(np.floor(self.track_seq_len(branch[BLOCK],seq_len_list[block_number])))
            if seq_len_list[block_number] < 0:
                raise Exception('seq_len become negative in branch {}'.format(BLOCK))
        return seq_len_list

    def List_Output_Channels_of_Branch(self,Branch):
        branch = copy.deepcopy(Branch)
        blocks = list(branch.keys())

        out_ch_list = list()
        for BLOCK in blocks:
            index = self.last_normal_layer(branch[BLOCK])
    
            layer_list = list(branch[BLOCK].keys())
            out_ch_list.append(branch[BLOCK][layer_list[index]][1][1])
        return out_ch_list

    def Append_Block_to_Network(self, Block):
        #Append the block to the Network
        block = copy.deepcopy(Block)
        
        network_layers = list(self.network.keys())
        block_layers = list(block.keys())
        
        if len(network_layers) > 0 or len(list(self.ALL_NETWORKS.keys())) > 0:
            At_The_Begin = False
        else:
            At_The_Begin = True
            self.Branch_First = False
            self.num_of_networks = self.num_of_networks + 1
            self.order.append('block')
        
        if At_The_Begin is False:
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


    def Append_Branch_to_Network_v2(self,Branch,All_Flatten):
        #Append the branch to the network
        #Flatten all branches if needed 

        #Get block list in the branch
        branch = copy.deepcopy(Branch)
        network_layers = list(self.network.keys())

        BB_keys = list(branch.keys())
        BB_keys = self.splitter(BB_keys)
        
        
        num_of_blocks = len(blocks)

        #Check if the branch in the beginning or not
        if len(network_layers) > 0:
            At_The_Begin = False
        else:
            At_The_Begin = True
            self.Branch_First = True
        
        #Calculates Branch Output Seq_lengths
        if At_The_Begin is False:
            init_seq_len_list = [self.seq_len for x in range(len(blocks))] 
        else:
            seq_list = input('Provide input sequences, separate them with comma(,) ')
            init_seq_len_list = seq_list.split(",")
        final_seq_len_list = self.Calculate_Output_Seq_Len_of_Branch(branch,init_seq_len_list)
        #Check if all seq_lengths are the same
        #Check if all blocks have flatten layer
        All_Same = all(x == final_seq_len_list[0] for x in final_seq_len_list)

        #Set In_Channels
        for [TYPE,NUM] in BB_keys:
            if At_The_Begin is False:
                branch[BLOCK]['1'][1][0] = self.out_channels
            else: 
                branch[BLOCK]['1'][1][0] = self.feature_size

            if All_Flatten == True:
                self.All_Flatten = True
                layerlist_of_block = list(branch[BLOCK].keys())
                branch[BLOCK][str(int(layerlist_of_block[-1]) + 1)] = ['Flatten', [1,-1]]

        out_channels_list = self.List_Output_Channels_of_Branch(branch)



        #If output seq lengths varying, Flatten assumed to be used
        #Error will be raised if seq_lengths are not equal and
        #and Flatten is not used in each block
        if All_Same is False:
            #SEQUENCE LENGHTS ARE IN-CONSISTENT
            #SEQUENCE LENGHTS ARE IN-CONSISTENT
            if All_Flatten is False:
                raise Exception('You need Flatten or set parameters to have same seq_len from branch blocks')
            else:
                out_channels = 0
                for i, BLOCK in enumerate(blocks):
                    out_channels = out_channels +  final_seq_len_list[i] * out_channels_list[i]
                    print('out channels: {}'.format(out_channels))
                    print('final_seq_len_list: {}'.format(final_seq_len_list[i]))
                    print('out_channels_list: {}'.format(out_channels_list[i]))

                self.seq_len = 1
        else:
            #SEQUENCE LENGHTS ARE CONSISTENT
            #SEQUENCE LENGHTS ARE CONSISTENT
            out_channels = 0
            if All_Flatten is False:
                for i, BLOCK in enumerate(blocks):
                    out_channels = out_channels + out_channels_list[i]
            else:
                for i, BLOCK in enumerate(blocks):
                    out_channels = out_channels +  final_seq_len_list[i] * out_channels_list[i]
                self.seq_len = 1


        self.ALL_NETWORKS[str(self.num_of_networks)] = self.network
        del self.network
        self.network = OrderedDict()
        self.num_of_networks = self.num_of_networks + 1

        self.out_channels = out_channels
        self.num_of_branches = self.num_of_branches + 1 

        self.ALL_BRANCHES[str(self.num_of_branches)] = branch
        
        self.order.append('branch') 


    def Append_Branch_to_Network(self,Branch,All_Flatten):
        #Append the branch to the network
        #Flatten all branches if needed 
        self.First = False
        #Get block list in the branch
        branch = copy.deepcopy(Branch)
        network_layers = list(self.network.keys())
        print(list(self.network.keys()))
        blocks = list(branch.keys())
        num_of_blocks = len(blocks)

        #Check if the branch in the beginning or not
        if len(network_layers) > 0:
            At_The_Begin = False
        else:
            At_The_Begin = True
            self.Branch_First = True
        
        #Calculates Branch Output Seq_lengths
        if At_The_Begin is False:
            init_seq_len_list = [self.seq_len for x in range(len(blocks))] 
        else:
            seq_list = input('Provide input sequences, separate them with comma(,) ')
            init_seq_len_list = seq_list.split(",")
        final_seq_len_list = self.Calculate_Output_Seq_Len_of_Branch(branch,init_seq_len_list)
        #Check if all seq_lengths are the same
        #Check if all blocks have flatten layer
        All_Same = all(x == final_seq_len_list[0] for x in final_seq_len_list)

        #Set In_Channels
        for BLOCK in blocks:
            if At_The_Begin is False:
                branch[BLOCK]['1'][1][0] = self.out_channels
            else: 
                branch[BLOCK]['1'][1][0] = self.feature_size

            if All_Flatten == True:
                self.All_Flatten = True
                layerlist_of_block = list(branch[BLOCK].keys())
                branch[BLOCK][str(int(layerlist_of_block[-1]) + 1)] = ['Flatten', [1,-1]]

        out_channels_list = self.List_Output_Channels_of_Branch(branch)



        #If output seq lengths varying, Flatten assumed to be used
        #Error will be raised if seq_lengths are not equal and
        #and Flatten is not used in each block
        if All_Same is False:
            #SEQUENCE LENGHTS ARE IN-CONSISTENT
            #SEQUENCE LENGHTS ARE IN-CONSISTENT
            if All_Flatten is False:
                raise Exception('You need Flatten or set parameters to have same seq_len from branch blocks')
            else:
                out_channels = 0
                for i, BLOCK in enumerate(blocks):
                    out_channels = out_channels +  final_seq_len_list[i] * out_channels_list[i]
                    print('out channels: {}'.format(out_channels))
                    print('final_seq_len_list: {}'.format(final_seq_len_list[i]))
                    print('out_channels_list: {}'.format(out_channels_list[i]))

                self.seq_len = 1
        else:
            #SEQUENCE LENGHTS ARE CONSISTENT
            #SEQUENCE LENGHTS ARE CONSISTENT
            out_channels = 0
            if All_Flatten is False:
                for i, BLOCK in enumerate(blocks):
                    out_channels = out_channels + out_channels_list[i]
            else:
                for i, BLOCK in enumerate(blocks):
                    out_channels = out_channels +  final_seq_len_list[i] * out_channels_list[i]
                self.seq_len = 1


        self.ALL_NETWORKS[str(self.num_of_networks)] = self.network
        del self.network
        self.network = OrderedDict()
        self.num_of_networks = self.num_of_networks + 1

        self.out_channels = out_channels
        self.num_of_branches = self.num_of_branches + 1 

        self.ALL_BRANCHES[str(self.num_of_branches)] = branch
        
        self.order.append('branch') 



    def seq_len_tracker(self,layer,seq_len):
        params = layer[1]
        if layer[0] == 'conv1d':
            seq_len = int(np.floor((seq_len + 2*params[4] - params[5]*(params[2]-1)-1)/params[3] + 1))

        elif layer[0] in ['MaxPool1d', 'AVERAGEPOOL']:
            seq_len = int(np.floor((seq_len + 2*params[2] - params[3]*(params[0]-1)-1)/params[1] + 1))
        elif layer[0] == 'Flatten':
            seq_len = 1

        if seq_len < 0:
            raise Exception('Seq length becomes negative')
 
    def Create_Forward_List(self):
        forward_pass_list = list()
        q = True
        while q:
            type_to_be_added = int(input('Add 1 block or branch'))
            if type_to_be_added == 0:
                break
            if type_to_be_added == 1:
                block_name = input('which block to be added? \n {}'.format(list(self.Blocks.keys())))
                forward_pass_list.append(block_name)
            elif type_to_be_added > 1:
                block_names = list()
                for block in range(type_to_be_added):
                    block_name = input('which block to be added? \n {}'.format(list(self.Blocks.keys())))
                    block_names.append(block_name)
                forward_pass_list.append(block_names)
        self.forward_list = forward_pass_list
    
    def Finish_Building(self):
        self.ALL_NETWORKS[str(self.num_of_networks)] = self.network
        self.order.append('block')
