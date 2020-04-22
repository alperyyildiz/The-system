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
                            'Linear': [int,int,self.Bool_Rework],
                            'MaxPool1d': [int,int,int,int,self.Bool_Rework,self.Bool_Rework],
                            'MaxPool2d': [int,int,int,int,self.Bool_Rework,self.Bool_Rework],
                            'BatchNorm1d': [int, float, float, self.Bool_Rework, self.Bool_Rework],
                            'BatchNorm2d': [int, float, float, self.Bool_Rework, self.Bool_Rework]
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
                           'conv2d': {'in_channels':None,
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
                                           'track_running_stats':True}}



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
              if Dict_Block[layer][0] in ['Linear','conv1d']:
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
            Dict_Branch[str(count_block)] = self.Blocks[block_name]

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



    def Append_Branch_to_Network(self,Branch,All_Flatten):
        #Append the branch to the network
        #Flatten all branches if needed 
        
        #Get block list in the branch
        branch = copy.deepcopy(Branch)
        network_layers = list(self.network.keys())
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


    def Append_Flatten_to_Network(self):

        network_layers = list(self.network.keys())
        count = int(network_layers[-1])
        self.network[str(count+1)] = ['Flatten',[1,-1]]
        self.out_channels = self.seq_len * self.out_channels
        self.seq_len = 1


    def seq_len_processor(self,layer):
        params = layer[1]
        if layer[0] == 'conv1d':
            self.out_channels = params[1]
            self.seq_len = int(np.floor((self.seq_len + 2*params[4] - params[5]*(params[2]-1)-1)/params[3] + 1))

        elif layer[0] == 'MaxPool1d':
            self.seq_len = int(np.floor((self.seq_len + 2*params[2] - params[3]*(params[0]-1)-1)/params[1] + 1))
        if self.seq_len < 0:
            raise Exception('Seq length becomes negative')

    def track_seq_len(self,block,seq_length):
        #Method to track sequence length of forward input
        #Used to calculate input channels for the layer after flatten
        #Also will be used to debugging in the future
        for layer in list(block.keys()):
            params = block[layer][1]
            if block[layer][0] == 'conv1d':
                
                seq_length = int(np.floor((seq_length + 2*params[4] - params[5]*(params[2]-1)-1)/params[3] + 1))
            elif block[layer][0] == 'MaxPool1d':
                seq_length = int(np.floor((seq_length + 2*params[2] - params[3]*(params[0]-1)-1)/params[1] + 1))

        return seq_length

 
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

        
class NET(MAIN_OBJ):
    def __init__(self):
        super().__init__()
        self.seq_len = 250
        self.feature_size = 50

        self.Blocks = OrderedDict()
        self.Branches_Created = OrderedDict()

        self.network = OrderedDict()

        self.ALL_NETWORKS = OrderedDict()
        self.ALL_BRANCHES = OrderedDict()
        
        self.num_of_branches = 0
        self.num_of_networks = 0

        self.order = list()
        
        

class Model(nn.Module):
    def __init__(self, SOURCE_OBJ):
        super().__init__()
        self.play_list = nn.ModuleDict()
        self.__dict__.update(SOURCE_OBJ.__dict__)

        ind_branch = 0
        ind_block = 0
        self.layers = nn.ModuleList()

        for i in range(self.num_of_branches + self.num_of_networks):

            if self.order[i] == 'branch':
                ind_branch = ind_branch + 1
                self.layers.append(Branch(self.ALL_BRANCHES[str(ind_branch)]))

            elif self.order[i] == 'block':
                ind_block = ind_block + 1
                self.layers.append(Block(self.ALL_NETWORKS[str(ind_block)]))


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
            return nn.LSTM(*args)
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
            return nn.LSTM(*args)
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

        BLOCK_LIST = list(BRANCH.keys())
        self.Blocks = nn.ModuleDict()
        for BLOCK in BLOCK_LIST:
            self.Blocks[BLOCK] = Block(BRANCH[BLOCK])
        

    def forward(self,x):
        block_key_list = self.Blocks.keys()
        for i, BLOCK in enumerate(block_key_list):
            if i == 0:
                branch_concat_out = torch.Tensor(self.Blocks[BLOCK](x))
            else:
                branch_concat_out = torch.cat([branch_concat_out,self.Blocks[BLOCK](x)],dim = 1)
        return branch_concat_out
    def layer_add(self,submodule,key,*args):
        submodule.append(self.layer_set(key,*args))
        return submodule

    def layer_set(self,key,*args):
        ## push args into key layer type, return it
        ## push args into key layer type, return it
        if key == 'conv1d':
            return nn.Conv1d(*args)
        elif key == 'LSTM':
            return nn.LSTM(*args)
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
        
        
NN = NET()

NN.Branches_Created = BB
NN.Blocks = DD

NN.Append_Block_to_Network(NN.Blocks['5con'])
NN.Append_Block_to_Network(NN.Blocks['5con'])
NN.Append_Branch_to_Network(NN.Branches_Created['branchbaby'],True)
NN.Append_Block_to_Network(NN.Blocks['End'])
NN.Finish_Building()
