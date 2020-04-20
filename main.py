from collections import OrderedDict 
import copy


class MAIN_OBJ():
    def __init__(self):
        super().__init__()
        self.Blocks = OrderedDict()
        self.Branches = OrderedDict()
        #default input types from pytorch documentation. 
        #User input will be processed by these functions
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


    def create_block(self):
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
        layerlist = [int(x) for x in list(Dict_Block.keys())]
        for layer in layerlist:
            if layer > 1 and  Dict_Block[str(layer)][0] != 'MaxPool1d' and Dict_Block[str(layer-1)][0] != 'MaxPool1d':
                Dict_Block[str(layer)][1][0] = Dict_Block[str(layer-1)][1][1]
            elif layer > 1 and Dict_Block[str(layer)][0] == 'MaxPool1d':
                pass
            elif layer > 1 and Dict_Block[str(layer-1)][0] == 'MaxPool1d':
                Dict_Block[str(layer)][1][0] = Dict_Block[str(layer-2)][1][1]
            elif layer == 1:
                Dict_Block[str(layer)][1][0] = None
                
        self.Blocks[NAME] = Dict_Block



    def create_branch(self):
        #Creates an ordered dictionary
        #Saves layer type with parameters specified by user
        NAME = input('Enter Block Name: ')
        Add_layers = True
        Add_branches = True
        Dict_Branch = OrderedDict()

        count_br = 0
        #While loop until user says stop adding branches
        while Add_branches:
            count_br = count_br + 1
            count_layers =0
            Branch = OrderedDict()
            #While loop until user says stop adding layers
            while Add_layers:
                count = count + 1
                TYPE = input('enter layer type: ')
                print('\n Seperate input with comma, write - for defalt value\n')
                print(self.Def_Params[TYPE])
                print('\n')
                input_params = input()
                input_params = self.input_reworker(TYPE, input_params)
                layer_num = str(count)
                Branch[layer_num] = [TYPE, input_params]

            layerlist = [int(x) for x in list(Branch.keys())]
            for layer in layerlist:
                if layer > 1 and  Branch[str(layer)][0] != 'MaxPool1d' and Branch[str(layer-1)][0] != 'MaxPool1d':
                    Branch[str(layer)][1][0] = Branch[str(layer-1)][1][1]
                elif layer > 1 and Branch[str(layer)][0] == 'MaxPool1d':
                    pass
                elif layer > 1 and Branch[str(layer-1)][0] == 'MaxPool1d':
                    Branch[str(layer)][1][0] = Branch[str(layer-2)][1][1]
                elif layer == 1:
                    Branch[str(layer)][1][0] = None



            Dict_Branch[str(count_br)] = Branch

            add_value = input('You want to add more branches?? \n')
            Add_branches = self.Bool_Rework(add_value)


        self.Branches[NAME] = Dict_Branch

   
    def Append_Block_to_Network(self, BB):
        #Append 2 layer blocks
        
        block = copy.deepcopy(BB)
        
        network_layers = list(self.network.keys())
        block_layers = list(block.keys())
        
        if len(network_layers) > 0:
            normal_layer_need = True
            count = 0
            while normal_layer_need:
                count = count - 1
                if network_layers[count][0] not in ['MaxPool1d','Dropout']:
                    last_normal_layer = count
                    normal_layer_need = False

            block_2[block_layers[0]][1][0] = self.network[network_layers[last_normal_layer]][1][1]
            count = int(network_layers[-1])

            for lay in block_layers:
                count = count + 1
                self.network[str(count)] = block[lay]
        else:
            block_2[block_layers[0]][1][0] = self.seq_len
            count = 0
            for lay in block_layers:
                count = count + 1
                self.network[str(count)] = block[lay]
        self.seq_len = self.track_seq_length(block,self.seq_len)
                
    def Append_Branch_to_Network(self,Br,same_seq_len = True):
        #Append a block and a branch
        branch = copy.deepcopy(Br)
        
        network_layers = list(self.network.keys())
        
        blocks = list(branch.keys())

        if len(network_layers) > 0:

            normal_layer_need = True
            count = 0
            while normal_layer_need:
                count = count - 1
                if network_layers[count][0] not in ['MaxPool1d','Dropout']:
                    last_normal_layer = count
                    normal_layer_need = False
            for br in blocks:

                branch[br]['1'][1][0] = self.network[network_layers[last_normal_layer]][1][1]
        else:
            for i, br in enumerate(blocks):
                branch[br]['1'][1][0] = self.seq_len_list[i]
        if same_seq_len:
            self.seq_len = self.track_seq_len(branch['1'],self.seq_len)
        else:
           out_channels = 0
           for i, br in enumerate(blocks):
              br_layers = list(branch[br].keys())
              normal_layer_need = True
              count = 0
              while normal_layer_need:
                  count = count - 1
                  if branch[br][count][0] not in ['MaxPool1d','Dropout']:
                      last_normal_layer = count
                      normal_layer_need = False

              s_len = self.track_seq_len(branch[br],self.seq_len)

              out_channels = out_channels +  s_len * branch[br][last_normal_layer][1][1]
        self.branch_out_channels = out_channels
        self.seq_len = 1







    def concat_using_flatten(self, B1, B2, initial_seq_len):
        #Append Blocks by using flatten in between
        block_1 = copy.deepcopy(B1)
        block_2 = copy.deepcopy(B2)
        seq_len = self.track_seq_len(block_1, initial_seq_len)

        b1_layers = list(block_1.keys())
        b2_layers = list(block_2.keys())


        #We need the last neural layer to bound parameters of two consecutive block
        #Dropout and Pooling parameters will not work
        normal_layer_need = True
        count = 0
        while normal_layer_need:
            count = count - 1
            if b1_layers[count][0] not in ['MaxPool1d','Dropout']:
                last_normal_layer = count
                normal_layer_need = False

        #We calculate feature size to set as in_channels to second block
        feature_size = block_1[b1_layers[last_normal_layer]][1][1] * seq_len

        block_2[b2_layers[0]][1][0] = feature_size
        count = 0
        New_Block = OrderedDict()

        for lay in b1_layers:
            count = count + 1
            New_Block[str(count)] = block_1[lay]

        count = count + 1
        New_Block[str(count)] = ['flatten', [1,-1]]

        
        for lay in b2_layers:
            count = count + 1
            New_Block[str(count)] = block_2[lay]
        
        return New_Block


    def seq_len_processor(self):
        if block[layer][0] == 'conv1d':
            self.seq_length = (self.seq_length + 2*params[4] - params[5]*(params[2]-1)-1)/params[3] + 1
        elif block[layer][0] == 'MaxPool1d':
            self.seq_length = (self.seq_length + 2*params[2] - params[3]*(params[0]-1)-1)/params[1] + 1



    def track_seq_len(self,block,seq_length):
        #Method to track sequence length of forward input
        #Used to calculate input channels for the layer after flatten
        #Also will be used to debugging in the future
        for layer in list(block.keys()):
            params = block[layer][1]
            if block[layer][0] == 'conv1d':
                seq_length = (seq_length + 2*params[4] - params[5]*(params[2]-1)-1)/params[3] + 1
            elif block[layer][0] == 'MaxPool1d':
                seq_length = (seq_length + 2*params[2] - params[3]*(params[0]-1)-1)/params[1] + 1
        return seq_length

    def Create_Network(self):
        #This will try to create whole network
        #I dont know about branches, what to do with them
        self.seq_length = 100
        block = self.Append_Blocks(self.Blocks['bisi'],self.Blocks['Dense'])
        self.seq_length = self.track_seq_len(block,self.seq_length)
        print(self.seq_length)

    
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


    def Append_Block_to_Branch(self,Bl,Br):
        #Append a branch to a block
        block = copy.deepcopy(Bl)
        branch = copy.deepcopy(Br)

        block_layers = list(block.keys())
        branch_layers = list(branch.keys())
        normal_layer_need = True
        count = 0
        channel_sum = 0

        while normal_layer_need:
            count = count - 1
            if b1_layers[count][0] not in ['MaxPool1d','Dropout']:
                last_normal_layer = count
                normal_layer_need = False



def Network(MAIN_OBJ):
    def __init__(self, single_input = True):
        if single_input:
            self.seq_len = int(input('provide sequence length \n'))
            self.in_channels = int(input('provide feature size \n'))
        else:
            seq_lengths = input('provide sequence lengths \n')
            in_channels = int(input('provide feature size \n'))

            seq_lengths = seq_lengths.split(",")
            self.seq_len_list =  [int(x) for x in seq_lengths]
 




