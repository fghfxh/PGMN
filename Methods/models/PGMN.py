import copy
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import numpy as np  
import torch.nn as nn 
import torch.nn.functional as F
from typing import List, Tuple, Union, Dict,Optional
from dataclasses import dataclass
from Utils.utils import match_dims, ordered_dict_mean
from .base_modular import ModularBaseNet
from Utils.utils import create_mask
from .PGMN_components import ComponentList, PGMN_conv_block, GatedOh
from simple_parsing import choice
####################################
from .gate_network import HyperGate_Net

device = nn_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PGMN_net(ModularBaseNet):
    @dataclass
    class Options(ModularBaseNet.Options):  
        #structural learning rate 
        lr_structural: float = 0.001
        #module selection activation
        mask_activation: str ='softmax'
        #deviation threshold
        deviation_threshold: float = 3.
        #test time oracle
        freeze_module_after_step_limit: bool = False 
        #if K_train=1 then at train tiem we only train the active modules (most recently added) [mostly unsued]
        K_train: Optional[int] = 0 
        #if K_test=1, at eval time we always select the module with max activation
        K_test: Optional[int] = 0
        #use deviation from mean for structural selection [mostly unsued]
        mask_with: bool = choice('str_act','z_score', 'deviation', default='str_act')
        #wether only to use new modules for outliers of old modules
        catch_outliers_old: bool = 1
        #reset_step_count_on_module_addition  
        reset_step_count_on_module_addition: bool = 1
        # wether to mask the outputs of the structural components as well
        mask_str_loss:bool=0                       
        # if 'True' uses task id (if given) to select the corresponding modules in layer forward; has no effect if task_id is not given to the forward (use train- and test-time oracle parameters of the metalearner)
        module_oracle:bool = 0
        #learnign rate used in case of offline ('normal') training
        lr:float = 0.001
        #the way to combine module outputs                                
        concat:str = choice('sum', 'beam', default='sum')     
        #if 'True', module selection is biased towards the expectation in the batch
        str_prior_factor: float = 0
        #tempreture structural prior
        str_prior_temp: float = 1.
        #if True the structural component is only optimized for modules which are 'free' - not yet frozen/learned
        optmize_structure_only_free_modules: bool = True
        #if True the network will check by it self on the batch level when to add new modules
        automated_module_addition: bool = False
        # -                  
        projection_phase_length: int = 2000
        # -
        module_addition_batch_number: int = 0 # if >0 wait for this number of outlier batches in a raw before addign module 
        # -
        ix_layers_below_on_additionf: int =0
        # - 
        active_unfreezing: int = 0
        # -
        unfreeze_structural: int = 0
        # - 
        treat_unfreezing_as_addition: int = 0
        #-
        beam_width: int = 0
        #-
        init_stats: int = 1
        #for ablation
        no_projection_phase: int = 0

    def __init__(self, options:Options = Options(), module_options:PGMN_conv_block.Options=PGMN_conv_block.Options(), gate_options:HyperGate_Net.Options = HyperGate_Net.Options(),i_size: int = 28, channels:int = 1, hidden_size=64, num_classes:int=5,n_tasks:int=6 ):
        super(PGMN_net, self).__init__(options, i_size, channels, hidden_size, num_classes)
        
        self.args: PGMN_net.Options = copy.copy(options)
        self.deviation_threshold=self.args.deviation_threshold
        self.module_options = module_options
        self.gate_options = gate_options
        self.lr_structural=self.args.lr_structural         
        self.catch_outliers_for_old_modules=self.args.catch_outliers_old
        ##############################
        #n_modules - might change over the runtime
        self.register_buffer('_n_modules', torch.tensor([float(self.n_modules)]*self.depth))    
        self.register_buffer('_steps_for_ghnet_MWR', torch.tensor(0.))
        self.register_buffer('_steps_since_last_addition', torch.tensor(0.))
        self.register_buffer('min_str_prior_temp', torch.tensor(float(self.args.str_prior_temp)))
        ##############################
        self.components: List[List[PGMN_conv_block]] = self.components
        self.gate_net: nn.ModuleList = nn.ModuleList()
        # self.gate_net: List[Gate_Net] = self.gate_net  ###gate_net
        self.num_tasks = n_tasks
        self.store_each_layer_str_prior = [0,0,0,0]           #Store the str_prior for each layer
        self.init_modules()    

        self.modules_to_unfreeze_func:List[PGMN_conv_block]=[]
        self.register_buffer('outlier_batch_counter_buffer', torch.tensor([0. for _ in range(self.depth)]))

    @property    
    def total_number_of_modules(self):
        return torch.sum(self._n_modules).item()
    
    @property
    def n_modules(self):
        return self.n_modules_per_layer
    
    @property
    def n_modules_per_layer(self):
        return self._n_modules#.cpu().numpy()
    
    @property
    def optimizers_update_needed(self): 
        return any(l.optimizer_update_needed for l in self.components)
        
    def n_modules_at_layer(self, at_layer):
        return int(self._n_modules[at_layer].item())

    def maybe_add_modules(self, outlier_signal, fixed_modules, layer, bottom_up=True, module_i=None, init_stats=False):
        add_modules_layers = False  
        unfreeze_modules = torch.zeros(len(self.components[layer]))       
        freeze_modules_functional = torch.zeros(len(self.components[layer]))
        if self.training:      
            _outlier_ = outlier_signal>=self.deviation_threshold 
            # print(outlier_signal) 
            criterion_1 = (_outlier_.sum(0)==self.n_modules[layer]) #if all modules report outliers
            criterion_2 = (fixed_modules.sum(0)==self.n_modules[layer])# if all modules are fixed
            add_modules_layers = (criterion_1 & criterion_2).item()
            
            if add_modules_layers and self.args.module_addition_batch_number>0:
                #Can only add modules if outlier was detected in several batches in a row (should be more robust)
                self.outlier_batch_counter_buffer[layer]=self.outlier_batch_counter_buffer[layer]+1
                if not self.outlier_batch_counter_buffer[layer]>=self.args.module_addition_batch_number:
                    add_modules_layers=False                    
            elif not add_modules_layers and self.args.module_addition_batch_number>0:
                self.outlier_batch_counter_buffer[layer]*=0

            #freeze and unfreeze modules
            unfreeze_modules = (fixed_modules==1) & (_outlier_==0) 
            #we dont want to unfreeze modules in layers where new modules have been already added
            unfreeze_modules = unfreeze_modules & (criterion_2)

            #newly added modules will not be frozen here
            freeze_modules_functional = (fixed_modules==1) & (_outlier_==1)

        new_m_idx=[]
        _params_changed=0      
        #prefer unfreezing over addition
        if add_modules_layers and (not self.args.active_unfreezing or (self.args.active_unfreezing and not torch.sum(unfreeze_modules)>0)):
            _params_changed=True  
            print('adding at layers ', layer)
            self.add_modules(at_layer=layer, module_i=module_i, init_stats=init_stats)
            new_m_idx.append(-1)
        else:
            if self.args.active_unfreezing: 
                if torch.sum(unfreeze_modules)>0:     
                    if self.args.treat_unfreezing_as_addition:  
                                    for c in self.components[layer]:
                                       c.on_module_addition_at_my_layer(inner_loop_free=(self.args.regime=='meta'))    
                    for i_m, m in enumerate(unfreeze_modules):
                            if m:     
                                unfreeze_completed = self.components[layer][i_m].unfreeze_learned_module(self.args.unfreeze_structural)
                                _params_changed = max( _params_changed, unfreeze_completed)
                                self.components[layer][i_m].detach_structural=True
                                # if self.args.treat_unfreezing_as_addition:  
                                self.components[layer][i_m].module_learned=torch.tensor(0.)
                                new_m_idx.append(i_m)
                                break
        return add_modules_layers, _params_changed, new_m_idx

    def maybe_add_modules_without_structural(self, init_stats=False, epoch=None, acc_pre_task = None,
                                             acc_test_pre = None, add_new_module_num = 0,task_id = 0):
        add_modules_layers = False
        _params_changed = False
        # unfreeze_modules = torch.zeros(len(self.components[layer]))
        layer_location = 0
        criterion = 0
        if torch.is_tensor(acc_pre_task):
            acc_pre = acc_pre_task.item()
        else:
            acc_pre = acc_pre_task
        if torch.is_tensor(acc_test_pre):
            acc_pre_test = acc_test_pre.item()
        else:
            acc_pre_test = acc_test_pre
        if self.training:
            criterion_1 = ((acc_pre > 0) and (acc_pre < self.args.criterion_1)) #The accuracy of the last iteration experiment was less than 50
            criterion_2 = (epoch == ((self.args.projection_phase_length + self.args.MWR_attempt_phase_length)/self.len_loader))# The arrival of a new task
            criterion_3 = ((acc_pre_test > 0)) and ((acc_pre_test < self.args.criterion_3))
            criterion_4 = (add_new_module_num < 4)
            criterion_5 = task_id>0
            if criterion_2:
                fzy = 5
            add_modules_layers = (criterion_1 or criterion_2 or criterion_3) and criterion_4 and criterion_5
        new_m_idx=[]
        num_modules = []
        if add_modules_layers:
            _params_changed=True

            for i in range(self.depth):
                num = self.n_modules_at_layer(i)
                num_modules.append(num)
            if self.args.add_module_scheme == 1:

                if criterion_1 or criterion_3:  # The accuracy of the last iteration experiment was less than 50
                    if len(num_modules) > 0:
                        min_value = min(num_modules)
                        min_index = num_modules.index(min_value)
                        layer_location = min_index
                        print(min_index)
            elif self.args.add_module_scheme == 2:
                # Add modules from the top
                max_indexes = self.max_index_of_min_duplicates(num_modules, add_new_module_num)
                layer_location = max_indexes
            elif self.args.add_module_scheme == 3:
                min_indexes = self.min_index_of_min_duplicates(num_modules, add_new_module_num)
                layer_location = min_indexes

            add_new_module_num = add_new_module_num+1

            print('adding at layers ', layer_location)

            str_prior = self.store_each_layer_str_prior[layer_location]
            assert (str_prior is not None)
            module_i = str_prior.argmax().item()
            self.add_modules_without_structural(at_layer=layer_location, module_i=module_i, init_stats=init_stats)
            new_m_idx.append(-1)
        return add_modules_layers, _params_changed, new_m_idx, layer_location, add_new_module_num


    def init_modules(self):
        self.encoder = None

        channels_in = self.channels

        if self.args.module_type=='linear':
            channels_in = self.i_size * self.i_size * self.channels  

        out_h=self.i_size  
        deeper=0
        self.str_priors=nn.ModuleList()
        gate_input = []
        hidden_size=self.hidden_size
        for i in range(self.depth):
            components_l = ComponentList()
            dropout_before=self.module_options.dropout
            for m_i in range(self.n_modules_at_layer(i)):
                if self.args.depper_first_layer and i==0:
                    deeper=1
                else:
                    deeper=0  
                module_type=self.args.module_type
                self.block_constructor=PGMN_conv_block

                conv = self.block_constructor(out_h, channels_in, hidden_size, out_h, name=f'components.{i}.{m_i}', module_type=module_type, initial_inv_block_lr=self.lr_structural, 
                                                            deviation_threshold=self.deviation_threshold, freeze_module_after_step_limit=self.args.freeze_module_after_step_limit, deeper=deeper,
                                                                                                options=self.module_options, num_classes=self.num_classes if (self.args.multihead=='modulewise' and i==self.depth-1) else 0)
                self.module_options.dropout=dropout_before
                
                ##################################################
                ###Initialize all modules in layers identically###   
                if self.per_layer_module_initialization[i] is not None:
                    conv.load_state_dict(self.per_layer_module_initialization[i])
                if self.args.module_init=='identical':
                    self.per_layer_module_initialization[i] = conv.state_dict()
                ##################################################
                components_l.append(conv)

            self.components.append(components_l)
            ###########  gate_input   ############
            gate_input_layer = [self.args.batch_size, channels_in, out_h, out_h ]
            if len(gate_input_layer) == 4:
                gate_input.append(gate_input_layer)

            out_h = conv.out_h
            # if module_type!='linear':
            channels_in = hidden_size
            ################################

        self.channels_in = channels_in        
        if self.args.module_type=='resnet_block':
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            out_h=1
        else:
            self.avgpool=nn.Identity()
        if self.args.module_type=='linear':
            out_h=1                             

        if self.args.multihead != 'modulewise':  
            if self.args.multihead=='gated_linear':
                self.representation_dim = self.channels_in*out_h*out_h      
                self.decoder = nn.ModuleList([GatedOh(out_h, self.representation_dim, self.representation_dim, self.num_classes, initial_inv_block_lr=self.lr_structural, module_type='linear', options=self.module_options, name='oh_0')])
            elif self.args.multihead=='gated_conv':
                self.representation_dim = self.channels_in
                self.decoder = nn.ModuleList([GatedOh(out_h, self.channels_in, hidden_size, self.num_classes, initial_inv_block_lr=self.lr_structural, module_type='conv', options=self.module_options, name='oh_0')])
            else: #self.args.multihead=='usual':
                self.representation_dim = self.channels_in*out_h*out_h      
                self.decoder = nn.Linear(self.representation_dim, self.num_classes) if self.args.multihead=='none' else nn.ModuleList([nn.Linear(self.representation_dim, self.num_classes)])
        
        else:
            self.decoder = nn.Identity()

        self.ssl_pred = torch.nn.Identity()        
        self.softmax = nn.Softmax(dim=0)
        ################################
        ######## Create Layer Gate net and Hypernet network    ########

        for i in range(self.depth):

            temb_size = self.args.temb_size

            # num_gate_output = self.n_modules_at_layer(i)   #int(self._n_modules[i].item())
            gate_net = HyperGate_Net(options = self.gate_options, num_tasks = self.num_tasks, temb_size = temb_size, importance=self.args.importance)
            self.gate_net.append(gate_net)

        if self.args.regime=='normal':
            # self.optimizer, self.optimizer_structure = self.get_optimizers()
            self.optimizer, self.optimizer_structure = self.get_optimizers()
            self.optimizer_hnet = self.get_hnet_optimizers()
            #task embedding optimizer
            self.task_emb_optimizer = self.get_task_emb_optimizers()
        self._reset_references_to_learnable_params()

    def fix_oh(self, i):              
        for p in self.decoder[i].parameters():
            p.requires_grad = False 
        try:
            if self.args.regime=='normal':
                self.optimizer, self.optimizer_structure = self.get_optimizers()
        except ValueError:
            #will throw Value error if all modules are frozen
            pass
                
    def add_output_head(self, num_classes=None, init_idx=None, state_dict=None):            
        num_classes = self.num_classes if num_classes is None else num_classes

        if self.args.multihead=='gated_linear':   
            out_h = self.decoder[-1].in_h                      
            l=GatedOh(out_h, self.representation_dim, self.representation_dim, num_classes, initial_inv_block_lr=self.lr_structural, module_type='linear', options=self.module_options, name=f'oh_{len(self.decoder)}')
            if init_idx is not None:
                l.load_state_dict(copy.deepcopy(self.decoder[init_idx].state_dict()))
            self.decoder.append(l.to(device))
        elif self.args.multihead=='gated_conv':
            out_h = self.decoder[-1].in_h
            l=GatedOh(out_h, self.channels_in, self.hidden_size, num_classes, initial_inv_block_lr=self.lr_structural, module_type='conv', options=self.module_options, name=f'oh_{len(self.decoder)}')
            if init_idx is not None:
                l.load_state_dict(copy.deepcopy(self.decoder[init_idx].state_dict()))
            self.decoder.append(l.to(device))

        elif self.args.multihead=='usual':     
            l=nn.Linear(self.representation_dim, num_classes)
            if init_idx is not None:
                l.load_state_dict(copy.deepcopy(self.decoder[init_idx].state_dict()))
            self.decoder.append(l.to(device))
        
        else:
            #single head 
            self.decoder=nn.Linear(self.representation_dim, num_classes).to(device)
            if state_dict is not None:
                for k in state_dict:
                    if hasattr(self.decoder, k):
                        if len(getattr(self.decoder, k).shape)==2:
                            getattr(self.decoder, k).data[:state_dict[k].size(0),:state_dict[k].size(1)] = state_dict[k].data
                        elif len(getattr(self.decoder, k).shape)==1:
                            getattr(self.decoder, k).data[:state_dict[k].size(0)]=state_dict[k].data
                        else:
                            raise NotImplementedError
                              
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
            # self.optimizer, self.optimizer_structure = self.get_optimizers()

    @property
    def projection_phase(self):      
        if self._steps_since_last_addition>=self.args.projection_phase_length:
            return False
        # print(self._steps_since_last_addition)    
        return True

    @property
    def projection_phase_MWR(self):
        if self.args.train_ghnet_MWR and (self.args.MWR_attempt_phase_length>=self._steps_for_ghnet_MWR):
            return True

        return False

    def get_mask(self, temp=None, train_inputs=None, task_id=0, params=None):
        temp = temp if temp is not None else self.args.temp
        if self.mask_activation=='softmax':
            return self.softmax(self.structure[task_id]/temp)
        elif self.mask_activation=='sigmoid':
            return torch.sigmoid(self.structure[task_id]*temp)
        else:
            raise NotImplementedError
           
    def freeze_permanently_functional(self, free_layer=None, inner_loop_free=True):
        for l, layer in enumerate(self.components):
            if l!=free_layer:
                for c, component in enumerate(layer):
                    component.freeze_functional(inner_loop_free)
        if self.args.regime=='normal':
            # self.optimizer, self.optimizer_structure = self.get_optimizers()
            self.optimizer, self.optimizer_structure = self.get_optimizers()
        self._reset_references_to_learnable_params()
    
    def freeze_permanently_structure(self, free_layer=None, clear_backups=True):
        for l, layer in enumerate(self.components):
            if l!=free_layer:
                for c, component in enumerate(layer):
                    if component.active:
                        component._reset_backups()  

        if self.optimizers_update_needed:
            if self.args.regime=='normal':
                # self.optimizer, self.optimizer_structure = self.get_optimizers()
                self.optimizer, self.optimizer_structure = self.get_optimizers()
            self._reset_references_to_learnable_params()
    
    def remove_module(self, at_layer):      
        self.components[at_layer]=self.components[at_layer][:-1]
        self._n_modules[at_layer]+=-1
        
    def add_modules(self, block_constructor=None, at_layer=None, strict=False, module_i=None, verbose=True, init_stats=False):
        #if at_layer == None add to all layers
        if block_constructor is None:            
            block_constructor = self.block_constructor      
        for l, layer in enumerate(self.components):  
            ################################################
            if l != at_layer and at_layer!=None:
                if self.args.freeze_module_after_step_limit and self.args.reset_step_count_on_module_addition:
                    for c in layer:
                        if c.active:
                            #reset step count for other modules
                            c.reset_step_count()

            elif l == at_layer or at_layer==None:
                if verbose:
                    print(f'adding modules at layer {l}')
                self._n_modules[l]+=1
                ################################################
                #new module's params
                in_h = layer[0].in_h
                channels_in = layer[0].in_channels
                channels = layer[0].out_channels
                i_size = layer[0].i_size
                deeper = layer[0].deeper
                module_type = layer[0].module_type         
                self.module_options.kernel_size=layer[0].args.kernel_size
                self.module_options.dropout=layer[0].args.dropout
                # out_h = layer[0].out_h       
                ################################################
                init_params = None         
                if self.args.module_init == 'identical':         
                    init_params = self.per_layer_module_initialization[l]
                elif self.args.module_init == 'mean':
                    if verbose:
                        print('adding module with mean innitialization')
                    l_state_dict=layer.state_dict()
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, l_state_dict.keys()))
                        for k in keys_to_pop:
                            l_state_dict.pop(k, None)
                    init_params = copy.deepcopy(ordered_dict_mean(l_state_dict))
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)
                elif self.args.module_init == 'existing':
                    init_params = layer[0].state_dict()
                elif self.args.module_init == 'previously_active':
                    for c in layer:
                        if c.active:
                            init_params = c.state_dict()                                       
                    assert init_params is not None, f'At least one module in the layer {l} should have been active'
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)
                elif self.args.module_init == 'most_likely' and module_i is not None:

                    init_params=layer[module_i].state_dict()
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)

                for c in layer:
                    if not c.args.anneal_structural_lr:
                        c.on_module_addition_at_my_layer(inner_loop_free=(self.args.regime=='meta'))    
                  
                new_module = block_constructor(in_h, channels_in, channels, i_size, initial_inv_block_lr=self.lr_structural, 
                                                            name=f'components.{l}.{len(layer)}', 
                                                            module_type=module_type, deviation_threshold=self.deviation_threshold, freeze_module_after_step_limit=self.args.freeze_module_after_step_limit, deeper=deeper, options=self.module_options,
                                                            num_classes=self.num_classes if (self.args.multihead=='modulewise' and l==self.depth-1) else 0 )
                if init_params is not None:  
                    if verbose:
                        print('loading state dict new module')     
                    new_module.load_state_dict(init_params, strict)

                new_module.to(device)
                new_module.reset()
                layer.append(new_module)
        self._reset_references_to_learnable_params()        
        if self.args.regime=='normal':
            self.optimizer, self.optimizer_structure = self.get_optimizers()
             
    def get_optimizers_copy(self):
        #set seperate optimizer for structure and model
        structure_names = ['inv_block', 'structure']          
        model_params = [param for n, param in self.named_parameters() if not any(map(n.__contains__, structure_names)) and param.requires_grad]

        try:
            optimizer = torch.optim.Adam(model_params, lr=self.args.lr, weight_decay=self.args.wdecay)
        except:
            optimizer=None
        structure_param_groups = []  
        #create module specific parameter groups
        for layer in self.components:
            for component in layer:   
                if hasattr(component, 'inv_block'):            
                    params = list(filter(lambda p: p.requires_grad, component.inv_block.parameters()))
                    if len(params)>0:
                        structure_param_groups.append({'params': params, 'lr': component.block_lr_buffer, 'name': component.name})

        if 'gated' in self.args.multihead:
            for decoder in self.decoder:
                if hasattr(decoder, 'inv_block'):            
                        params = list(filter(lambda p: p.requires_grad, decoder.inv_block.parameters()))
                        if len(params)>0:
                            structure_param_groups.append({'params': params, 'lr': decoder.block_lr_buffer, 'name': decoder.name})

        if len(structure_param_groups)>0:
            optimizer_structure = torch.optim.Adam(structure_param_groups,weight_decay=self.args.wdecay) # lr=self.args.lr_structural)
        else:
            optimizer_structure=None   
        return optimizer, optimizer_structure

    def get_optimizers(self):
        #set seperate optimizer for structure and model
        structure_names = ['inv_block', 'structure']
        model_param = []
        model_params = [param for n, param in self.named_parameters() if not any(map(n.__contains__, structure_names)) and param.requires_grad]
        for layer in self.components:
            for component in layer:
                params = [param for n, param in component.named_parameters() if
                            not any(map(n.__contains__, structure_names)) and param.requires_grad]
                if len(params) > 0:
                    model_param.append({'params': params})
        #####decoder.classifier  parameters
        if 'gated' in self.args.multihead:
            for decoder in self.decoder:
                if hasattr(decoder, 'module'):
                    params = list(filter(lambda p: p.requires_grad, decoder.module.parameters()))
                    if len(params) > 0:
                        model_param.append({'params': params})

                if hasattr(decoder, 'classifier'):
                    params = list(filter(lambda p: p.requires_grad, decoder.classifier.parameters()))
                    if len(params) > 0:
                        model_param.append({'params': params})

        try:
            optimizer = torch.optim.Adam(model_param, lr=self.args.lr, weight_decay=self.args.wdecay)
        except:
            optimizer=None

        structure_param_groups = []
        #create module specific parameter groups
        if self.args.use_structural_for_outlier:
            for layer in self.components:
                for component in layer:
                    if hasattr(component, 'inv_block'):
                        params = list(filter(lambda p: p.requires_grad, component.inv_block.parameters()))
                        if len(params)>0:
                            structure_param_groups.append({'params': params, 'lr': component.block_lr_buffer, 'name': component.name})

        if 'gated' in self.args.multihead:
            for decoder in self.decoder:
                if hasattr(decoder, 'inv_block'):
                        params = list(filter(lambda p: p.requires_grad, decoder.inv_block.parameters()))
                        if len(params)>0:
                            structure_param_groups.append({'params': params, 'lr': decoder.block_lr_buffer, 'name': decoder.name})

        if len(structure_param_groups)>0:
            optimizer_structure = torch.optim.Adam(structure_param_groups,weight_decay=self.args.wdecay) # lr=self.args.lr_structural)
        else:
            optimizer_structure=None
        return optimizer, optimizer_structure

    def get_hnet_optimizers(self):
        hnet_param_groups = []

        for gate_hnet in self.gate_net:
            params = list(filter(lambda p: p.requires_grad, gate_hnet.PoolGateNet.theta))
            if len(params)>0:
                hnet_param_groups.append({'params': params})
        if len(hnet_param_groups)>0:
            optimizer_hnet = torch.optim.Adam(hnet_param_groups,lr = gate_hnet.args.lr , weight_decay=self.args.wdecay, betas=(0.99, 0.999)) # lr=self.args.lr_structural)
        else:
            optimizer_hnet=None
        return optimizer_hnet

    def get_task_emb_optimizers(self,task_id: int = 0):
        task_emb_param_groups = []
        if self.args.train_embeddings:
        # Set the embedding optimizer only for the current task embedding.
        # Note that we could here continue training the old embeddings.
            for gate in self.gate_net:
                params = [gate.PoolGateNet.get_task_emb(task_id)]
                # print('%-40s%-20s%s' %(f.grad, f.requires_grad, f.is_leaf))
                if len(params)>0:
                    task_emb_param_groups.append({'params': params})
            if  len(task_emb_param_groups)>0:
                d_emb_optimizer =  torch.optim.Adam(task_emb_param_groups, lr=self.args.dec_lr_emb, weight_decay=self.args.wdecay, betas=(0.99, 0.999))
            else:
                d_emb_optimizer = None
        else:
            d_emb_optimizer = None
        return d_emb_optimizer

    def forward_layer(self, layer, X, beam_results, temp, info='', log_at_this_iter_flag=False):
        #collect module outputs in collections
        X_layer = []
        str_layer = []    
        str_from_backup_detached_layer = []
        str_from_backup_layer = []
        deviation_layer = []
        z_scores_layer = []
        outliers_layer = []   
        module_learned_layer = []     
        fixed_functional_layer = []

        if len(beam_results)>0:   
            #in case of beam search         
            for X in beam_results:
                X=X[-1]
                beam_X_layer = []
                beam_str_layer = []     
                beam_str_from_backup_detached_layer = []
                beam_str_from_backup_layer = []
                beam_deviation_layer = []
                beam_z_scores_layer = []
                beam_outliers_layer = []   
                beam_module_learned_layer = []     
                beam_fixed_functional_layer = []
                for _, conv in enumerate(layer):        
                    #collect outputs from each module at layer for each beam path                                
                    func_out, str_to_minimize, str_from_backup, deviation, outliear_score, outlier, is_learned, fixed_functional = conv(X, info=info, log_at_this_iter=log_at_this_iter_flag)
                    beam_z_scores_layer.append(outliear_score)
                    beam_module_learned_layer.append(is_learned) #[m_i]=age
                    beam_fixed_functional_layer.append(fixed_functional) #[m_i]=fixed_functional
                    beam_outliers_layer.append(outlier)
                    beam_X_layer.append(func_out)
                    beam_str_layer.append(str_to_minimize) # torch.norm(torch.norm(inv.mean(dim=0), p=2, dim=0).mean(0), p=2)
                    beam_str_from_backup_detached_layer.append(str_from_backup.detach())
                    beam_str_from_backup_layer.append(str_from_backup)
                    beam_deviation_layer.append(deviation)
                assert len(beam_X_layer)>0
                beam_str_layer_for_masks_layer = torch.stack(beam_str_from_backup_detached_layer)  


                beam_deviation_layer = torch.stack(beam_deviation_layer) if beam_deviation_layer[0] is not None else torch.zeros_like(beam_str_layer)
                beam_module_learned_layer = torch.tensor(beam_module_learned_layer).to(device)
                beam_fixed_functional_layer = torch.tensor(beam_fixed_functional_layer).to(device)
                
                #############################################
                beam_z_scores_layer = torch.stack(beam_z_scores_layer)

                idx_old_modules = torch.where(beam_module_learned_layer>0)
                idx_new_modules = torch.where(beam_module_learned_layer==0)      

                if self.catch_outliers_for_old_modules and not self.args.K_train:   
                    if self.training and len(idx_new_modules[0])>0 and len(idx_old_modules[0])>0:    
                        idx_outlier_old_modules = torch.where(beam_outliers_layer[idx_old_modules].sum(0)==len(idx_old_modules[0]))[0] #outliers for all old modules
                        beam_str_layer_for_masks_layer[idx_new_modules,idx_outlier_old_modules]=1e-10
                        # print(str_layer_for_masks_layer)
                        beam_deviation_layer[idx_new_modules,idx_outlier_old_modules]=1e-10
                
                # add the current beam path output to the collection of outputs for the lauyer
                X_layer.extend(beam_X_layer)
                str_layer.extend(beam_str_layer)
                str_from_backup_layer.extend(beam_str_from_backup_layer)    
                str_from_backup_detached_layer.append(beam_str_layer_for_masks_layer)
   
                deviation_layer.append(beam_deviation_layer)
                z_scores_layer.append(beam_z_scores_layer)
                fixed_functional_layer.append(beam_fixed_functional_layer)
                module_learned_layer.append(beam_module_learned_layer)
                # outliers_layer.extend(beam_outliers_layer)
            #############################################
            #TODO: if no new modules adeed to a layer, then dont fix funcitonal compnent of a module
            str_layer = torch.stack(str_layer)               
            str_layer_from_backup_layer = torch.stack(str_from_backup_layer)
            str_layer_for_masks_layer = torch.cat(str_from_backup_detached_layer) 
            z_scores_layer = torch.stack(z_scores_layer).mean(0)
            deviation_layer = torch.zeros_like(z_scores_layer) #torch.cat(deviation_layer)
            fixed_functional_layer=torch.cat(fixed_functional_layer)
            module_learned_layer =torch.cat(module_learned_layer)
            return torch.stack(X_layer), str_layer, str_layer_from_backup_layer, str_layer_for_masks_layer, deviation_layer, z_scores_layer, fixed_functional_layer, beam_module_learned_layer
        else:
            for _, conv in enumerate(layer):         
                func_out, str_to_minimize, str_from_backup, deviation, outliear_score, outlier, is_learned, fixed_functional = conv(X, info=info, log_at_this_iter=log_at_this_iter_flag)
                #str_to_minimize - structural loss
                #str_from_backup - structural loss from backup component (should be used for masking i.e. the weighter sum for the layer output)
                #deviation - str_to_minimize - current running stats. mean
                #outliear_score - z_score
                #outlier - 'True' if outliear_score > deviation_threshold (if module is already fixed then False)
                #is_learned - 'True' if fixed
                #fixed_functional - 'True' if functional component is fixed
                z_scores_layer.append(outliear_score)
                module_learned_layer.append(is_learned) 
                fixed_functional_layer.append(fixed_functional) 
                outliers_layer.append(outlier)
                X_layer.append(func_out)
                str_layer.append(str_to_minimize) 
                str_from_backup_detached_layer.append(str_from_backup.detach())
                str_from_backup_layer.append(str_from_backup)
                deviation_layer.append(deviation)

            assert len(X_layer)>0
            # end1 = time.time()    
            str_layer = torch.stack(str_layer) # structural layer from the currently used structural component
            str_layer_from_backup_layer = torch.stack(str_from_backup_layer) # structural loss from backup (ignore if the backup mechanism is not being used)
            str_layer_for_masks_layer = torch.stack(str_from_backup_detached_layer) # detached tructural loss for masking

            deviation_layer = torch.stack(deviation_layer) if deviation_layer[0] is not None else torch.zeros_like(str_layer)
            module_learned_layer = torch.tensor(module_learned_layer).to(device)
            fixed_functional_layer = torch.tensor(fixed_functional_layer).to(device)
            
            #############################################
            outliers_layer = torch.stack(outliers_layer)          
            z_scores_layer = torch.stack(z_scores_layer)
            temp = temp * torch.ones(outliers_layer.size(1)).to(device) #is temp changes here?

            idx_old_modules = torch.where(module_learned_layer>0)
            idx_new_modules = torch.where(module_learned_layer==0)

            if self.catch_outliers_for_old_modules and not self.args.K_train:   
                if self.training and len(idx_new_modules[0])>0 and len(idx_old_modules[0])>0:   
                    #only consider free modules for outlier samples 
                    idx_outlier_old_modules = torch.where(outliers_layer[idx_old_modules].sum(0)==len(idx_old_modules[0]))[0] #outliers for all old modules          
                    if idx_outlier_old_modules.ndim>1:
                        yy = 0
                    if len(idx_new_modules) > 1:
                        yy = idx_new_modules
                        idx_new_modules = yy[-1]
                    str_layer_for_masks_layer[idx_new_modules,idx_outlier_old_modules]=1e-10
                    deviation_layer[idx_new_modules,idx_outlier_old_modules]=1e-10
            
            #############################################
            #TODO: if no new modules adeed to a layer, then dont fix funcitonal compnent of a module
            if isinstance(X_layer[0], Tuple):
                X_1 = torch.stack(list(map(lambda x: x[0], X_layer)))
                X_2 = torch.stack(list(map(lambda x: x[1], X_layer)))
                X=(X_1, X_2)
            else:
                X = torch.stack(X_layer)
                ######   fzy   ########
                # functional_weight = torch.stack(functional_weight_layer)
            return X, str_layer, str_layer_from_backup_layer, str_layer_for_masks_layer, deviation_layer, z_scores_layer, fixed_functional_layer, module_learned_layer

    def forward(self, X, task_id=None, temp=None, inner_loop=False, decoder = None, record_running_stats=False, env_assignments=None, info='', detach_head=False,
                str_prior_temp=None, epoch=None, acc_pre_task = 0, acc_pre_test = 0, hnet_out_list = [], calc_reg = False, modules_selected = [],add_new_module_num = 0,**kwargs):
        #start = time.time() 
        if env_assignments is not None:
            raise NotImplementedError
        if self.encoder is not None:      
            X = self.encoder(X) 
        if self.args.module_type=='linear':
                X=X.view(X.size(0), -1)
        temp = self.args.temp     

        if str_prior_temp is not None:  
            str_prior_temp=str_prior_temp           
            if str_prior_temp<self.min_str_prior_temp.item(): #store minimal temp
                self.min_str_prior_temp = str_prior_temp.clone()#.to(device)
        else:
            str_prior_temp=self.args.str_prior_temp

        if not self.training:
            str_prior_temp = min(self.min_str_prior_temp.item(),str_prior_temp)
        
        if isinstance(X, Tuple):
            n=X[0].size(0)
            c = X[0].shape[1]     
        else:
            n = X.shape[0]
            c = X.shape[1]     
        
        beam_results = []   
        if self.args.concat=='beam' and self.args.beam_width:
            #will store the best beam
            beam_results.append(([],[],X))

        str_loss_total = torch.zeros(self.depth).to(device) 
        outlier_signal = []
        total_mask = []
        deviation_mask = []
        raw_mask = []
        # add_modules_at_leayer=set()
        fixed_modules_functional = []
        fixed_modules = []

        loss_reg_total= torch.zeros(self.depth).to(device)

        ghnet_MWR_loss_total = torch.zeros(self.depth).to(device)
        ################################

        ################################
        #Manage projection phase timing#
        if not self.training:     
            projection_phase_flag=True
            projection_phase_MWR_flag = True
        else:     
            projection_phase_flag=self.projection_phase
            projection_phase_MWR_flag = self.projection_phase_MWR
            if (projection_phase_flag == True) and (projection_phase_MWR_flag == False):
                self._steps_since_last_addition=self._steps_since_last_addition+1
            elif (projection_phase_flag == False) and (projection_phase_MWR_flag == True):
                self._steps_for_ghnet_MWR=self._steps_for_ghnet_MWR+1
            elif (projection_phase_flag ==False) and (projection_phase_MWR_flag == False):
                fzy = 0
            else:
                fzy = 1
        ###############################        
        for l, layer in enumerate(self.components):                     
            fw_layer = self.forward_layer(layer, X, beam_results, temp, info=info, log_at_this_iter_flag=(self.log_at_this_iter_flag and not inner_loop))
            X_tmp, str_loss, str_loss_from_backup, str_loss_detached_for_mask, deviation_layer, z_scores_layer, fixed_modules_function_layer, fixed_module = fw_layer
            fixed_modules_functional.append(fixed_modules_function_layer)
            fixed_modules.append(fixed_module)
            
            deviation_layer = torch.abs(deviation_layer)

            ##########################################################################################
            # Calculate gate_output for the layer
            ##########################################################################################
            for g, gate_layer in enumerate(self.gate_net):
                if l == g:
                    weights_of_weighted_sum = gate_layer.PoolGateNet.forward(task_id=task_id)
                    weights = weights_of_weighted_sum[0].view(1, -1)
                    ind = self.n_modules_at_layer(l)
                    W = weights[0][0:ind]
                    # W = W.view(*s)
                    gate_output = W
                    gate_output = gate_output.unsqueeze(-1)


            ###################################################################
            ########   Calculate X_tmp selected and selected mask for weighting thew modules' outputs    #######
            #  First, select index of functional components
            if not self.args.use_structural_for_outlier and not self.args.use_structural:
                if self.args.use_ghnet:
                    if self.args.strategy_of_stacking_X==1:
                        if self.args.softmax_ghnet:
                            mask_tmp = self.using_hard_selection(gate_out = gate_output)
                        else:
                            mask_tmp = gate_output
                            # mask_tmp = mask_tmp.view(mask_tmp.size(1), mask_tmp.size(0))
                    mask_for_prior = gate_output.detach()


            ######################################################################
            ##     Move str_prior's calculation to the calculate_mask function    ##
            ########   fzy_hnet   ########
            if self.args.str_prior_factor>0:
                #bias module selection towards majority in the batch
                str_prior = torch.softmax(-torch.log(mask_for_prior.mean(1))/str_prior_temp, dim=0) #torch.softmax(-torch.log(str_loss_detached_for_mask.mean(1))/self.args.str_prior_temp, dim=0)
            else:
                str_prior = torch.softmax(torch.ones_like(deviation_layer.mean(1)).to(deviation_layer.device), dim=0)
            self.store_each_layer_str_prior[l] = str_prior
            ###################################################################
            # Manage projection phase and module addition #
            added=False
            if self.args.use_structural_for_outlier:
                if self.args.automated_module_addition and self.training:
                    if not projection_phase_flag:
                        # ====================================
                        # unfreeze functional for modules that were frozen in the previous projection phase
                        if len(self.modules_to_unfreeze_func) > 0:
                            for m in self.modules_to_unfreeze_func:
                                m.unfreeze_functional()
                                m.args.detach_structural = True
                            self.modules_to_unfreeze_func = []
                            # self.optimizer, _ = self.get_optimizers()
                            self.optimizer, _ = self.get_optimizers()
                        # ====================================
                        if self.args.concat == 'beam':
                            module_indicies = np.tile(np.arange(len(self.components[l])), len(beam_results))
                            most_likely_module = module_indicies[str_prior.argmax().item()]
                        else:
                            most_likely_module = str_prior.argmax().item()
                        added, params_changed, new_m_idx = self.maybe_add_modules(z_scores_layer.mean(1), fixed_module,
                                                                                  layer=l, bottom_up=True,
                                                                                  module_i=most_likely_module,
                                                                                  init_stats=self.args.init_stats)
                        # if new module was added start the projection phase
                        if added or (self.args.treat_unfreezing_as_addition and params_changed):
                            if not self.args.no_projection_phase:
                                ##########################
                                # Start projection phase
                                # - let the gradients from structural losses from layers above flow
                                ##########################
                                # current layer
                                for ni in new_m_idx:
                                    # the current layers structural components are detached (no gradient flows)
                                    layer[ni].args.detach_structural = True
                                    # layers below
                                if self.args.fix_layers_below_on_addition:
                                    for other_layer in self.components[:l]:
                                        for module_below in other_layer:
                                            if module_below.module_learned:
                                                frozen = module_below.freeze_functional()
                                                if frozen:
                                                    # if it was not frozen before, will be unfrozen after the projection phase
                                                    self.modules_to_unfreeze_func.append(module_below)

                                                    # module_below.args.detach_structural=True
                                # layers above
                                for other_layer in self.components[l + 1:]:
                                    for module_above in other_layer:
                                        # other_c._reset_backups()
                                        frozen = module_above.freeze_functional()
                                        if frozen:
                                            # if it was not frozen before, will be unfrozen after the projection phase
                                            self.modules_to_unfreeze_func.append(module_above)
                                            # let the gradient flow from structural
                                        module_above.args.detach_structural = False
                            # no module addition for the next self.args.projection_phase_length iterations
                            self._steps_since_last_addition = self._steps_since_last_addition * 0.
                            projection_phase_flag = True
                        if params_changed:
                            # self.optimizer, self.optimizer_structure = self.get_optimizers()
                            self.optimizer, self.optimizer_structure = self.get_optimizers()
            else:
                if self.args.automated_module_addition and self.training:
                    if (not projection_phase_flag) and (not projection_phase_MWR_flag):
                        if self.args.concat == 'beam':
                            module_indicies = np.tile(np.arange(len(self.components[l])), len(beam_results))
                            most_likely_module = module_indicies[str_prior.argmax().item()]
                        else:
                            most_likely_module = str_prior.argmax().item()
                        # self.store_each_layer_str_prior.insert(l, most_likely_module)
                        #################################### ####
                        #Modify the addition mechanism when using hnet
                        #    and calculating outlier without using structural components
                        added, params_changed, new_m_idx, layer_index,add_new_num = self.maybe_add_modules_without_structural(init_stats=self.args.init_stats,
                                                                            epoch=epoch, acc_pre_task = acc_pre_task, acc_test_pre = acc_pre_test, add_new_module_num = add_new_module_num,task_id=task_id)
                        #Modify the output layer structure of gate when you add a new module
                        add_new_module_num = add_new_num
                        # if new module was added start the projection phase
                        if added or (self.args.treat_unfreezing_as_addition and params_changed):
                            if not self.args.no_projection_phase:
                                ##########################
                                # Start projection phase
                                # - let the gradients from structural losses from layers above flow
                                ##########################
                                # current layer
                                for ni in new_m_idx:
                                    # the current layers structural components are detached (no gradient flows)
                                    layer[ni].args.detach_structural = True
                                    # layers below
                                if self.args.fix_layers_below_on_addition:
                                    for other_layer in self.components[:layer_index]:
                                        for module_below in other_layer:
                                            # if module_below.module_learned:
                                                frozen = module_below.freeze_functional()
                                                if frozen:
                                                    # if it was not frozen before, will be unfrozen after the projection phase
                                                    self.modules_to_unfreeze_func.append(module_below)
                                                module_below.args.detach_structural = False
                                                    # module_below.args.detach_structural=True
                                # layers above
                                else:
                                    for other_layer in self.components[layer_index + 1:]:
                                        for module_above in other_layer:
                                            # other_c._reset_backups()
                                            frozen = module_above.freeze_functional()
                                            if frozen:
                                                # if it was not frozen before, will be unfrozen after the projection phase
                                                self.modules_to_unfreeze_func.append(module_above)
                                                # let the gradient flow from structural
                                            module_above.args.detach_structural = False
                            # no module addition for the next self.args.projection_phase_length iterations
                            self._steps_since_last_addition = self._steps_since_last_addition * 0.
                            projection_phase_flag = True
                        if params_changed:
                            # self.optimizer, self.optimizer_structure = self.get_optimizers()
                            self.optimizer, self.optimizer_structure = self.get_optimizers()

            ###################################################################

            #Calculate the MWR loss of gate_hnet
            if self.args.using_MWR and task_id != 0 and not self.args.train_ghnet_MWR:
                MWR_loss = gate_layer.penalty(task_id)
                ghnet_MWR_loss_total[l]= self.args.MWR_lambda * MWR_loss

            ###################################################################

            ###################################################################
            # Calculate mask for weighting thew modules' outputs
            #####################################################
            if self.args.use_structural_for_outlier and self.args.use_structural:
                likelihood = torch.softmax(-torch.log((str_loss_detached_for_mask))/temp, dim=0)
                mask_unnorm = likelihood * str_prior.unsqueeze(0).repeat(n,1).T
                mask = (mask_unnorm)/mask_unnorm.sum(0)  # torch.softmax(torch.log(mask_unnorm), dim=0)
            else:
                #Probability distribution mode
                if self.args.probability_distribution_mode == 1:
                    likelihood = torch.softmax(mask_tmp)
                    likelihood = torch.softmax(-torch.log((mask_tmp))/temp, dim=0)
                    # mask = torch.softmax( torch.log(likelihood) + self.args.str_prior_factor * torch.log(str_prior.unsqueeze(0).repeat(n,1).T), dim=0)
                    mask_unnorm = likelihood * str_prior.unsqueeze(0).repeat(n,1).T
                    mask = (mask_unnorm)/mask_unnorm.sum(0)  # torch.softmax(torch.log(mask_unnorm), dim=0)
                elif self.args.probability_distribution_mode == 2:
                    if not torch.any(torch.isnan(mask_tmp)):
                        mask = mask_tmp/mask_tmp.sum(0)
                elif self.args.probability_distribution_mode == 3:
                    if not torch.any(torch.isnan(mask_tmp)):
                        mask_tmp = mask_tmp * str_prior.unsqueeze(0).repeat(n,1).T
                        mask = mask_tmp/mask_tmp.sum(0)
                elif self.args.probability_distribution_mode == 0:
                    if not torch.any(torch.isnan(mask_tmp)):
                        mask = mask_tmp
                elif self.args.probability_distribution_mode == 4:
                    if not torch.any(torch.isnan(mask_tmp)):
                         a = -torch.log((mask_tmp))/temp
                         mask = torch.softmax(-torch.log((mask_tmp))/temp, dim=0)


            ####################################
            #this block is related to debugging and is mostly unused
            if env_assignments is None:   
                if self.args.K_train==1 and self.training and not layer.all_modules_learned and self.args.regime=='meta':        
                    #mask will be 1 x depth, need to transform it into n_modules x depth
                    X_tmp_= torch.zeros(len(layer),*X_tmp.shape[1:]).to(device)
                    X_tmp_[fixed_modules_function_layer==0]=X_tmp
                    X_tmp=X_tmp_
                    mask_ = torch.zeros((int(self.n_modules[l]), n)).to(device)
                    mask_[fixed_modules_function_layer==0]=mask
                    mask=mask_
                    # fixed_modules_function_layer=fixed_modules_function_layer[fixed_modules_function_layer==0]
                elif (self.args.K_test==1 and not self.training) or (self.args.regime=='normal' and self.training and self.args.K_train==1):
                    #select only module with max activation at test time
                    idx = mask.max(0).indices
                    mask = torch.zeros_like(mask).to(device)
                    mask[idx,torch.arange(n)] = 1         
            elif not self.training and env_assignments is not None:
                mask = torch.zeros_like(mask).to(device)
                _env_as = env_assignments.clone() #copy.deepcopy(env_assignments)          
                _env_as[_env_as>(self.n_modules[l].item()-1)]=int(self.n_modules[l].item()-1)
                _env_as=np.array([_env_as.cpu().numpy(),np.arange(n)])
                mask[_env_as]=1
                del _env_as
            ####################################
            if mask is not None and not torch.any(torch.isnan(mask)):
                #mask for playback and drawing
                mask_for_playback_and_drawing = mask.clone()
                mask_for_playback_and_drawing = mask_for_playback_and_drawing.detach()
                total_mask.append(mask_for_playback_and_drawing)
                raw_mask.append(mask_for_playback_and_drawing)

            
            ################################################################### 
            # Take care about updates of the running stats components        
            ###################################################################
            if self.args.concat=='beam':      
                assert not self.args.module_oracle  
                components =  np.tile(self.components[l][:len(self.components[l])-int(added)],len(beam_results))  #list(itertools.chain.from_iterable(itertools.repeat(x, len(beam_results)) for x in self.components[l][:len(self.components[l])-int(added)]))
                assert len(components)==str_loss.size(0)==mask.size(0)==str_loss_from_backup.size(0)
                for m, v, v_backup, module, in zip(mask, str_loss.detach(), str_loss_from_backup, components):
                    module.update_stats(v, v_backup, m, record=True)       
            else:
                if self.training and not inner_loop and record_running_stats:
                    ################################################
                    # mostly used for debugging with module oracle
                    if task_id is not None and self.args.module_oracle:
                        if self.module_options.use_structural:
                            for m, str_l, str_l_backup, module in zip(mask, str_loss.detach(), str_loss_from_backup, [self.components[l][-1]]):
                                module.update_stats(str_l, str_l_backup, m)
                        else:
                            for m, module in zip(mask, [self.components[l][-1]]):
                                module.update_stats_without_structural(m)
                    ################################################
                    else:
                        if self.module_options.use_structural:
                            for m, str_l, str_l_backup, module in zip(mask, str_loss.detach(), str_loss_from_backup, [self.components[l][-1]]):
                                module.update_stats(str_l, str_l_backup, m)
                        else:
                            for m, module in zip(mask, [self.components[l][-1]]):
                                module.update_stats_without_structural(m)     ##hnet

            ################################################################### 
            deviation_mask.append(deviation_layer)    
            outlier_signal.append(z_scores_layer)     
            ##########################################################################################
            # Calculate structural losses for the layer
            ##########################################################################################
            if not inner_loop:      
                if torch.sum(fixed_modules_function_layer) != len(fixed_modules_function_layer) or self.args.regime=='normal':      
                    if self.args.optmize_structure_only_free_modules and not projection_phase_flag:   
                        idx_free = torch.lt(fixed_modules_function_layer,1)
                        str_act = (str_loss[idx_free]) if torch.sum(idx_free)>0 else torch.zeros_like(str_loss)
                        if self.args.mask_str_loss: 
                            if torch.sum(idx_free)>0:
                                str_act*=F.normalize(mask[torch.lt(fixed_modules_function_layer,1)],dim=0, p=1)
                        str_act=str_act.mean(1).mean(0)
                    else:
                        str_act = str_loss   
                        if self.args.mask_str_loss: 
                            str_act*=mask
                        str_act=str_act.mean(1).mean(0)    

                    if str_loss_from_backup.requires_grad and self.args.regime=='normal':
                            str_act += (str_loss_from_backup*mask).mean(1).mean(0)
                    str_loss_total[l]+=str_act
            else:
                # in the inner loop of continual meta CL
                str_act = str_loss
                if self.args.mask_str_loss: 
                    str_act*=mask
                str_loss_total[l]+=str_act.mean(1).mean(0)
            ##########################################################################################

            #############################################
            # Calculate the layer output using structural components
            #############################################
            if self.args.concat=='sum':
                # if torch.sum(torch.isnan(mask_hnet))>0:
                if torch.sum(torch.isnan(mask))>0:
                    X = X_tmp.sum(0)
                else:
                    # X = torch.einsum("ijklm,ij->jklm", X_tmp, mask)
                    if isinstance(X_tmp, Tuple):
                        X_1 = (X_tmp[0] * mask.view(mask.size(0), mask.size(1), *[1]*len(X_tmp[0].shape[2:]))).sum(0)
                        X_2 = (X_tmp[1] * mask.view(mask.size(0), mask.size(1), *[1]*len(X_tmp[1].shape[2:]))).sum(0)
                        X = (X_1, X_2)
                    else:
                        if modules_selected != []:
                            #The fill element of mask
                            fill_element_of_mask = modules_selected[l]
                            mask = torch.Tensor(self.n_modules_at_layer(l),n).to(device)
                            for i in range(self.n_modules_at_layer(l)):
                                if i < len(fill_element_of_mask):
                                    mask[i,  : ] = fill_element_of_mask[i]
                                else:
                                    mask[i,  : ] = 0

                        mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        X = torch.mul(X_tmp, mask)
                        X = X.sum(0)

            elif self.args.concat=='beam':
                beam_results, str_prior = self.update_beam_results(str_prior, X_tmp, beam_results, l, added)
            else:
                raise NotImplementedError
            #############################################
            # ############################
        X = self.avgpool(X)
        if self.args.multihead!='gated_conv':
            if isinstance(X, Tuple):
                X=X[0]
            X = X.reshape(n, -1)

        if not self.training and self.args.regime!='meta':
            total_mask=torch.stack(create_mask(total_mask, 0))
   
        str_loss_total = torch.sum(str_loss_total)
        ghnet_MWR_loss_total = torch.sum(ghnet_MWR_loss_total[l])
        decoder_idx=None
        fixed_modules = torch.stack(match_dims(fixed_modules)).T

        loss_reg_total = torch.sum(loss_reg_total)



        #############################################
        # Classification head
        #############################################
        if detach_head:
            X=X.detach()        
        if self.args.multihead=='none':
            #single head
            logit=self.decoder(X)         
        elif 'gated' in self.args.multihead:    
            # automatic task selection
            if task_id is not None and task_id<=(len(self.decoder)-1):
                # task id is given
                if self.args.concat=='beam':
                    idx = str_prior.argmax()
                    structure=beam_results[idx][0]
                    self.on_beam_path_selected(structure, beam_results[idx][1])
                    X=beam_results[idx][-1]
                    X = X.reshape(n, -1)
                    total_mask = torch.zeros(self.depth,max([len(self.components[k]) for k in range(len(self.components))]))
                    for m, z in zip(total_mask, structure):
                        m[z]=1
                    # total_mask=total_mask.T
                logit, str_act = self.decoder[task_id](X)
                if self.training:
                    ###   hnet
                    # Detects the existence of NaN
                    nan_mask = torch.isnan(str_loss_total)
                    # Replace NaN with another value (such as 0)
                    str_loss_total = str_loss_total.masked_fill(nan_mask, 0)
                    str_loss_total+=str_act.mean()
            else:
                if self.args.concat=='beam':
                    logit, best_str_act, decoder_idx, structure=None, None, None,None
                    for (struct,dec,X) in beam_results:
                        for d, decoder in enumerate(self.decoder):
                            logit_d, str_act = decoder(X)
                            if best_str_act is None or str_act.mean()<best_str_act:
                                best_str_act=str_act.mean()
                                decoder_idx=d
                                logit=logit_d
                                structure=struct
                                deicsions=dec
                    self.on_beam_path_selected(structure, deicsions)
                    total_mask = torch.zeros(self.depth,max([len(self.components[k]) for k in range(len(self.components))]))
                    for m, z in zip(total_mask, structure):
                        m[z]=1
                    # total_mask=total_mask.T
                    
                else:
                    #dont use task label, instead, select most lilely head
                    assert not self.training
                    logit, best_str_act, decoder_idx=None, None, None
                    for d, decoder in enumerate(self.decoder):
                        logit_d, str_act = decoder(X)
                        if best_str_act is None:
                            best_str_act=str_act.mean()
                            decoder_idx=d
                            logit=logit_d
                        elif str_act.mean()<best_str_act: 
                            best_str_act=str_act.mean()
                            decoder_idx=d
                            logit=logit_d
        elif self.args.multihead=='usual':
            #usual multihead
            if self.args.concat=='beam':
                idx = str_prior.argmax()
                structure=beam_results[idx][0]
                self.on_beam_path_selected(structure, beam_results[idx][1])
                X=beam_results[idx][-1]
                X = X.reshape(n, -1)
                total_mask = torch.zeros(self.depth,max([len(self.components[k]) for k in range(len(self.components))]))
                for m, z in zip(total_mask, structure):
                    m[z]=1
                # total_mask=total_mask.T
            if task_id is not None and task_id<=(len(self.decoder)-1):
                logit= self.decoder[task_id](X)
            else:
                logit=self.decoder[-1](X)  
        elif self.args.multihead=='modulewise':
            logit=X   
        #############################################

        if self.training and self.args.regime=='normal':
            self.handle_iteration_finished(True)

        return self.forward_template(mask=total_mask, mask_bf_act=raw_mask, hidden=X, ssl_pred=self.ssl_pred(X), logit=logit, regularizer=str_loss_total,
                                     info={'deviation_mask':deviation_mask , 'outlier_signal': outlier_signal, 'fixed_modules': fixed_modules,
                                           'selected_decoder': decoder_idx}),  loss_reg_total, add_new_module_num, ghnet_MWR_loss_total        #, 'modules_lr': modules_lr}) 'add_modules_layers': add_modules_at_leayer,

    def update_beam_results(self, str_prior, X_temp, beam_results, l, added): 
        # added is True is a module was added at this layer          
        if len(beam_results)==1 and (len(self.components[l])-int(added))==1:
            beam_width=1
        else:          
            beam_width=self.args.beam_width
        beam_result_idx = torch.topk(str_prior, min(beam_width, str_prior.size(0)), dim=0)  
        if len(beam_results)==1:
            module_idx = np.tile(np.arange(str_prior.size(0)//len(beam_results)),len(self.components[l])-int(added)) #e.g. array([0, 1, 2, 0, 1, 2, 0, 1, 2]) - indicies or modules from where input came
            beam_path_idx = np.arange(len(beam_results)).repeat(len(module_idx))
        elif len(self.components[l])-int(added)==1:
            beam_path_idx = np.arange(len(beam_results)).repeat(len(self.components[l])-int(added))
            module_idx = np.tile(np.arange(str_prior.size(0)//len(beam_results)),len(beam_path_idx))
        else:
            module_idx = np.tile(np.arange(str_prior.size(0)//len(beam_results)),len(self.components[l])-int(added))
            beam_path_idx = np.arange(len(beam_results)).repeat(len(self.components[l])-int(added))
        # output decision gives [(structure (selected modules per layer), input decision, i.e. module from which the input comes (is recodred in order to later update the running stats of the modules), selected X_temp) ,...]
        # input decision is always 0 at the beggining (right most) as there is only one input 
        beam_result = [(beam_results[beam_path_idx[k]][0]+[module_idx[k]], beam_results[beam_path_idx[k]][1]+[beam_path_idx[k]], X_temp[k]) for k in beam_result_idx[1]]
        return beam_result, beam_result_idx[0]
    
    def on_beam_path_selected(self, structure, decisions):
        for l, layer in enumerate(self.components):
            for i, c in enumerate(layer):
                if structure[l]==i:          
                    c.update_running_stats_from_record(decisions[l])
                else:
                    c.clean_runing_stats_record()

    def handle_iteration_finished(self, *args, **kwargs):
        return self.handle_outer_loop_finished(*args, **kwargs)

    def handle_outer_loop_finished(self, finished_outer_loop: bool):
        for l in self.components:
            for c in l:
                c.on_finished_outer_loop(finished=finished_outer_loop)
    def on_before_zero_grad(self):
        pass
     
    def load_state_dict(self, state_dict: Union[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], strict: bool=True):
        #TODO: test this
        k=None
        if '_n_modules' in state_dict:
            k='_n_modules'
        elif 'n_modules' in state_dict:
            k='n_modules'
        if k is not None:                    
            for l, n_modules_in_layer in enumerate(state_dict[k]):
                # print(int(self._n_modules[l].item()))
                for _ in range(int(n_modules_in_layer.item()) - int(self._n_modules[l].item())):
                    self.add_modules(at_layer=l, verbose=False)
        components = filter( lambda x: '_backup' in x, state_dict.keys())
        for m in list(components):
            layer, comp = [int(i) for i in m if i.isdigit()][:2]
            if 'functional' in m:
                self.components[layer][comp]._maybe_backup_functional(verbose=False)
            if 'inv' in m:
                self.components[layer][comp]._maybe_backup_structural(verbose=False)  
        if self.args.multihead=='modulewise':      
            if any(map(lambda x: 'decoder' in x and not 'components' in x, state_dict.keys())): #if state dict contains decoder
                classifier_dict = {k: v for k, v in state_dict.items() if 'decoder' in k and not 'components' in k}
                for l in self.components:
                    for c in l:
                        if hasattr(c, 'num_classes'): 
                            print(f'Loading classifier into {c.name}')
                            if c.num_classes>0:
                                c.load_state_dict(classifier_dict, strict=False)
        return super().load_state_dict(state_dict,strict)

    def add_modules_without_structural(self, block_constructor=None, at_layer=None, strict=False, module_i=None, verbose=True,
                    init_stats=False):
        # if at_layer == None add to all layers
        if block_constructor is None:
            block_constructor = self.block_constructor
        for l, layer in enumerate(self.components):
            ################################################
            if l != at_layer and at_layer != None:
                if self.args.freeze_module_after_step_limit and self.args.reset_step_count_on_module_addition:
                    for c in layer:
                        if c.active:
                            # reset step count for other modules
                            c.reset_step_count()

            elif l == at_layer or at_layer == None:
                if verbose:
                    print(f'adding modules at layer {l}')
                self._n_modules[l] = self._n_modules[l] + 1
                ################################################
                # new module's params
                in_h = layer[0].in_h
                channels_in = layer[0].in_channels
                channels = layer[0].out_channels
                i_size = layer[0].i_size
                deeper = layer[0].deeper
                module_type = layer[0].module_type
                self.module_options.kernel_size = layer[0].args.kernel_size
                self.module_options.dropout = layer[0].args.dropout
                # out_h = layer[0].out_h
                ################################################
                init_params = None
                if self.args.module_init == 'identical':
                    init_params = self.per_layer_module_initialization[l]
                elif self.args.module_init == 'mean':
                    if verbose:
                        print('adding module with mean innitialization')
                    l_state_dict = layer.state_dict()
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, l_state_dict.keys()))
                        for k in keys_to_pop:
                            l_state_dict.pop(k, None)
                    init_params = copy.deepcopy(ordered_dict_mean(l_state_dict))
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)
                elif self.args.module_init == 'existing':
                    init_params = layer[0].state_dict()
                elif self.args.module_init == 'previously_active':
                    for c in layer:
                        if c.active:
                            init_params = c.state_dict()
                    assert init_params is not None, f'At least one module in the layer {l} should have been active'
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)
                elif self.args.module_init == 'most_likely' and module_i is not None:
                    # assert (str_prior is not None)
                    # module_i = str_prior.argmax().item()
                    init_params = layer[module_i].state_dict()
                    if not init_stats:
                        keys_to_pop = list(filter(lambda x: 'runing_activation' in x, init_params.keys()))
                        for k in keys_to_pop:
                            init_params.pop(k, None)
                ##########

                new_module = block_constructor(in_h, channels_in, channels, i_size,
                                               initial_inv_block_lr=self.lr_structural,
                                               name=f'components.{l}.{len(layer)}',
                                               module_type=module_type, deviation_threshold=self.deviation_threshold,
                                               freeze_module_after_step_limit=self.args.freeze_module_after_step_limit,
                                               deeper=deeper, options=self.module_options,
                                               num_classes=self.num_classes if (
                                                       self.args.multihead == 'modulewise' and l == self.depth - 1) else 0)
                if init_params is not None:
                    if verbose:
                        print('loading state dict new module')
                    new_module.load_state_dict(init_params, strict)

                new_module.to(device)
                new_module.reset()
                layer.append(new_module)
        self._reset_references_to_learnable_params()
        if self.args.regime == 'normal':
            # self.optimizer, self.optimizer_structure = self.get_optimizers()
            self.optimizer, self.optimizer_structure = self.get_optimizers()
            ########################


    def modify_parameter_shape(self, layer):
        #modify parameter shape of the mnet output
        for g, gate_layer in enumerate(self.gate_net):
            if layer == g:
                # gate_layer = self.gate_net(layer)
                num_modules_layer = int(self._n_modules[layer].item())
                modify_shape = False
                modify_hyper_shapes = self.modify_mnet_hyper_shapes(gate_layer,num_modules_layer)
                modify_hyper_shapes_learned = self.modify_mnet_hyper_shapes_learned(gate_layer,num_modules_layer)
                modify_param_shapes = self.modify_mnet_param_shapes(gate_layer,num_modules_layer)
                 # modify parameter shape of the hnet output
                self.modify_hnet_target_shapes(gate_layer,num_modules_layer)
                if modify_hyper_shapes and modify_hyper_shapes_learned and modify_param_shapes:
                    modify_shape = True

        return modify_shape

    def modify_mnet_param_shapes(self, gate_layer, num_modules_layer):
        modify_shape = False
        if hasattr(gate_layer.mnet, 'mlp'):
            gate_layer.mnet.mlp.param_shapes.pop()
            original_weight = gate_layer.mnet.mlp.param_shapes[-1]
            arch_original_weight = original_weight[-1]
            gate_layer.mnet.mlp.param_shapes.pop()
            new_weight = [num_modules_layer, arch_original_weight]
            new_bias = [num_modules_layer]
            gate_layer.mnet.mlp.param_shapes.append(new_weight)
            gate_layer.mnet.mlp.param_shapes.append(new_bias)
            modify_shape = True
        else:
            gate_layer.mnet.resnet.param_shapes.pop()
            original_weight = gate_layer.mnet.resnet.param_shapes[-1]
            arch_original_weight = original_weight[-1]
            gate_layer.mnet.resnet.param_shapes.pop()
            new_weight = [num_modules_layer, arch_original_weight]
            new_bias = [num_modules_layer]
            gate_layer.mnet.resnet.param_shapes.append(new_weight)
            gate_layer.mnet.resnet.param_shapes.append(new_bias)
            modify_shape = True
        return modify_shape

    def modify_mnet_hyper_shapes_learned(self, gate_layer, num_modules_layer):
        modify_shape = False
        if hasattr(gate_layer.mnet, 'mlp'):
            gate_layer.mnet.mlp.hyper_shapes_learned.pop()
            original_weight = gate_layer.mnet.mlp.hyper_shapes_learned[-1]
            arch_original_weight = original_weight[-1]
            gate_layer.mnet.mlp.hyper_shapes_learned.pop()
            new_weight = [num_modules_layer, arch_original_weight]
            new_bias = [num_modules_layer]
            gate_layer.mnet.mlp.hyper_shapes_learned.append(new_weight)
            gate_layer.mnet.mlp.hyper_shapes_learned.append(new_bias)
            modify_shape = True
        else:
            gate_layer.mnet.resnet.hyper_shapes_learned.pop()
            original_weight = gate_layer.mnet.resnet.hyper_shapes_learned[-1]
            arch_original_weight = original_weight[-1]
            gate_layer.mnet.resnet.hyper_shapes_learned.pop()
            new_weight = [num_modules_layer, arch_original_weight]
            new_bias = [num_modules_layer]
            gate_layer.mnet.resnet.hyper_shapes_learned.append(new_weight)
            gate_layer.mnet.resnet.hyper_shapes_learned.append(new_bias)
            modify_shape = True
        return modify_shape

    def modify_mnet_hyper_shapes(self, gate_layer, num_modules_layer):
        modify_shape = False
        if hasattr(gate_layer.mnet, 'mlp'):
            gate_layer.mnet.mlp.hyper_shapes.pop()
            original_weight = gate_layer.mnet.mlp.hyper_shapes[-1]
            arch_original_weight = original_weight[-1]
            gate_layer.mnet.mlp.hyper_shapes.pop()
            new_weight = [num_modules_layer, arch_original_weight]
            new_bias = [num_modules_layer]
            gate_layer.mnet.mlp.hyper_shapes.append(new_weight)
            gate_layer.mnet.mlp.hyper_shapes.append(new_bias)
            modify_shape = True
        else:
            gate_layer.mnet.resnet.hyper_shapes.pop()
            original_weight = gate_layer.mnet.resnet.hyper_shapes[-1]
            arch_original_weight = original_weight[-1]
            gate_layer.mnet.resnet.hyper_shapes.pop()
            new_weight = [num_modules_layer, arch_original_weight]
            new_bias = [num_modules_layer]
            gate_layer.mnet.resnet.hyper_shapes.append(new_weight)
            gate_layer.mnet.resnet.hyper_shapes.append(new_bias)
            modify_shape = True
        return modify_shape

    def modify_hnet_target_shapes(self, gate_layer, num_modules_layer):
        gate_layer.hnet.target_shapes.pop()
        original_weight = gate_layer.hnet.target_shapes[-1]
        arch_original_weight = original_weight[-1]
        gate_layer.hnet.target_shapes.pop()
        new_weight = [num_modules_layer, arch_original_weight]
        new_bias = [num_modules_layer]
        gate_layer.hnet.target_shapes.append(new_weight)
        gate_layer.hnet.target_shapes.append(new_bias)


    def get_top_indices(self, out_indices= None, k_top = 1):
        num_element = out_indices.numel()
        if num_element>1:
            #If the dimension of out_indices is greater than 1,
            # the values of out_indices are sorted and the index of the first k values is selected
            values, indices = torch.topk(out_indices, k_top, largest=True)
            return indices.tolist()
        else:
            indices = [0]
            return indices

    def get_top_functional_weight(self, dict_functional_weight, k_top):

        sorted_dict_functional_weight = sorted(dict_functional_weight.items(), key=lambda x: x[1], reverse=True)

        top_k_sorted_dict_functional_weight = {key: value for key, value in sorted_dict_functional_weight[:k_top]}

        top_k_sorted_component_list = list(top_k_sorted_dict_functional_weight.key())

        return top_k_sorted_dict_functional_weight, top_k_sorted_component_list

    def select_X_tmp_calculate_mask(self, X_tmp=None, index_list = [],functional_component_weight = None, layer = 0, str_prior_temp =0.1, temp =0.1,n = 1):

        X_selected_list = []
        for i, index in enumerate(index_list):
            indices=torch.tensor(index).to(device)
            #Select index_select Select the element in dimension 0 that is specified as the index
            X_selected = self.select_X_tmp(X_tmp = X_tmp, indices = indices)
            X_selected_list.append(X_selected)
            selected_X_stack = torch.stack(X_selected_list)

        #Adds the selected elements
        if self.args.without_using_weighted_sums >0:
            #Weighted sums are not used
            selected_X_stack = selected_X_stack.sum(0)
            shape = X_tmp.shape[0]
            mask = torch.zeros(shape,64).to(device)         #64
            for i, index in enumerate(index_list):
                mask[index,  : ] = 1
            # X_try = ((X_tmp * mask.view(mask.size(0), mask.size(1), *[1]*len(X_tmp.shape[2:]))).sum(0)).to(device)
            # X = selected_X_stack
        else:
            selected_functioanl_weight=[]
            shape = X_tmp.shape[0]
            mask_for_selected_functioanl_weight = torch.zeros(shape,64)       # 64   torch.zeros(shape,64)
            for i, index in enumerate(index_list):
                indices=torch.tensor(index).to(device)
                functional_component_weight_selected = self.select_functional_component_weight(functional_component_weight=functional_component_weight, indices = indices)
                selected_functioanl_weight.append(functional_component_weight_selected)
                mask_for_selected_functioanl_weight[index,  : ] = functional_component_weight_selected
            selected_functioanl_weight_stack = torch.stack(selected_functioanl_weight)
            if self.args.str_prior_factor>0:
                #bias module selection towards majority in the batch
                mask_fzy = self.calculate_mask(selected_functioanl_weight_stack = selected_functioanl_weight_stack, str_prior_temp = str_prior_temp, temp = temp, n=n)
                mask = self.calculate_mask(selected_functioanl_weight_stack = mask_for_selected_functioanl_weight, str_prior_temp = str_prior_temp, temp = temp, n=n)
            else:
                likelihood = torch.softmax(-torch.log((functional_component_weight))/temp, dim=0)
                mask = (likelihood)/likelihood.sum(0)
            # X_try = (selected_X_stack * mask_fzy.view(mask.size(0), mask.size(1), *[1]*len(X_tmp.shape[2:]))).sum(0)
            # X = (X_tmp * mask.view(mask.size(0), mask.size(1), *[1]*len(X_tmp.shape[2:]))).sum(0)

        return mask

    def select_X_tmp(self, X_tmp = None, indices = None):
        selected_Xtmp = torch.index_select(X_tmp, 0, indices)
        selected_Xtmp = selected_Xtmp.sum(0)

        return selected_Xtmp

    def select_functional_component_weight(self,functional_component_weight=None, indices = None):
        # selected_functioanl_weight=[]
        selected_functional_component_weight_tmp = torch.index_select(functional_component_weight, 0, indices)
        selected_functional_component_weight_tmp = selected_functional_component_weight_tmp.sum(0)
        # selected_functioanl_weight.append(selected_functional_component_weight_tmp)
        return selected_functional_component_weight_tmp

    def calculate_mask(self,selected_functioanl_weight_stack = None,str_prior_temp=0.1, temp=0.1, n=1):
        str_prior = torch.softmax(-torch.log(selected_functioanl_weight_stack.mean(1))/str_prior_temp, dim=0)
        likelihood = torch.softmax(-torch.log((selected_functioanl_weight_stack))/temp, dim=0)
        mask_unnorm = likelihood * str_prior.unsqueeze(0).repeat(n,1).T
        mask = (mask_unnorm)/mask_unnorm.sum(0)
        return mask

    def reshape_weights(self, new_target_shapes,weights):
        ### Reshape weights dependent on the main networks architecture.
        ind = 0
        ret = []
        if new_target_shapes is None:
            target_shapes = self.target_shapes
        else:
            target_shapes = new_target_shapes
        for j, s in enumerate(target_shapes):
            num = int(np.prod(s))
            weight = weights[0]
            W = weight[ind:ind+num]
            ind += num
            W = W.view(*s)
            ret.append(W)
        return ret

    def max_index_of_min_duplicates(self, lst,add_new_module_num):
        element_dict = {}
        for index, element in enumerate(lst):
            if index in element_dict:
                if element > element_dict[index]:
                    element_dict[index] = element
            else:
                element_dict[index] = element
        sorted_key = sorted(element_dict, reverse=True)
        max_layer = sorted_key[add_new_module_num]
        return  max_layer

    def min_index_of_min_duplicates(self, lst,add_new_module_num):
        element_dict = {}
        for index, element in enumerate(lst):
            if index in element_dict:
                if element > element_dict[index]:
                    element_dict[index] = element
            else:
                element_dict[index] = element
        sorted_key = sorted(element_dict)
        min_layer = sorted_key[add_new_module_num]
        return  min_layer

    def using_hard_selection(self, gate_out = None):
        max_values, max_indices = torch.max(gate_out, dim=0)
        hard_select_gate_out = torch.zeros_like(gate_out)
        hard_select_gate_out[torch.arange(gate_out.shape[0]).view(-1, 1), max_indices] = 1

        return hard_select_gate_out
