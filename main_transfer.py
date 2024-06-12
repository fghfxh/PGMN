import copy
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional

import ctrl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from simple_parsing import ArgumentParser, choice
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor

import wandb
from Data.Utils import TensorDataset     
from Methods.models.cnn_independent_experts import ExpertMixture
from Methods.models.PGMN import PGMN_net
from Methods.replay import BalancedBuffer, Buffer
from Utils.ctrl.ctrl.tasks.task_generator import TaskGenerator
from Utils.logging_utils import log_wandb
from Utils.nngeometry.nngeometry.metrics import FIM
from Utils.nngeometry.nngeometry.object import PMatDiag, PVector
from Utils.nngeometry.nngeometry.object.pspace import PMatAbstract
from Utils.utils import construct_name_ctrl, cosine_rampdown, set_seed
from collections import Counter


device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(torch.cuda.is_available())
print(torch.__version__)

@dataclass#(eq=True, frozen=False)
class ArgsGenerator():       
    ##############################
    #learning process related
    MWR_lambda = 10
    using_MWR = False
    criterion_1 = 0.99
    criterion_3 = 0.8
    MWR_attempt_phase_length: int = 10
    projection_phase_length: int = 2
    fix_layers_below_on_addition: int = 1 #if 1 also layers below the layer where module is added are frozen during the projection phase
    deviation_threshold: float = 4
    reg_factor: float = 10
    temp: float = 0.1
    anneal: int = 0
    lr_structural: float = 0.001
    regime: str = choice('multitask','cl', default='cl') # multitask regime = ofline trainiogn on all tasks (use single head for it, i.e. multihead=none)
    module_init: str = choice('none','mean','previously_active', 'most_likely', default='mean') # new module initialization strategy
    mask_str_loss: int = 1
    structure_inv: str = choice('linear_no_act', 'pool_only_large_lin_no_act', 'linear_act', 'ae', default='ae')
    use_backup_system: int = 0
    use_backup_system_structural: int = 0
    running_stats_steps: int = 100
    str_prior_factor: float = 1.
    str_prior_temp: float = 0.1
    str_anneal: int = 0
    concat: str = choice('beam', 'sum', default='sum') # -
    beam_width: int = 1
    catch_outliers_old: bool = 1
    momentum_bn: float = 0.1
    track_running_stats_bn: int = 1
    keep_bn_in_eval_after_freeze: bool = 1
    detach_structural:bool = 1
    init_runingstats_on_addition: int = 1
    optmize_structure_only_free_modules: int = 1
    ################
    
    #model related
    hidden_size: int = 64
    module_type: str = 'conv'
    gating: str = choice('experts', 'locspec', default='locspec')     #
    num_modules: int = 1
    net_arch: int = choice('none', default='none') # -
    activation_structural: str = choice('sigmoid', 'relu', 'tanh', default='sigmoid')
    depth: int = 4
    use_bn: int = 1
    use_structural: int = 0
    ################
    
    #output module related (output leayer)
    multihead: str = choice('usual', 'gated_linear', 'gated_conv', 'none', default='gated_linear') #multihead type, if 'none' uses single head
    normalize_oh: bool = 1 # -
    projection_layer_oh: bool = 0 # -
    structure_inv_oh: str =choice('ae', 'linear_no_act', 'linear_act', default='linear_no_act') # -
    use_bn_decoder_oh: int = 0 # -
    activate_after_str_oh: int = 0 # -
    init_oh: str = choice('mean', 'none', 'most_likely', default='none') # -
    ################

    #unfreezing of modules
    active_unfreezing: int = 0
    unfreeze_structural: int = 0
    treat_unfreezing_as_addition: int = 0
    ################
    
    #########
    # ae          
    use_bn_decoder:int = 1 #wether to use batchnorm in the decoder of structural (ae)
    momentum_bn_decoder: float = 0.1 #momentum of the structural decoder
    activation_target_decoder: str = choice('sigmoid', 'relu', 'tanh', 'None', default='None') #activation for the decoders' target (output of previous layer)

    task_sequence_train: Optional[str]=None
    task_sequence_test: Optional[str]=None

    ##############################
    #Optimization
    wdecay: float = 0.001
    lr: float = 0.001
    ##############################
    #Logging
    pr_name: Optional[str]= 'pgmn'
    wand_notes: str = ''
    log_avv_acc: int = 0 # if 'True' calculates the average accuracy over tasks sofar after each task
    ##############################

    ##############################  
    #Data generation                             
    stream_seed: int = 180 # seed of the ctrl stream
    n_tasks: int = 6
    task_sequence: str = choice('s_minus', 's_pl','s_plus', 's_mnist_svhn', 's_pnp_comp', 's_pnp_tr', 's_pnp', 's_in', 's_out', 's_long', 's_long30', 's_ood', default='s_pl') #task sequence from ctrl
    batch_size: int = 64
    normalize_data: int=0
    ##############################
    #Hparams tuning & training
    regenerate_seed: int = 0
    n_runs:int = 1 # -
    seed:int = 178 #seed
    debug: int = 0 #debug mode
    early_stop_complete: bool = 0 # it 'True' resets best model to None every time a new module was added during learning a task
    warmup_bn_bf_training: int = 0 # -
    task_agnostic_test: int = 1 #if 'True' (1) no task_id is given at test time
    keep_best_model: int = 1 # if 'True' keeps bestvalidation model
    num_epochs_added_after_projection:int = 10 #least number training epochs run after a projection phase 
    epochs_str_only_after_addition: int = 0 # number of epochs after module addition during which only the structural loss is used
    epochs_structure_only_at_start: int = 0 # number of epochs during which only structural loss will be used ad the beginning of training on each task
    epochs: int = 105 # number of epochs to train on each task
    shuffle_test: int = 0 #if 'True', shuffls test and validation sets (might give better performance when using batchnorm warmup)
    ##############################
    #EWC
    ewc_online: bool = 0. # online modular network consolidates the FIMs
    ewc: float = 0. #if >0 EWC regularization is used
    ##############################
    #Replay
    replay_capacity: int = 0 #if > 0 uses replay buffer, if -1 calculates the replay size automatically to match the max size of the modular network in case of linear growth
    ##############################

    #ablation
    no_projection_phase:int = 0 #-
    save_figures: int = 0 #-
    n_heads_decoder: int = 1 #-
    ##############################
    #Gate net
    train_ghnet_MWR = False    #'True' mean using MWR in ghnet training process
    add_module_scheme = 2     ## Add modules Option 1: Add modules in layers with minimal modules
                                    # Add modules Option 2: Add modules from the top
                                    # Add modules Option 3: Add modules from the lowest level

    probability_distribution_mode = 2     #1 mean : Simply normalize
                                            #2 mean : Find the probability distribution using softmax and log
                                            #3 mean : Use str_prior in conjunction with normalization
                                            #0 mean : Use the output of the gate network directly
    gate_lr: float=0.001
    gate_dropout_rate = -1
    gate_arch=[100,10]     #[100,20] [100,50]
    hnet_arch=[100]
    train_from_scratch = False
    custom_network_init = True
    net_act_output_layer: str = choice('sigmoid', 'relu', 'elu', 'leakyrelu', 'linear', default='sigmoid')     # mlp: 'relu' sigmoid, resnet:None, zeken
    dropout_rate = -1
    specnorm = None
    batchnorm = None
    no_batchnorm = None
    bn_no_running_stats = False
    bn_distill_stats = False
    normal_init =False
    std_normal_init = 0.02
    std_normal_emb = 1.0
    std_normal_temb = 1.0

    no_weights = True
    no_bias = False
    hyper_chunks = [6]
    hnet_act: str = choice('sigmoid', 'relu', 'elu', 'leakyrelu' ,'linear', default='elu')

    temb_size = 64

    hnet_dropout_rate = -1

    temb_std = -1

    backprop_dt = False
    continue_emb_training = False
    train_embeddings = True
    dec_lr_emb = 0.001
    use_structural_for_outlier = False
    use_ghnet = True
    use_sgd_change = False
    hnet_beta = 1
    mask_hnet_loss = 1
    multilayer_PGMN= True
    length_layer_hnet = 4
    k_top = 2
    importance = 10000  #Adjustable parameter
    strategy_of_stacking_X: int = 1.     #The strategy of stacking X_tmp
                                            #strategy 1: Comparison of module repetitions
                                            #strategy 2: Comparison of module weights

    without_using_weighted_sums: float = 1.            #Weighted sums are not used，This parameter is used with strategy_of_stacking_X = 1

    softmax_ghnet = 0
    reg_factor_hnet: float = 10
    ##############################
    
    def __post_init__(self):   
        if self.task_sequence == 's_ood':
            self.task_sequence_train ='s_ood_train'
            self.task_sequence_test ='s_ood_test'
        else:
            self.task_sequence_train = self.task_sequence

        if not self.use_backup_system:
            self.use_backup_system_structural=0
        if self.debug:  
            self.epochs=1      
            self.regenerate_seed=0
            self.generate_args=0
            self.hidden_size=8         

    def generate_seed(self):
        self.seed=random.randint(1, 2021)

loss_function = nn.CrossEntropyLoss()

def create_dataloader_ctrl(task_gen:TaskGenerator, task, args:ArgsGenerator, split=0, 
                            batch_size=64, num_batches=None, labeled=True, normalize=False, **kwargs):
    single_head=(args.multihead=='none')
    normalize=args.normalize_data
    y = task.get_labels(split=split, prop=0)
    x = task.get_data(split=split)
    if labeled:
        idx = torch.where(y!=-1)
        y = y[idx]
        x = x[idx]
    if num_batches is not None:       
        batch_size=int(len(y)//num_batches)
    transform=None
    
    if x.shape[1]<task.x_dim[-1] and args.task_sequence=='s_mnist_svhn':
        transform = transforms.Compose([ transforms.ToPILImage(),transforms.Resize((task.x_dim[-1],task.x_dim[-1])), ToTensor()])
    if normalize:
        if min(task.statistics['mean'])>0 and 'mnist' in str(task.concepts) and 'ood' in args.task_sequence: 
            #if no dimention is completely zeros we use statistics of the complete MNIST dataset (for simplisity) - will be used for task sequence s_ood_bkgrnd_white_digits
            if transform is None:
                transform = transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
            else:
                transform.append(transforms.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081)))
        else:
            #we leave dimentions with only 0s to stay only 0s
            if transform is None:    
                transform = transforms.Normalize(task.statistics['mean'], [s if s>0 else s+1 for s in task.statistics['std']])
            else:
                transform.append(transforms.Normalize(task.statistics['mean'], [s if s>0 else s+1  for s in task.statistics['std']]))    

    if single_head:
        # adjust class labels for the single head regime
        adjust_y=0
        for t,old_t in enumerate(task_gen.task_pool):
            if str(old_t.concepts)==str(task.concepts):
                break
            else:
                adjust_y+=old_t.info()['n_classes'][0]           
        y+=adjust_y 
    if args.shuffle_test and split!=0:
        idx = torch.randperm(x.size(0))
        x=x[idx]
        y=y[idx]
    
    dataset = TensorDataset([x,y], transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split==0)) #or shuffle_test))

def init_model(args:ArgsGenerator, gating='locspec', n_classes=10, i_size=28):
    multihead=args.multihead
    from Methods import ModelOptions
    from Methods.models.PGMN import PGMN_net
    model_options = ModelOptions()
    #Gate net
    model_options.PGMN.importance=args.importance
    model_options.PGMN.train_ghnet_MWR=args.train_ghnet_MWR
    model_options.PGMN.MWR_lambda = args.MWR_lambda
    model_options.PGMN.using_MWR = args.using_MWR
    model_options.PGMN.criterion_1 = args.criterion_1
    model_options.PGMN.criterion_3 = args.criterion_3
    model_options.Gate.lr=args.gate_lr
    model_options.Gate.gate_dropout_rate = args.gate_dropout_rate
    model_options.Gate.gate_arch=args.gate_arch
    model_options.Gate.track_running_stats_bn=args.track_running_stats_bn
    model_options.Gate.maxpool_kernel=2
    model_options.Gate.padding=2
    model_options.Gate.momentum_bn=args.momentum_bn
    model_options.Gate.use_bn=args.use_bn
    model_options.Gate.train_from_scratch = args.train_from_scratch
    model_options.Gate.custom_network_init = args.custom_network_init
    model_options.Gate.net_act_output_layer = args.net_act_output_layer
    model_options.Gate.dropout_rate = args.dropout_rate
    model_options.Gate.specnorm = args.specnorm
    model_options.Gate.batchnorm = args.batchnorm
    model_options.Gate.no_batchnorm = args.no_batchnorm
    model_options.Gate.bn_no_running_stats = args.bn_no_running_stats
    model_options.Gate.bn_distill_stats = args.bn_distill_stats
    model_options.Gate.normal_init = args.normal_init
    model_options.Gate.std_normal_init = args.std_normal_init
    model_options.Gate.std_normal_emb = args.std_normal_emb
    model_options.Gate.std_normal_temb = args.std_normal_temb
    model_options.Gate.no_weights = args.no_weights
    model_options.Gate.no_bias = args.no_bias
    model_options.Gate.hyper_chunks = args.hyper_chunks
    model_options.Gate.hnet_arch = args.hnet_arch
    model_options.Gate.hnet_act = args.hnet_act
    model_options.Gate.hnet_dropout_rate = args.hnet_dropout_rate
    model_options.Gate.temb_std = args.temb_std
    model_options.Gate.backprop_dt = args.backprop_dt
    model_options.Gate.continue_emb_training = args.continue_emb_training
    model_options.Gate.use_sgd_change=args.use_sgd_change
    model_options.Gate.multilayer_PGMN = args.multilayer_PGMN
    model_options.Gate.length_layer_hnet = args.length_layer_hnet
    model_options.Module.use_backup_system=args.use_backup_system
    model_options.Module.structure_inv=args.structure_inv
    model_options.Module.maxpool_kernel=2
    model_options.Module.padding=2
    model_options.Module.use_bn=args.use_bn   
    model_options.Module.use_structural=args.use_structural
    model_options.Module.activation_structural=args.activation_structural       
    model_options.Module.use_backup_system_structural=args.use_backup_system_structural
    #ae
    model_options.Module.use_bn_decoder=args.use_bn_decoder
    model_options.Module.momentum_bn_decoder=args.momentum_bn_decoder
    model_options.Module.activation_target_decoder=args.activation_target_decoder

    model_options.Module.running_stats_steps= args.running_stats_steps if args.running_stats_steps>0 else 100
    model_options.Module.momentum_bn=args.momentum_bn
    model_options.Module.track_running_stats_bn=args.track_running_stats_bn
    model_options.Module.kernel_size = 3
    model_options.Module.keep_bn_in_eval_after_freeze=args.keep_bn_in_eval_after_freeze

    model_options.Module.normalize_oh=args.normalize_oh
    model_options.Module.projection_layer_oh=args.projection_layer_oh
    model_options.Module.structure_inv_oh = args.structure_inv_oh   
    model_options.Module.use_bn_decoder_oh = args.use_bn_decoder_oh   
    model_options.Module.activate_after_str_oh = args.activate_after_str_oh
    if gating=='locspec':
        model_options.PGMN.train_embeddings = args.train_embeddings
        model_options.PGMN.use_ghnet=args.use_ghnet
        model_options.PGMN.temb_size= args.temb_size
        # model_options.PGMN.emb_size = args.emb_size
        model_options.PGMN.dec_lr_emb = args.dec_lr_emb
        model_options.PGMN.batch_size = args.batch_size
        model_options.PGMN.hnet_beta = args.hnet_beta
        model_options.PGMN.mask_hnet_loss = args.mask_hnet_loss
        model_options.PGMN.add_module_scheme = args.add_module_scheme
        model_options.PGMN.k_top = args.k_top
        model_options.PGMN.use_structural = args.use_structural
        model_options.PGMN.strategy_of_stacking_X = args.strategy_of_stacking_X
        # model_options.PGMN.strategies_for_comparing_module_weights = args.strategies_for_comparing_module_weights
        model_options.PGMN.without_using_weighted_sums = args.without_using_weighted_sums
        model_options.Module.without_using_weighted_sums= args.without_using_weighted_sums
        model_options.PGMN.softmax_ghnet = args.softmax_ghnet
        model_options.PGMN.probability_distribution_mode =  args.probability_distribution_mode


        model_options.Module.detach_structural=args.detach_structural
        model_options.PGMN.no_projection_phase=args.no_projection_phase
        model_options.PGMN.init_stats=args.init_runingstats_on_addition
        model_options.PGMN.regime='normal'
        model_options.PGMN.lr=args.lr
        model_options.PGMN.wdecay=args.wdecay
        model_options.PGMN.depth=args.depth
        model_options.PGMN.lr_structural=args.lr_structural

        model_options.PGMN.net_arch=args.net_arch
        model_options.PGMN.n_modules=args.num_modules
        model_options.PGMN.temp=args.temp
        model_options.PGMN.str_prior_temp=args.str_prior_temp
        model_options.Module.n_heads_decoder=args.n_heads_decoder
        model_options.PGMN.fix_layers_below_on_addition=args.fix_layers_below_on_addition
        model_options.PGMN.module_type=args.module_type
        model_options.PGMN.str_prior_factor=args.str_prior_factor
        model_options.PGMN.concat=args.concat
        model_options.PGMN.beam_width=args.beam_width
        model_options.PGMN.catch_outliers_old=args.catch_outliers_old
        model_options.PGMN.module_init=args.module_init
        model_options.PGMN.multihead=multihead
        model_options.PGMN.deviation_threshold=args.deviation_threshold
        model_options.PGMN.mask_str_loss=args.mask_str_loss
        model_options.PGMN.MWR_attempt_phase_length=args.MWR_attempt_phase_length
        model_options.PGMN.projection_phase_length=args.projection_phase_length
        model_options.PGMN.optmize_structure_only_free_modules=args.optmize_structure_only_free_modules
        model_options.PGMN.automated_module_addition=1
        model_options.PGMN.active_unfreezing=args.active_unfreezing
        model_options.PGMN.unfreeze_structural=args.unfreeze_structural
        model_options.PGMN.treat_unfreezing_as_addition=args.treat_unfreezing_as_addition
        model_options.PGMN.use_structural_for_outlier = args.use_structural_for_outlier

        model = PGMN_net(model_options.PGMN,
                                    model_options.Module,
                                    model_options.Gate,
                                    i_size =i_size, 
                                    channels=3,
                                    hidden_size=args.hidden_size, 
                                    num_classes= n_classes,
                                    n_tasks=args.n_tasks).to(device)
    elif gating == 'experts':    
        model_options.Experts.lr=args.lr     
        model_options.Experts.wdecay=args.wdecay
        model_options.Experts.regime='normal' 
        model_options.Experts.depth=args.depth   
        model_options.Experts.net_arch=args.net_arch
        model_options.Experts.n_modules=args.num_modules
        model_options.Experts.module_type=args.module_type
        model = ExpertMixture(model_options.Experts, 
                                model_options.Module, 
                                i_size =i_size, 
                                channels=3,
                                hidden_size=args.hidden_size, 
                                num_classes=n_classes).to(device)
    
    return model

def test(model, classes, test_loader, temp, str_prior_temp, task_id=None, modules_selected =[]):
    model.eval()
    for g, gate_layer in enumerate(model.gate_net):
        gate_layer.PoolGateNet.eval()

    model_test = 1

    result = defaultdict(lambda: 0)
    acc_test = 0                   
    mask = []   
    task_head_selection=[]
    for i, (x,y) in enumerate(test_loader):
        i+=1  
        x,y = x.to(device), y.to(device)           
        forward_out, regularizer_hnet, add_new_num, ghnet_MWR_loss_total = model(x, inner_loop=False, task_id=task_id, temp=temp, str_prior_temp=str_prior_temp, modules_selected = modules_selected)
        logit = forward_out.logit    
        logit = logit.squeeze()
        if task_id is None:
            task_head_selection.append(forward_out.info['selected_decoder'])
        acc_test += torch.sum(logit.max(1)[1] == y).float()/len(y)
        if isinstance(model, PGMN_net):
            mask.append(forward_out.mask)
            if classes is not None:     
                dev_mask = list(map(lambda x: x.T.detach().cpu().numpy().mean(0),  forward_out.info['deviation_mask']))
                str_loss_per_module = list(map(lambda x: x.T.detach().cpu().numpy().mean(0), forward_out.mask_bf_act))
                z_score_per_module = list(map(lambda x: x.T.detach().cpu().numpy().mean(0), forward_out.info['outlier_signal']))
                for l, ms in enumerate(dev_mask):
                    for m, v in enumerate(ms):   
                        result['deviation_mask/'+f'l_{l}_m_{m}'] += (v - result['deviation_mask/'+f'l_{l}_m_{m}']) /i 
                        result['loss_str/'+f'l_{l}_m_{m}'] += (str_loss_per_module[l][m] - result['loss_str/'+f'l_{l}_m_{m}'])/i
                        
                        result['z_score/'+f'l_{l}_m_{m}'] += (z_score_per_module[l][m] - result['z_score/'+f'l_{l}_m_{m}'])/i
    result['task_head_selection']=np.array(task_head_selection)
    if len(mask)>0:
        ######          hnet         ######
        selected_mask_hnet = mean_mask(mask,model_test)

        #############################
        mask=torch.stack(mask).mean(0)         
    return acc_test/len(test_loader), result, mask, selected_mask_hnet


def consolidate_fim(fim_previous, fim_new, task):
    # consolidate the fim_new into fim_previous in place
    if isinstance(fim_new, PMatDiag):
        fim_previous.data = (
            copy.deepcopy(fim_new.data) + fim_previous.data * (task)
        ) / (task + 1)
    else:
        raise NotImplemented
    return fim_previous  

def train_on_task(model:nn.Module, args:ArgsGenerator, train_loader, valid_loader, test_loader, epochs=400, 
                    str_temp=1, anneal=False, str_anneal=False,  task_id=None, str_only=False, classes=range(10),
                  fim=None, fims=[], train_str=True, reg_factor=1,reg_factor_hnet = 10, patience=0, er_buffer:Buffer=None):
    if isinstance(fims, PMatAbstract):
        fim=fims
    train_hnet = True
    temp=args.temp
    str_temp=args.str_prior_temp
    # these are set to 0 in all experiments
    epochs_str_only_after_addition =(int((task_id>0))*args.epochs_str_only_after_addition)
    epochs_structure_only_at_start=(int((task_id>0))*args.epochs_structure_only_at_start)
    ############
    _epochs_str_only_after_addition = 0
    ewc=args.ewc
    s=None
    if ewc>0:
        anchor = PVector.from_model(model.components).clone().detach()
    
    e=0
    n_modules_model = copy.deepcopy(model.n_modules)
    best_model = None
    best_val=0.
    hnet_out_list = []
    history= {}
    history['accuracy']= 0.0
    history['selected_history'] = []
    best_model_selected_modules = []
    ##############################
    acc_pre_task= 0
    acc_pre_test = 0
    add_new_module_num = 0


    while e<epochs:
        # After the first projection phase is completed, the pool-gate parameters are stored at this time and will be used in the subsequent MWR_loss
        if e == model.args.projection_phase_length/model.len_loader:
            for g, gate_layer in enumerate(model.gate_net):
                gate_layer.params = {n: p.clone().detach() for n, p in gate_layer.PoolGateNet.named_parameters() if p.requires_grad}
        #Start pool-gate training during the MWR update phase
        if (e>=model.args.projection_phase_length/model.len_loader) and (e<= ((model.args.projection_phase_length + model.args.MWR_attempt_phase_length)/model.len_loader)):
            model.args.train_ghnet_MWR = True
            model.args.using_MWR = False
        else:
            model.args.train_ghnet_MWR = False
            if e > ((model.args.projection_phase_length + model.args.MWR_attempt_phase_length)/model.len_loader):
                model.args.using_MWR = True
            if e < model.args.projection_phase_length/model.len_loader:
                model.args.using_MWR = False
        # Whether we will calculate the regularizer.
        calc_reg = task_id > 0 and model.args.hnet_beta > 0 and len(model.gate_net) > 0 and e>0
        torch.cuda.empty_cache()
        len_loader = len(train_loader)
        loader = train_loader
        model.train()
        ####       hnet
        for g, gate_layer in enumerate(model.gate_net):
            gate_layer.PoolGateNet.train()

        ####       hnet
        acc=0
        reg = 0
        selected_history_loader = []
        for bi, batch in enumerate(loader):
            x,y = batch[0].to(device), batch[1].to(device)
            ##################################################
            # Add to ER Buffer only during the first epoch
            if er_buffer is not None and e==0:
                er_buffer.add_reservoir({"x": x, "y": y, "t": task_id})
            ##################################################
            model.zero_grad()
            model.optimizer_hnet.zero_grad()
            model.task_emb_optimizer.zero_grad()

            temp_e = torch.tensor(temp) if not anneal else torch.tensor(temp) * cosine_rampdown(e, epochs+10)
            str_temp_e = torch.tensor(str_temp) if not str_anneal else torch.tensor(str_temp) * cosine_rampdown(e, epochs+10)

            forward_out, regularizer_hnet,add_new_num, ghnet_MWR_loss_total = model(x, inner_loop=False, task_id=task_id, temp=temp_e, str_prior_temp=str_temp_e,
                                   record_running_stats=True, hnet_out_list = hnet_out_list, calc_reg = calc_reg, detach_head=(e<_epochs_str_only_after_addition and e<epochs_structure_only_at_start),
                                   epoch=e, acc_pre_task = acc_pre_task, acc_pre_test = acc_pre_test,add_new_module_num= add_new_module_num, s=s)

            if not any(map(lambda x: isinstance(model, x), [ExpertMixture])):     
                if torch.sum(model.n_modules) > torch.sum(n_modules_model):     
                    #A new module was added at this iteration 
                    if args.early_stop_complete:
                        #discard the best model found sofar
                        best_model=None        
                        best_val=0.      
                    if not model.n_modules[-1] > n_modules_model[-1]:
                        #if it was added not on the last layer
                        _epochs_str_only_after_addition = e+epochs_str_only_after_addition # use only structural loss for epochs_str_only_after_addition epochs
                        #model.args.projection_phase_length/len_loader = projection phase length in epochs
                        #train at least for args.projection_phase_length +  args.num_epochs_added_after_projection epochs more
                        epochs=max(epochs, e+int(model.args.projection_phase_length/len_loader+args.num_epochs_added_after_projection))
                        
                    else:
                        #module was added on the last layer = no projection phase
                        #train at elast for 10 epochs more
                        epochs=max(epochs, e+10) 
                n_modules_model = copy.deepcopy(model.n_modules)
            add_new_module_num = add_new_num
            logit = forward_out.logit
            logit=logit.squeeze()                
            logit = logit[:len(y)]
            outer_loss = loss_function(logit, y)

            if forward_out.regularizer is not None and train_str and not torch.isnan(forward_out.regularizer):
                regularizer = forward_out.regularizer
                reg+=regularizer.detach()
                outer_loss+= reg_factor*regularizer

            if ghnet_MWR_loss_total is not None and not torch.isnan(ghnet_MWR_loss_total):
                if ghnet_MWR_loss_total != 0:
                     outer_loss+= reg_factor_hnet*ghnet_MWR_loss_total

            if not (regularizer_hnet == 0):
                outer_loss+= reg_factor_hnet*regularizer_hnet

        ###############
            ###  EWC  ##  
            if ewc>0:
                if fim is not None:
                    v_current = PVector.from_model(model.components)
                    regularizer=(fim.vTMv(v_current - anchor))
                    outer_loss += ewc*regularizer
                    reg+=regularizer.detach()
                elif len(fims)>0:
                    regularizer=0
                    v_current = PVector.from_model(model.components)
                    for f in fims:
                        regularizer+=(f.vTMv(v_current - anchor))                    
                    outer_loss += ewc*regularizer
                    reg+=regularizer.detach()
            ##############
            ###  REPLAY ##   
            if task_id > 0 and er_buffer:
                x_buffer = []
                y_buffer = []
                if args.multihead=='none':
                    #single head
                    for past_t in range(task_id):
                        replay_bs=x.size(0)
                        b_samples = er_buffer.sample(replay_bs,only_task=past_t)
                        x_buffer.append(b_samples['x'])
                        y_buffer.append(b_samples['y'])
                    x_buffer=torch.cat(x_buffer)
                    y_buffer=torch.cat(y_buffer)

                    b_logits = forward_out = model(x_buffer.to(device), inner_loop=False, task_id=past_t, temp=temp_e, str_prior_temp=str_temp_e, record_running_stats=True,
                                                        detach_head=(e<_epochs_str_only_after_addition and e<epochs_structure_only_at_start)).logit
                    loss_replay = loss_function(b_logits, y_buffer.to(device))
                    outer_loss += loss_replay
                else:
                    for past_t in range(task_id):
                        replay_bs=x.size(0)
                        b_samples = er_buffer.sample(replay_bs,only_task=past_t)
                        b_logits = forward_out = model(b_samples['x'].to(device), inner_loop=False, task_id=past_t, temp=temp_e, str_prior_temp=str_temp_e, record_running_stats=True,
                                                            detach_head=(e<_epochs_str_only_after_addition and e<epochs_structure_only_at_start)).logit
                        loss_replay = loss_function(b_logits, b_samples["y"].to(device))
                        outer_loss += loss_replay

                        print("loss_replay:",loss_replay)
                    outer_loss/=task_id+1
            ##############
            if outer_loss.requires_grad:
                outer_loss.backward()

                if model.args.train_ghnet_MWR and task_id > 0:
                    for gate in model.gate_net:
                        gate.fisher = gate._compute_fisher(model.args.MWR_attempt_phase_length/model.len_loader, task_id)

                model.optimizer.step()
                if model.optimizer_structure is not None:
                    model.optimizer_structure.step()
                if (task_id > 0) and (not model.args.train_ghnet_MWR):
                    if model.optimizer_hnet is not None:
                        model.optimizer_hnet.step()
                    if model.task_emb_optimizer is not None:
                        model.task_emb_optimizer.step()


            acc += torch.sum(logit.max(1)[1] == y).float()/len(y)
            selected_history_loader.append(forward_out.mask)
        acc_pre_task = acc/len_loader
        print('train acc: ',acc/len_loader, 'epoch: ',e, 'reg: ', reg/len_loader)
        selected_mask = mean_mask(selected_history_loader)
        # max_common_list = find_most_common_sublist(selected_history_loader,add_module_scheme = args.add_module_scheme,k=args.k_top)
        update_history_selected_module(history,acc/len_loader, selected_mask)

        # keep track of the best model as measured on the validation set
        if args.keep_best_model:
            if e>=_epochs_str_only_after_addition:  
                validate=False
                if hasattr(model, 'projection_phase'):
                    if not model.projection_phase: 
                        #should not be in the eprojection phase when validating
                        validate=True
                else:
                    validate=True
                if validate:
                    model.eval()
                    for g, gate_layer in enumerate(model.gate_net):
                        gate_layer.PoolGateNet.eval()

                    with torch.no_grad():
                        acc_valid, _, _ ,selected_mask_hnet = test(model, classes, valid_loader, temp=temp_e, str_prior_temp=str_temp_e, task_id=task_id, modules_selected=history['selected_history'])
                    if best_val < acc_valid:
                        best_val = acc_valid
                        best_model = copy.deepcopy(model.state_dict())
                        best_model_selected_modules = selected_mask_hnet

        if e %5 == 0:
            #test on the test set
            model.eval()
            for g, gate_layer in enumerate(model.gate_net):
                gate_layer.PoolGateNet.eval()

            with torch.no_grad():
                acc_test, result, _, selected_mask_hnet = test(model, classes, test_loader, temp=temp_e, str_prior_temp=str_temp_e, task_id=task_id, modules_selected = history['selected_history'] )    #hnet
            if 's_long' not in args.task_sequence:
                log_wandb(result, prefix=f'result_{task_id}/')      
            print('test acc: ', acc_test, ' epoch ', e)
            acc_pre_test = acc_test
            log_wandb({f'task_{task_id}/test_acc':acc_test})
        e+=1  
    selected_modules = history['selected_history']
    if best_model is not None:          
        if args.gating=='locspec': 
            # make sure the model returned has same number of modules as the best_model
            # potentially, best model can have less modules than the current model
            modules_best_model=best_model['_n_modules']
            for l, n_mod_at_layer in enumerate(modules_best_model):
                if n_mod_at_layer<model.n_modules[l]:
                    model.remove_module(at_layer=l)
        model.load_state_dict(best_model, strict=True)
        selected_modules = best_model_selected_modules
        print('Module selection of the best model at the end of training: ',selected_modules)
        print('Num of functional components in Layer 0:', model._n_modules[0])
        print('Num of functional components in Layer 1:', model._n_modules[1])
        print('Num of functional components in Layer 2:', model._n_modules[2])
        print('Num of functional components in Layer 3:', model._n_modules[3])

    if ewc>0:  
        #calculate FIM 
        model.eval()
        def function(*d):                    
            return model(d[0].to(device)).logit
        fim = FIM(model=model.components,
                    function=function,
                    loader=train_loader,
                    representation=PMatDiag,
                    n_output=model.num_classes,
                    variant='classif_logits',
                    device=device)
        return model, fim
    return model, selected_modules

def bn_warmup(model, loader:DataLoader, task_id=None, bn_warmup_steps=100):
    """ warms up batchnorms by running several forward passes on the model in training mode """
    was_training=model.training
    model.train()
    ####    hnet
    for g, gate_layer in enumerate(model.gate_net):
        gate_layer.PoolGateNet.eval()

    ####    hnet
    automated_module_addition_before=1#model.args.automated_module_addition
    model.args.automated_module_addition=0
    if bn_warmup_steps>0:   
        for i, (x,_) in enumerate(loader):
            model(x.to(device), record_running_stats=False, task_id=task_id if task_id is not None else -1, inner_loop=False) #temp=temp, str_prior_temp=str_temp,
            if i>=bn_warmup_steps:
                break
    model.args.automated_module_addition=automated_module_addition_before
    if not was_training:
        model.eval()
    return model

def test_with_bn(model, classes, test_loader, temp, str_temp, task_id=None, bn_warmup_steps=100, modules_selected=[]):
    """ test mode with batchnomr warmup """
    model = bn_warmup(model, test_loader, task_id, bn_warmup_steps)  
    return test(model, classes, test_loader, temp, str_temp, task_id=task_id, modules_selected=modules_selected)

def get_accs_for_tasks(model, args:ArgsGenerator, loaders:List[DataLoader], accs_past: List[float]=None, task_agnostic_test: bool=False, remember_gate_select = []):
    accs=[]        
    Fs = []
    masks=[]               
    task_oh_selection_accs=[]                    
    #make sure we test the same model for each task, since we may do batchnorm warm-up, this is needed here
    state_dict=copy.deepcopy(model.state_dict())
    for ti, test_loader in enumerate(loaders):
        # pool-gate_PGMN without using MWR
        model.args.using_MWR = False

        model.load_state_dict(state_dict, strict=True)             
        #dont warm up batch norm on the last task, as it just trained on it anyways   
        # no warm up for the last loader, if no batch norm is used
        steps_bn_warmup = 200*int(args.use_bn)*int(args.gating!='experts')*(1-(int(ti==(len(loaders)-1))*int(not task_agnostic_test)))*(1-int(args.keep_bn_in_eval_after_freeze))
        #make this explicit here
        if args.keep_bn_in_eval_after_freeze:
            steps_bn_warmup=0
        print('steps_bn_warmup', steps_bn_warmup)
        print(ti)
        task_module_selected = remember_gate_select[ti]
        task_index = task_module_selected
        acc, info, mask, selected_mask_hnet = test_with_bn(model, None, test_loader, model.min_temp, model.min_str_prior_temp, task_id=ti, modules_selected= task_index)
        acc = acc.cpu().item()
        accs.append(acc)
        masks.append(mask)
        if info is not None and len(info['task_head_selection'])>0:
            task_oh_selection_accs.append(sum(info['task_head_selection']==ti)/len(info['task_head_selection']))
        else:
            task_oh_selection_accs.append(1.)
    #     ####################
        if accs_past is not None:
            Fs.append(acc-accs_past[ti])
    model.load_state_dict(state_dict, strict=True)   
    return accs,Fs,masks,task_oh_selection_accs

def get_oh_init_idx(model, dataloader:DataLoader, args:ArgsGenerator):
    if args.init_oh=='most_likely':
        selected_head=[]
        for x,_ in dataloader:
            x = x.to(device)
            selected_head.append(model(x).info['selected_decoder'])
        return Counter(selected_head).most_common(1)[0][0]
    else:
        return None
    pass

def train(args:ArgsGenerator, model, task_idx, train_loader_current, test_loader_current, valid_dataloader, fim_prev,er_buffer):
    #args.projection_phase_length*len(train_loader_current) = prpojection phase length in number of iterations (batch updates)
    model.args.projection_phase_length = args.projection_phase_length*len(train_loader_current)
    model.args.MWR_attempt_phase_length = args.MWR_attempt_phase_length*len(train_loader_current)
    model.len_loader = len(train_loader_current)
    if task_idx>0:
        if args.warmup_bn_bf_training: 
            #warup batchnorms before training on task
            steps_bn_warmup = 200*int(args.use_bn)*int(args.gating!='experts')
            model = bn_warmup(model, train_loader_current, None, steps_bn_warmup)

        #PGMN
        #When training new tasks, ensuring that the training starts with a attempt phase length allows pool-te to explore combinations of existing modules
        model._steps_since_last_addition =model._steps_since_last_addition * 0

    if args.running_stats_steps==0:
        model.module_options.running_stats_steps=len(train_loader_current)

    epochs=args.epochs
    best_valid_acc, best_model = None, None
    model , history_selected_module=train_on_task(model, args, train_loader_current, valid_dataloader, test_loader_current, epochs=epochs, anneal=args.anneal, str_anneal=args.str_anneal, task_id=task_idx,
                                                  reg_factor=args.reg_factor, fims=fim_prev, er_buffer=er_buffer, reg_factor_hnet = args.reg_factor_hnet)
    
    if args.ewc>0:  
        model, fim = model
        if args.ewc_online:
            if not isinstance(fim_prev, PMatAbstract):
                fim_prev=fim
            else:
                fim_prev=consolidate_fim(fim_previous=fim_prev ,fim_new=fim, task=task_idx)
        else:
            fim_prev.append(fim)
    # model_p=copy.deepcopy(model)
    with torch.no_grad():
        test_acc = test(model, None, test_loader_current, model.min_temp, model.min_str_prior_temp, task_id=task_idx, modules_selected=history_selected_module)[0].cpu().item()          ##  version of hnet

    if best_valid_acc is None:
        with torch.no_grad():
            valid_acc = test(model, None, valid_dataloader, model.min_temp, model.min_str_prior_temp, task_id=task_idx, modules_selected=history_selected_module)[0].cpu().item()           ##  hnet version

    else:
        valid_acc=best_valid_acc
    return model,test_acc,valid_acc,fim_prev, history_selected_module

def main(args:ArgsGenerator, task_gen:TaskGenerator):              
    t = task_gen.add_task()  
    model=init_model(args, args.gating, n_classes=t.n_classes.item(),  i_size=t.x_dim[-1]) 

    ##############################
    #Replay Buffer                 
    if args.replay_capacity!=0:
        rng = np.random.RandomState(args.seed)
        if args.replay_capacity<0:
            net_size = 24 * sum([np.prod(p.size()) for p in model.parameters()]) * args.n_tasks
            # we assume that 1 pixel can be stored using 1 byte of memory
            args.replay_capacity = net_size // np.prod(t.x_dim)
        er_buffer=BalancedBuffer(args.replay_capacity,
                        input_shape=t.x_dim,   
                        extra_buffers={"t": torch.LongTensor},
                        rng=rng).to(device)
    else:
        er_buffer = None
    ##############################
             
    try:
        wandb.watch(model)
    except:
        pass 
    n_tasks=args.n_tasks
    train_loaders=[]
    test_loaders=[]
    valid_loaders=[]      
    test_accuracies_past = []
    valid_accuracies_past = [] 
    fim_prev=[]
    remember_gate_select = []
    for i in range(n_tasks):                     
        print('==='*10)
        print(f'Task train {i}, Classes: {t.concepts}')   
        print('==='*10)                                                                                         
        train_loader_current, valid_dataloader, test_loader_current = create_dataloader_ctrl(task_gen, t, args,0, batch_size=args.batch_size, labeled=True, task_n=i), create_dataloader_ctrl(task_gen, t, args,1,args.batch_size, labeled=True, shuffle_test=('ood' in args.task_sequence), task_n=i), create_dataloader_ctrl(task_gen, t, args,2,args.batch_size, labeled=True, shuffle_test=('ood' in args.task_sequence), task_n=i) 
        ##########################
        #task embedding optimizer
        model.task_emb_optimizer = model.get_task_emb_optimizers(i)

        ######
        if args.regime=='cl':
            model,test_acc,valid_acc,fim_prev,history_selected_module = train(args,model,i,train_loader_current,test_loader_current,valid_dataloader,fim_prev,er_buffer)
            remember_gate_select.append(history_selected_module)
            test_accuracies_past.append(test_acc)
            valid_accuracies_past.append(valid_acc)
            test_loaders.append(test_loader_current)
            valid_loaders.append(valid_dataloader)
            ####################
            #Logging
            ####################
            #Current accuracy     
            log_wandb({f'test/test_acc_{i}':test_acc})
            log_wandb({f'valid/valid_acc_{i}':valid_acc})
            #Avv acc sofar (A)
            if args.log_avv_acc:
                accs, _, _,_ = get_accs_for_tasks(model, args, test_loaders, task_agnostic_test=args.task_agnostic_test,remember_gate_select=remember_gate_select)
                log_wandb({f'test/avv_test_acc_sofar':np.mean(accs+[test_acc])})
                accs_valid, _, _,_ = get_accs_for_tasks(model, args, valid_loaders, task_agnostic_test=args.task_agnostic_test,remember_gate_select=remember_gate_select)
                log_wandb({f'test/avv_test_acc_sofar':np.mean(accs_valid+[valid_acc])})
        elif args.regime=='multitask':
                #collect data first
                train_loaders.append(train_loader_current)
                test_loaders.append(test_loader_current)
                valid_loaders.append(valid_dataloader)
        #Model
        n_modules = torch.tensor(model.n_modules).cpu().numpy()     
        log_wandb({'total_modules': np.sum(np.array(n_modules))}, prefix='model/')
        ####################
        #Get new task
        try:
            t = task_gen.add_task()
        except:
            print(i)
            break
        model._steps_for_ghnet_MWR = model._steps_for_ghnet_MWR * 0.
        for g, gate_layer in enumerate(model.gate_net):
           gate_layer.fisher = gate_layer._init_fisher(task_id = i+1)
        if args.task_sequence=='s_long30' and i==30:
            print(i)
            break 
        #fix previous output head          
        if isinstance(model, PGMN_net):
            if isinstance(model.decoder, nn.ModuleList):   
                if hasattr(model.decoder[i],'weight'):
                    print(torch.sum(model.decoder[i].weight))
                
        if args.multihead!='none':
            model.fix_oh(i)   
            init_idx=get_oh_init_idx(model, create_dataloader_ctrl(task_gen, t, args,0,batch_size=args.batch_size, labeled=True, task_n=i), args)
            print('init_idx', init_idx)        
            model.add_output_head(t.n_classes.item(), init_idx=init_idx)
        else:
            #single head mode: create new, larger head
            model.add_output_head(model.decoder.out_features+t.n_classes.item(), state_dict=model.decoder.state_dict())

        if args.gating not in ['experts']:
            for l in range(len(n_modules)):
                log_wandb({f'total_modules_l{l}': n_modules[l]}, prefix='model/')
            if args.use_ghnet:
                if args.use_backup_system:
                    model.freeze_permanently_structure()
                else:
                    for l, layer in enumerate(model.components):
                        for m in layer:
                            m.freeze_functional(inner_loop_free=False)
                            # m.freeze_structural()
                            m.module_learned = torch.tensor(1.)
                            # model.add_modules(at_layer=l)
            else:
                if args.use_structural:
                    if args.use_backup_system:
                        model.freeze_permanently_structure()
                    else:
                        for l,layer in enumerate(model.components):
                            for m in layer:
                                m.freeze_functional(inner_loop_free=False)
                                m.freeze_structural()
                                m.module_learned=torch.tensor(1.)
                                # model.add_modules(at_layer=l)
            #model.optimizer, model.optimizer_structure = model.get_optimizers()
            model.optimizer, model.optimizer_structure = model.get_optimizers()


    if args.regime=='multitask':
        #train
        train_set = torch.utils.data.ConcatDataset([dl.dataset for dl in train_loaders])
        test_set = torch.utils.data.ConcatDataset([dl.dataset for dl in test_loaders])
        valid_set = torch.utils.data.ConcatDataset([dl.dataset for dl in valid_loaders])
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=1)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=1)
        valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=1)
        model,test_acc,valid_acc,_,history_selected_module = train(args,model,0,train_loader,test_loader,valid_loader,None,None,None)
        remember_gate_select.append(history_selected_module)
        test_accuracies_past=None
        valid_accuracies_past=None

    #########################
    # this is for debugging     
    if isinstance(model, PGMN_net):
        if isinstance(model.decoder, nn.ModuleList):
            for d in model.decoder:
                if hasattr(d,'weight'):   
                    print(torch.sum(d.weight))
    #########################
    accs_test, Fs, masks_test, task_selection_accs = get_accs_for_tasks(model, args, test_loaders, test_accuracies_past, task_agnostic_test=args.task_agnostic_test, remember_gate_select= remember_gate_select)
    # masks_test = remember_gate_select
    max_length = 0
    masks_test_=[]
    # 遍历list1中的每个子列表
    for sublist in remember_gate_select:
        # 遍历子列表中的每个tensor，并找到其最大长度
        for tensor in sublist:
            if len(tensor) > max_length:
                max_length = len(tensor)
    for i, module_selection_task in enumerate(remember_gate_select):
        print(f'Module selection in task {i} ')
        max_length_layer = max(len(sublist) for sublist in module_selection_task)
        mask_test = torch.zeros(model.args.depth, max_length).to(device)
        #The fill element of mask
        for l,  layer_weight in enumerate(module_selection_task):
            # num_module = model.n_modules_at_layer(l)
            mask_test[l, :max_length_layer]  = layer_weight
            print(f'Module selection of layer{l} {mask_test[l]}')
            masks_test_.append(mask_test)


    for ti, (acc, Frg, task_selection_acc) in enumerate(zip(accs_test, Fs, task_selection_accs)):
        log_wandb({f'test_acc_{ti}':acc}, prefix='test/')
        #Forgetting (test)
        log_wandb({f'F_test_{ti}':Frg}, prefix='test/')           
        #Task selection accuracy (only relevant in not ask id is geven at test time) (test)
        log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='test/')    
    ####################
    #Average accuracy (test) at the end of the sequence 
    print(accs_test)
    print('Average accuracy (test) at the end of the sequence:',np.mean(accs_test))
    log_wandb({"mean_test_acc":np.mean(accs_test)})#, prefix='test/')
    #Average forgetting (test)
    log_wandb({"mean_test_F":np.mean(Fs)})#, prefix='test/')
    ####################
    #Masks / Module usage
    if len(masks_test_)>0 and args.gating=='locspec':
        pyplot.clf()
        fig, axs = pyplot.subplots(1,len(test_loaders),figsize=(15,4))
        for i, ax in enumerate(axs):
            im = sns.heatmap(F.normalize(masks_test_[i].cpu().T, p=1, dim=0), vmin=0, vmax=1, cmap='Blues', cbar=False, ax=ax, xticklabels=[0,1,2,3])
            ax.set_title(f'Task {i}')
            for _, spine in im.spines.items():
                spine.set_visible(True)
        pyplot.setp(axs[:], xlabel=f'layer')
        pyplot.setp(axs[0], ylabel='module')
        log_wandb({f"module usage": wandb.Image(fig)})
        if args.save_figures:
            for i in range(len(masks_test_)):
                print(masks_test_[i].cpu().T)
            for i in range(len(masks_test_)):
                print(F.normalize(masks_test_[i].cpu().T, p=1, dim=0))
            fig.savefig(f'module_selection_{args.task_sequence}.pdf', format='pdf', dpi=300)
    ####################
    accs_valid, Fs_valid, _, task_selection_accs = get_accs_for_tasks(model, args, valid_loaders, valid_accuracies_past, task_agnostic_test=args.task_agnostic_test, remember_gate_select = remember_gate_select)
    for ti, (acc, Frg, task_selection_acc) in enumerate(zip(accs_valid, Fs_valid, task_selection_accs)):
        log_wandb({f'valid_acc_{ti}':acc}, prefix='valid/')
        #Forgetting (valid)
        log_wandb({f'F_valid_{ti}':Frg}, prefix='valid/') 
        #Task selection accuracy (only relevant in not ask id is geven at test time)(valid)
        log_wandb({f'Task_selection_acc{ti}':task_selection_acc}, prefix='valid/')        
    ####################
    print('Average accuracy (valid) at the end of the sequence:',np.mean(accs_valid))
    #Average accuracy (valid) at the end of the sequence 
    log_wandb({"mean_valid_acc":np.mean(accs_valid)})#, prefix='valid/')
    #Average forgetting (valid)
    log_wandb({"mean_valid_F":np.mean(Fs_valid)})#, prefix='test/')
    ####################    
         
    if args.task_sequence_test is not None and 'ood' in args.task_sequence:
        #test on all combinations of features and classes
        state_dict_learned=model.state_dict()
        task_gen_test = ctrl.get_stream(args.task_sequence_test, seed=args.stream_seed)
        classes=[]
        transformations=[]
        task_id = -1
        accuracies=[]    
        accuracies_valid=[]
        masks_test=[]
            
        for i, t in enumerate(task_gen_test):    
            model.load_state_dict(state_dict_learned)
            classes_name = str([int(s) for s in str(t.concepts).split() if s.isdigit()])
            if len(classes)==0 or classes[-1]!=classes_name:
                #task witched
                task_id+=1
            print(f'Task {i}, Classes: {t.concepts}')     
            print(t.transformation.trans_descr)
            print(f"Task id {task_id}")
            classes.append(classes_name)      
            descr=t.transformation.trans_descr.split('->')[-1]
            name=construct_name_ctrl(descr)
            transformations.append(name)#t.transformation.trans_descr.split('->')[-1])
            loader_valid, loader_test = create_dataloader_ctrl(task_gen, t, args,1, batch_size=args.batch_size, labeled=True, task_n=i), create_dataloader_ctrl(task_gen, t, args,2, batch_size=args.batch_size, labeled=True, task_n=i)       
            task_modules_selected = remember_gate_select[i]
            task_index = task_modules_selected['selected_history']
            test_acc, _, mask,selected_mask_hnet = test_with_bn(model, None, loader_test, model.min_temp, model.min_str_prior_temp, task_id=task_id if not args.task_agnostic_test else None, bn_warmup_steps=200, modules_selected=task_index)
            test_acc=test_acc.cpu().item()
            try:
                masks_test.append(mask.detach())
            except:
                masks_test.append(mask)
            valid_acc, _, mask, selected_mask_hnet = test_with_bn(model, None, loader_valid, model.min_temp, model.min_str_prior_temp, task_id=task_id if not args.task_agnostic_test else None, bn_warmup_steps=100, modules_selected=task_index)[0].cpu().item()
            accuracies.append(test_acc)
            accuracies_valid.append(valid_acc)
        
        log_wandb({f"mean_test_ood": np.mean(accuracies)}) 
        log_wandb({f"mean_valid_ood": np.mean(accuracies_valid)})
        array=[]
        array_valid=[]
        indexes = np.unique(transformations, return_index=True)[1]
        unique_transformations = [transformations[index] for index in sorted(indexes)]
        for tr in unique_transformations:
            results_for_transform=[]     
            results_for_transform_valid=[]    
            for i, tr2 in enumerate(transformations):
                if tr==tr2:
                    results_for_transform.append(accuracies[i])
                    results_for_transform_valid.append(accuracies_valid[i])
            array.append(results_for_transform)
            array_valid.append(results_for_transform_valid)
        ####################
        #Masks / Module usage ood  
        if len(masks_test)>0 and args.gating=='locspec':  
            fig, axs = pyplot.subplots(len(unique_transformations),len(np.unique(classes)),figsize=(10,2*len(unique_transformations)))
            fig.tight_layout(pad=2.5)
            for row, ax_row in enumerate(axs):
                for column, ax in enumerate(ax_row):
                    im = ax.imshow(masks_test[column*len(axs)+row].cpu().T, cmap='Blues')
                    ax.set_title(unique_transformations[row].replace('\n', ''))
                    ax.set_yticks([0,1,2,3,4])  
                    ax.set_xticks([0,1,2,3])
                    if row == column:
                        for spine in ax.spines.values(): 
                            spine.set_edgecolor('red')#, linewidth=2)
            # set labels
            for i,cl in enumerate(np.unique(classes)):
                plt.setp(axs[-1, i], xlabel=f'layer\nClasses {cl}')
            plt.setp(axs[:, 0], ylabel='module')
            pyplot.savefig('module_selection.pdf', format='pdf',dpi=300, bbox_inches='tight')

            log_wandb({f"ood/module_usage": wandb.Image(fig)})
        
        col = np.unique(classes)
        df_cm = pd.DataFrame(array[:len(col)], index = unique_transformations[:len(col)],columns = np.unique(classes))

        log_wandb({f"mean_test_ood": np.mean(array[:len(col)])}) 
        log_wandb({f"mean_valid_ood": np.mean(array_valid[:len(col)])})
        plot_confusion(df_cm, wandb_tag='confusion_matrix')
        return df_cm
    return None
                        
def plot_confusion(df_cm, wandb_tag=None, save_dir=None, labels=None):    
    #################### 
    #create a confusion matrix/
    fig = pyplot.figure(figsize = (15.5,15))
    sn.set(font_scale=2.0)   
    if labels is not None:
        hm=sn.heatmap(df_cm, annot=labels, vmin=0, vmax=1, fmt="", annot_kws={"size":28})
    else:
        hm=sn.heatmap(df_cm, annot=True, vmin=0, vmax=1, fmt=".2%", annot_kws={"size":28})
    hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=30, va="center")
    hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=30)
    
    #confusion matrix
    if wandb_tag is not None:
        log_wandb({f"{wandb_tag}": wandb.Image(fig)})
    if save_dir is not None:
        fig.savefig(save_dir, format='pdf', dpi=300, bbox_inches = 'tight',pad_inches = 0)
    
    matplotlib.rc_file_defaults()

def find_most_common_sublist(nested_list,add_module_scheme,k):
    # nested_list = [[[0,1],[0],[0],[0]],[[0],[0,],[0],[0]],[[0,1],[0],[0],[0]]]

    # 创建一个空列表来存储子列表
    sublists = []

    # 迭代嵌套列表中的每个元素
    for sublist in nested_list:
        # 将每个子列表的元素连接成一个新的子列表
        flattened_sublist = [item for sublist_i in sublist for item in sublist_i]
        sublists.append(flattened_sublist)
    counter = Counter(tuple(item) for item in sublists)
    # 找出出现次数最多的子列表
    most_common_subtuple = counter.most_common(1)[0][0]
    most_common_sublist = list(most_common_subtuple)
    print(most_common_sublist)

    # def split_tuple(tup):
    #     return [list(tup[i:j]) for i, j in zip([0]+[i for i, x in enumerate(tup) if x == 0], [i for i, x in enumerate(tup) if x == 0]+[None])]

    # tup = (0, 1, 0, 0, 1, 0)
    new_lst = split_list(lst=most_common_sublist, add_module_scheme= add_module_scheme, k = k)
    # new_lst = [sublst for sublst in tt if sublst]
    print(new_lst)
    return new_lst

# def select_module_sequence():

def mean_mask(nested_list, model_test = 0):
    list_total =[]
    list_layer0 = []
    list_layer1 = []
    list_layer2 = []
    list_layer3 = []
    # 迭代嵌套列表中的每个元素
    for sublist in nested_list:
        for i, mask in enumerate(sublist):
            if i==0:
                if list_layer0 and mask.shape[0] != list_layer0[-1].shape[0]:
                    list_layer0.clear()
                if model_test == 0:
                    list_layer0.append(mask.mean(1))
                else:
                    list_layer0.append(mask)
            elif i==1:
                if list_layer1 and mask.shape[0] != list_layer1[-1].shape[0]:
                    list_layer1.clear()
                if model_test == 0:
                    list_layer1.append(mask.mean(1))
                else:
                    list_layer1.append(mask)
            elif i==2:
                if list_layer2 and mask.shape[0] != list_layer2[-1].shape[0]:
                    list_layer2.clear()
                if model_test == 0:
                    list_layer2.append(mask.mean(1))
                else:
                    list_layer2.append(mask)
            elif i==3:
                if list_layer3 and mask.shape[0] != list_layer3[-1].shape[0]:
                    list_layer3.clear()
                if model_test == 0:
                    list_layer3.append(mask.mean(1))
                else:
                    list_layer3.append(mask)

    # 将列表中的所有Tensor按第一维度相加,求tensor2第一维度的平均值

    first_dimension_average_layer0 = torch.stack(list_layer0).mean(dim=0)
    first_dimension_average_layer1 = torch.stack(list_layer1).mean(dim=0)
    first_dimension_average_layer2 = torch.stack(list_layer2).mean(dim=0)
    first_dimension_average_layer3 = torch.stack(list_layer3).mean(dim=0)

    list_total.append(first_dimension_average_layer0)
    list_total.append(first_dimension_average_layer1)
    list_total.append(first_dimension_average_layer2)
    list_total.append(first_dimension_average_layer3)
    return list_total


def split_list(lst,add_module_scheme, k =1):
    result = []
    result_finally = []
    temp = []
    sub_list = []
    if add_module_scheme == 1 or add_module_scheme == 3:
        for i in range(0,len(lst),k):
            result.append(lst[i:i+k])
        for index, item in enumerate(result):
            if result[index] == [0,0]:
                temp = item
                temp_i = temp[0]
                result.pop(index)
                result.insert(index,[temp_i])
                temp_i = temp[1]
                result.insert(index+1,[temp_i])
        for iii in result:
            result_finally.append(iii)
    elif add_module_scheme == 2:
        reversed_list = list(reversed(lst))
        for i in range(0,len(reversed_list),k):
            result.append(reversed_list[i:i+k])
            # sub_list = copy.copy(result)
        for index, item in enumerate(result):
            if result[index] == [0,0]:
                temp = item
                temp_i = temp[0]
                result.pop(index)
                result.insert(index,[temp_i])
                temp_i = temp[1]
                result.insert(index+1,[temp_i])
        lst_new = list(reversed(result))
        for iii in lst_new:
            iiii = list(reversed(iii))
            result_finally.append(iiii)
    # elif add_module_scheme == 3:

    return result_finally

def update_history_selected_module(history,new_accuary, max_common_list):
    if new_accuary>history['accuracy'] or new_accuary == history['accuracy']:
        history['accuracy'] = new_accuary
        history['selected_history'] = max_common_list

def modify_hnet_target_shapes_before_train_new_task(list_shapes):
        list_shapes_before_train = list_shapes
        list_shapes_before_train.pop()
        original_weight = list_shapes_before_train[-1]
        arch_original_weight = original_weight[-1]
        arch_original_output = original_weight[0]
        list_shapes_before_train.pop()
        new_weight = [arch_original_output+1, arch_original_weight]
        new_bias = [arch_original_output+1]
        list_shapes_before_train.append(new_weight)
        list_shapes_before_train.append(new_bias)
        return list_shapes_before_train

# def modify_hnet_target_shapes_best_model(list_shapes, train_new_task = True, num_gate_output= None):
#     list_shapes_before_train = list_shapes
#     list_shapes_before_train.pop()
#     original_weight = list_shapes_before_train[-1]
#     arch_original_weight = original_weight[-1]
#     arch_original_output = original_weight[0]
#     list_shapes_before_train.pop()
#     if train_new_task:
#         arch_new_output = arch_original_output +1
#     else:
#         arch_new_output = num_gate_output
#         new_weight = [arch_new_output, arch_original_weight]
#         new_bias = [arch_new_output]
#         list_shapes_before_train.append(new_weight)
#         list_shapes_before_train.append(new_bias)
#         return list_shapes_before_train


if __name__== "__main__":                                     
    parser = ArgumentParser()     
    parser.add_arguments(ArgsGenerator, dest="Global")
    args = parser.parse_args()
    args_generator = args.Global
    dfs=[]
    pr_name=f'pgmn' if args_generator.pr_name is None else args_generator.pr_name
    for r in range(args_generator.n_runs):
        if args_generator.regenerate_seed:
            args_generator.generate_seed()             
        task_gen = ctrl.get_stream(args_generator.task_sequence_train, seed=args_generator.stream_seed)  
        if args_generator.debug:
            pr_name='test'
        # if not args_generator.debug:
        run = wandb.init(project=pr_name, notes=args_generator.wand_notes, settings=wandb.Settings(start_method="fork"), reinit=(args_generator.n_runs>1))
        if not args_generator.debug:      
            wandb.config.update(args_generator, allow_val_change=False)  
        set_seed(manualSeed=args_generator.seed)
        df= main(args_generator, task_gen)
        if df is not None:
            dfs.append(df)
        if not args_generator.debug:
            if not r==(args_generator.n_runs-1):
                try:
                    run.finish()
                except:
                    pass
    #for ood experiments, plot the confusion matrix with the standard deviations
    if len(dfs)>1: 
        df_concat = pd.concat(dfs)   
        mean=df_concat.groupby(df_concat.index, sort=False).mean()
        std=df_concat.groupby(df_concat.index, sort=False).std()
        lables=[]
        for i_r in range(mean.shape[0]):
            l_row=[]
            for i_c in range(mean.shape[0]):
                m_formated="{:.1f}".format(100*mean.iloc[i_r,i_c])
                std_formated="{:.1f}".format(100*std.iloc[i_r,i_c])
                pm=u"\u00B1" #'+/-'
                l_row.append(f"{m_formated}\n{pm}{std_formated}")
            lables.append(l_row)
        plot_confusion(mean, wandb_tag='confusion_matrix_final', save_dir=f'confusion_final_{pr_name}_{args_generator.gating}_{args_generator.ewc}_ood.pdf', labels=lables)
