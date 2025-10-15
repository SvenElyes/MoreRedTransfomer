#code to evalute my models, as it didnt run through the whole trainng and never reached the test part.

#JT made it to epoch 519, val_loss = 0.371, b853ddf8-9356-11f0-9121-e8ebd33aa110
#base sampler made it to epoch 676, val_loss = .366 , 85edbca8-9356-11f0-b8c1-a088c2c68cbc
#how many epochs did the base take? 3969 epochs 

'''
svenelzes@hydra:~/MoreRedTransfomer/MoreRed/logs/LONG_TRAIN$ head -n 500 mdetJT_morered-2998303.out 
INFO:    gocryptfs not found, will not be able to use gocryptfs
sys.path: ['/home/svenelzes/MoreRedTransfomer/MoreRed/src', '/home/svenelzes/MoreRedTransfomer/MoreRed/src/scripts', '/usr/local/lib/python310.zip', '/usr/local/lib/python3.10', '/usr/local/lib/python3.10/lib-dynload', '/usr/local/lib/python3.10/site-packages']
src_path exists: True
Contents: ['scripts', 'morered']

Checking /input-data ...
input_data_dir exists: True
Contents of /input-data: ['energy_U0']
/usr/local/lib/python3.10/site-packages/hydra/_internal/config_loader_impl.py:216: UserWarning: provider=hydra.searchpath in main, path=/home/svenelzes/MoreRedTransfomer/MoreRed/configs is not available.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)

░▒▓██████████████▓▒░ ░▒▓██████▓▒ ▒▓███████▓▒░ ▒▓████████▓▒░▒▓███████▓▒░ ▒▓████████▓▒ ▒▓███████▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓███████▓▒░ ▒▓██████▓▒░ ░▒▓███████▓▒░ ▒▓██████▓▒░  ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ░▒▓██████▓▒ ▒▓█▓▒  ▒▓█▓▒ ▒▓████████▓▒░▒▓█▓▒  ▒▓█▓▒ ▒▓████████▓▒ ▒▓███████▓▒░

[2025-09-17 01:41:46,575][morered.train][INFO] - Running on host: head023
⚙ Running with the following config:
├── run
│   └── work_dir: ${hydra:runtime.cwd}                                          
│       data_dir: /input-data/energy_U0                                         
│       path: ${run.work_dir}/runs                                              
│       experiment: mdet_et_backbone_JT                                         
│       id: ${uuid:1}                                                           
│       ckpt_path: null                                                         
│                                                                               
├── globals
│   └── model_path: best_model                                                  
│       cutoff: 5.0                                                             
│       lr: 0.0001                                                              
│       n_atom_basis: 256                                                       
│       noise_target_key: eps                                                   
│       noise_output_key: eps_pred                                              
│       time_target_key: t                                                      
│       max_atoms:                                                              
│         _target_: morered.utils.get_max_atoms                                 
│       noise_schedule:                                                         
│         _target_: morered.noise_schedules.PolynomialSchedule                  
│         T: 1000                                                               
│         s: 1.0e-05                                                            
│         dtype: float64                                                        
│         variance_type: lower_bound                                            
│       diffusion_process:                                                      
│         _target_: morered.processes.VPGaussianDDPM                            
│         noise_schedule: ${globals.noise_schedule}                             
│         invariant: true                                                       
│         noise_key: ${globals.noise_target_key}                                
│         dtype: float64                                                        
│       time_output_key: t_pred                                                 
│       grad_accum_steps: 2                                                     
│                                                                               
├── data
│   └── _target_: morered.datasets.QM9Filtered                                  
│       datapath: ${run.data_dir}/qm9.db                                        
│       data_workdir: null                                                      
│       batch_size: 16                                                          
│       num_train: 55000                                                        
│       num_val: 10000                                                          
│       num_test: 10000                                                         
│       num_workers: 6                                                          
│       num_val_workers: null                                                   
│       num_test_workers: null                                                  
│       remove_uncharacterized: false                                           
│       distance_unit: Ang                                                      
│       property_units:                                                         
│         energy_U0: eV                                                         
│         energy_U: eV                                                          
│         enthalpy_H: eV                                                        
│         free_energy: eV                                                       
│         homo: eV                                                              
│         lumo: eV                                                              
│         gap: eV                                                               
│         zpve: eV                                                              
│       n_atoms_allowed: null                                                   
│       shuffle_train: true                                                     
│       permute_indices: false                                                  
│       n_overfit_molecules: null                                               
│       pin_memory: true                                                        
│       indices_path: ${run.data_dir}/n_atoms_indices.pkl                       
│       load_properties:                                                        
│       - energy_U0                                                             
│       transforms:                                                             
│       - _target_: schnetpack.transform.CastTo64                               
│       - _target_: schnetpack.transform.SubtractCenterOfGeometry               
│       - _target_: morered.transform.Diffuse                                   
│         diffuse_property: _positions                                          
│         diffusion_process: ${globals.diffusion_process}                       
│         time_key: ${globals.time_target_key}                                  
│       - _target_: morered.transform.AllToAllNeighborList                      
│       - _target_: schnetpack.transform.CastTo32                               
│                                                                               
├── model
│   └── _target_: morered.model.heads_pe.PairEncoder                            
│       activation: gelu                                                        
│       attention_dropout: 0.1                                                  
│       cls_token: false                                                        
│       decomposer_type: pooling                                                
│       embd_dim: 192                                                           
│       ffn_dropout: 0.0                                                        
│       ffn_multiplier: 4                                                       
│       head_dropout: 0.0                                                       
│       n_layers: 12                                                            
│       norm: layer                                                             
│       norm_first: true                                                        
│       num_3d_kernels: 128                                                     
│       num_heads: 12                                                           
│       head_project_down: true                                                 
│       output_key: ${globals.noise_output_key}                                 
│       include_time: true                                                      
│       target_heads:                                                           
│       - forces                                                                
│       - time                                                                  
│       time_head: true                                                         
│       detach_time_head: false                                                 
│       time_key: ${globals.time_target_key}                                    
│       time_output_key: ${globals.time_output_key}                             
│                                                                               
├── task
│   └── optimizer_cls: torch.optim.AdamW                                        
│       optimizer_args:                                                         
│         lr: ${globals.lr}                                                     
│         weight_decay: 0.0                                                     
│       scheduler_cls: schnetpack.train.ReduceLROnPlateau                       
│       scheduler_monitor: val_loss                                             
│       scheduler_args:                                                         
│         mode: min                                                             
│         factor: 0.5                                                           
│         patience: 150                                                         
│         threshold: 0.0                                                        
│         threshold_mode: rel                                                   
│         cooldown: 10                                                          
│         min_lr: 0.0                                                           
│         smoothing_factor: 0.0                                                 
│       _target_: morered.DiffusionTask                                         
│       warmup_steps: 0                                                         
│       skip_exploding_batches: true                                            
│       include_l0: false                                                       
│       time_key: ${globals.time_target_key}                                    
│       noise_key: ${globals.noise_target_key}                                  
│       noise_pred_key: ${globals.noise_output_key}                             
│       outputs:                                                                
│       - _target_: schnetpack.task.ModelOutput                                 
│         name: ${globals.time_output_key}                                      
│         target_property: ${globals.time_target_key}                           
│         loss_fn:                                                              
│           _target_: torch.nn.MSELoss                                          
│         metrics:                                                              
│           mse:                                                                
│             _target_: torchmetrics.regression.MeanSquaredError                
│             squared: true                                                     
│         loss_weight: 0.1                                                      
│       - _target_: morered.task.DiffModelOutput                                
│         name: ${globals.noise_output_key}                                     
│         target_property: ${globals.noise_target_key}                          
│         loss_fn:                                                              
│           _target_: torch.nn.MSELoss                                          
│         metrics:                                                              
│           mse:                                                                
│             _target_: torchmetrics.regression.MeanSquaredError                
│             squared: true                                                     
│         loss_weight: 1.0                                                      
│         nll_metric: null                                                      
│       diffuse_property: _positions                                            
│                                                                               
├── trainer
│   └── _target_: pytorch_lightning.Trainer                                     
│       devices: 1                                                              
│       min_epochs: null                                                        
│       max_epochs: 10000                                                       
│       enable_model_summary: true                                              
│       profiler: null                                                          
│       gradient_clip_val: 0.5                                                  
│       accumulate_grad_batches: 1                                              
│       val_check_interval: 1.0                                                 
│       check_val_every_n_epoch: 1                                              
│       num_sanity_val_steps: 0                                                 
│       fast_dev_run: false                                                     
│       overfit_batches: 0                                                      
│       limit_train_batches: 1.0                                                
│       limit_val_batches: 1.0                                                  
│       limit_test_batches: 1.0                                                 
│       detect_anomaly: false                                                   
│       precision: 32                                                           
│       accelerator: auto                                                       
│       num_nodes: 1                                                            
│       deterministic: false                                                    
│       inference_mode: false                                                   
│                                                                               
├── callbacks
│   └── model_checkpoint:                                                       
│         _target_: schnetpack.train.ModelCheckpoint                            
│         monitor: val_loss                                                     
│         save_top_k: 1                                                         
│         save_last: true                                                       
│         mode: min                                                             
│         verbose: false                                                        
│         dirpath: checkpoints/                                                 
│         filename: '{epoch:02d}'                                               
│         model_path: ${globals.model_path}                                     
│       early_stopping:                                                         
│         _target_: pytorch_lightning.callbacks.EarlyStopping                   
│         monitor: val_loss                                                     
│         patience: 300                                                         
│         mode: min                                                             
│         min_delta: 0.0                                                        
│         check_on_train_epoch_end: false                                       
│         check_finite: false                                                   
│       lr_monitor:                                                             
│         _target_: pytorch_lightning.callbacks.LearningRateMonitor             
│         logging_interval: epoch                                               
│       ema:                                                                    
│         _target_: schnetpack.train.ExponentialMovingAverage                   
│         decay: 0.999                                                          
│       sampling:                                                               
│         _target_: morered.callbacks.SamplerCallback                           
│         sampler: ${sampler}                                                   
│         name: sampling                                                        
│         t: null                                                               
│         max_steps: 200                                                        
│         sample_prior: true                                                    
│         store_path: samples                                                   
│         every_n_batchs: 1                                                     
│         every_n_epochs: 100000                                                
│         start_epoch: 1                                                        
│         log_rmsd: false                                                       
│         log_validity: true                                                    
│         bonds_data_path: null                                                 
│                                                                               
├── logger
│   └── tensorboard:                                                            
│         _target_: pytorch_lightning.loggers.tensorboard.TensorBoardLogger     
│         save_dir: tensorboard/                                                
│         name: default                                                         
│                                                                               
└── seed
    └── None                                                                    
[2025-09-17 01:41:46,640][morered.train][INFO] - Setting float32 matmul precision to <medium>
[2025-09-17 01:41:46,641][morered.train][INFO] - Seed randomly...
[rank: 0] Seed set to 947628727
[2025-09-17 01:41:46,664][morered.train][INFO] - Instantiating datamodule <morered.datasets.QM9Filtered>
[2025-09-17 01:41:48,258][morered.datasets.qm9_filtered][INFO] - Partitioning dataset with 133885 molecules and train dataset size 55000
[2025-09-17 01:41:48,276][morered.datasets.qm9_filtered][INFO] - Train dataset has 55000 molecules and is of type <class 'schnetpack.data.atoms.ASEAtomsData'>
[2025-09-17 01:41:48,278][morered.datasets.qm9_filtered][INFO] - loaded <class 'schnetpack.data.atoms.ASEAtomsData'> with 133885 molecules
[2025-09-17 01:41:48,280][morered.datasets.qm9_filtered][INFO] - in the datadloader for test
[2025-09-17 01:41:49,714][morered.train][INFO] -  keys of the batch: dict_keys(['_idx', 'energy_U0', '_n_atoms', '_atomic_numbers', '_positions', '_cell', '_pbc', 'original__positions', 'eps', 't', '_idx_i_local', '_idx_j_local', '_offsets', '_idx_m', '_idx_i', '_idx_j', 'mask', '_atomic_numbers_padded', '_positions_padded'])
[2025-09-17 01:41:49,715][morered.train][INFO] - Instantiating model <morered.model.heads_pe.PairEncoder>
[2025-09-17 01:41:51,235][morered.train][INFO] - Instantiating task <morered.DiffusionTask>
/usr/local/lib/python3.10/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'model' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['model'])`.
[2025-09-17 01:41:51,632][morered.train][INFO] - Instantiating callback <schnetpack.train.ModelCheckpoint>
[2025-09-17 01:41:51,640][morered.train][INFO] - Instantiating callback <pytorch_lightning.callbacks.EarlyStopping>
[2025-09-17 01:41:51,642][morered.train][INFO] - Instantiating callback <pytorch_lightning.callbacks.LearningRateMonitor>
[2025-09-17 01:41:51,644][morered.train][INFO] - Instantiating callback <schnetpack.train.ExponentialMovingAverage>
[2025-09-17 01:41:51,645][morered.train][INFO] - Instantiating callback <morered.callbacks.SamplerCallback>
[2025-09-17 01:41:51,684][root][INFO] - sample prior True
[2025-09-17 01:41:51,699][morered.train][INFO] - Instantiating logger <pytorch_lightning.loggers.tensorboard.TensorBoardLogger>
[2025-09-17 01:41:51,706][morered.train][INFO] - Instantiating trainer <pytorch_lightning.Trainer>
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
`Trainer(limit_train_batches=1.0)` was configured so 100% of the batches per epoch will be used..
`Trainer(limit_val_batches=1.0)` was configured so 100% of the batches will be used..
`Trainer(limit_test_batches=1.0)` was configured so 100% of the batches will be used..
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
[2025-09-17 01:41:51,746][morered.train][INFO] - run id b853ddf8-9356-11f0-9121-e8ebd33aa110
[2025-09-17 01:41:51,747][morered.train][INFO] - Trainer profiler: <pytorch_lightning.profilers.advanced.AdvancedProfiler object at 0x7f9c9d8b4cd0>
[2025-09-17 01:41:51,748][morered.train][INFO] - Logging hyperparameters.
[2025-09-17 01:41:52,130][morered.train][INFO] - Starting training.
[2025-09-17 01:41:52,217][morered.datasets.qm9_filtered][INFO] - loaded <class 'schnetpack.data.atoms.ASEAtomsData'> with 133885 molecules
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/usr/local/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(

  | Name    | Type        | Params | Mode 
------------------------------------------------
0 | model   | PairEncoder | 6.4 M  | train
1 | outputs | ModuleList  | 0      | train
------------------------------------------------
6.4 M     Trainable params
0         Non-trainable params
6.4 M     Total params
25.436    Total estimated model params size (MB)
307       Modules in train mode
0         Modules in eval mode
[2025-09-17 01:41:52,450][morered.datasets.qm9_filtered][INFO] - instantion train dataloader with orginal batch size 16 To be augmented'''


'''
svenelzes@hydra:~/MoreRedTransfomer/MoreRed/logs/LONG_TRAIN$ head -n 400 mdet_ddpm-2998302.out 
INFO:    gocryptfs not found, will not be able to use gocryptfs
sys.path: ['/home/svenelzes/MoreRedTransfomer/MoreRed/src', '/home/svenelzes/MoreRedTransfomer/MoreRed/src/scripts', '/usr/local/lib/python310.zip', '/usr/local/lib/python3.10', '/usr/local/lib/python3.10/lib-dynload', '/usr/local/lib/python3.10/site-packages']
src_path exists: True
Contents: ['scripts', 'morered']

Checking /input-data ...
input_data_dir exists: True
Contents of /input-data: ['energy_U0']
/usr/local/lib/python3.10/site-packages/hydra/_internal/config_loader_impl.py:216: UserWarning: provider=hydra.searchpath in main, path=/home/svenelzes/MoreRedTransfomer/MoreRed/configs is not available.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)

░▒▓██████████████▓▒░ ░▒▓██████▓▒ ▒▓███████▓▒░ ▒▓████████▓▒░▒▓███████▓▒░ ▒▓████████▓▒ ▒▓███████▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓███████▓▒░ ▒▓██████▓▒░ ░▒▓███████▓▒░ ▒▓██████▓▒░  ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ░▒▓██████▓▒ ▒▓█▓▒  ▒▓█▓▒ ▒▓████████▓▒░▒▓█▓▒  ▒▓█▓▒ ▒▓████████▓▒ ▒▓███████▓▒░

[2025-09-17 01:40:22,675][morered.train][INFO] - Running on host: head079
⚙ Running with the following config:
├── run
│   └── work_dir: ${hydra:runtime.cwd}                                          
│       data_dir: /input-data/energy_U0                                         
│       path: ${run.work_dir}/runs                                              
│       experiment: md_et_backbone                                              
│       id: ${uuid:1}                                                           
│       ckpt_path: null                                                         
│                                                                               
├── globals
│   └── model_path: best_model                                                  
│       cutoff: 5.0                                                             
│       lr: 0.0001                                                              
│       n_atom_basis: 256                                                       
│       noise_target_key: eps                                                   
│       noise_output_key: eps_pred                                              
│       time_target_key: t                                                      
│       max_atoms:                                                              
│         _target_: morered.utils.get_max_atoms                                 
│       noise_schedule:                                                         
│         _target_: morered.noise_schedules.PolynomialSchedule                  
│         T: 1000                                                               
│         s: 1.0e-05                                                            
│         dtype: float64                                                        
│         variance_type: lower_bound                                            
│       diffusion_process:                                                      
│         _target_: morered.processes.VPGaussianDDPM                            
│         noise_schedule: ${globals.noise_schedule}                             
│         invariant: true                                                       
│         noise_key: ${globals.noise_target_key}                                
│         dtype: float64                                                        
│                                                                               
├── data
│   └── _target_: morered.datasets.QM9Filtered                                  
│       datapath: ${run.data_dir}/qm9.db                                        
│       data_workdir: null                                                      
│       batch_size: 16                                                          
│       num_train: 55000                                                        
│       num_val: 10000                                                          
│       num_test: 10000                                                         
│       num_workers: 6                                                          
│       num_val_workers: null                                                   
│       num_test_workers: null                                                  
│       remove_uncharacterized: false                                           
│       distance_unit: Ang                                                      
│       property_units:                                                         
│         energy_U0: eV                                                         
│         energy_U: eV                                                          
│         enthalpy_H: eV                                                        
│         free_energy: eV                                                       
│         homo: eV                                                              
│         lumo: eV                                                              
│         gap: eV                                                               
│         zpve: eV                                                              
│       n_atoms_allowed: null                                                   
│       shuffle_train: true                                                     
│       permute_indices: false                                                  
│       n_overfit_molecules: null                                               
│       pin_memory: true                                                        
│       indices_path: ${run.data_dir}/n_atoms_indices.pkl                       
│       load_properties:                                                        
│       - energy_U0                                                             
│       transforms:                                                             
│       - _target_: schnetpack.transform.CastTo64                               
│       - _target_: schnetpack.transform.SubtractCenterOfGeometry               
│       - _target_: morered.transform.Diffuse                                   
│         diffuse_property: _positions                                          
│         diffusion_process: ${globals.diffusion_process}                       
│         time_key: ${globals.time_target_key}                                  
│       - _target_: morered.transform.AllToAllNeighborList                      
│       - _target_: schnetpack.transform.CastTo32                               
│                                                                               
├── model
│   └── _target_: morered.model.heads_pe.PairEncoder                            
│       activation: gelu                                                        
│       attention_dropout: 0.1                                                  
│       cls_token: false                                                        
│       decomposer_type: pooling                                                
│       embd_dim: 192                                                           
│       ffn_dropout: 0.0                                                        
│       ffn_multiplier: 4                                                       
│       head_dropout: 0.0                                                       
│       n_layers: 12                                                            
│       norm: layer                                                             
│       norm_first: true                                                        
│       num_3d_kernels: 128                                                     
│       num_heads: 12                                                           
│       head_project_down: true                                                 
│       output_key: ${globals.noise_output_key}                                 
│       include_time: true                                                      
│       target_heads:                                                           
│       - forces                                                                
│       time_head: null                                                         
│       detach_time_head: false                                                 
│       time_key: ${globals.time_target_key}                                    
│                                                                               
├── task
│   └── optimizer_cls: torch.optim.AdamW                                        
│       optimizer_args:                                                         
│         lr: ${globals.lr}                                                     
│         weight_decay: 0.0                                                     
│       scheduler_cls: schnetpack.train.ReduceLROnPlateau                       
│       scheduler_monitor: val_loss                                             
│       scheduler_args:                                                         
│         mode: min                                                             
│         factor: 0.5                                                           
│         patience: 150                                                         
│         threshold: 0.0                                                        
│         threshold_mode: rel                                                   
│         cooldown: 10                                                          
│         min_lr: 0.0                                                           
│         smoothing_factor: 0.0                                                 
│       _target_: morered.DiffusionTask                                         
│       warmup_steps: 0                                                         
│       skip_exploding_batches: true                                            
│       include_l0: false                                                       
│       time_key: ${globals.time_target_key}                                    
│       noise_key: ${globals.noise_target_key}                                  
│       noise_pred_key: ${globals.noise_output_key}                             
│       outputs:                                                                
│       - _target_: morered.task.DiffModelOutput                                
│         name: ${globals.noise_output_key}                                     
│         target_property: ${globals.noise_target_key}                          
│         loss_fn:                                                              
│           _target_: torch.nn.MSELoss                                          
│         metrics:                                                              
│           mse:                                                                
│             _target_: torchmetrics.regression.MeanSquaredError                
│             squared: true                                                     
│         loss_weight: 1.0                                                      
│         nll_metric: null                                                      
│       diffuse_property: _positions                                            
│                                                                               
├── trainer
│   └── _target_: pytorch_lightning.Trainer                                     
│       devices: 1                                                              
│       min_epochs: null                                                        
│       max_epochs: 10000                                                       
│       enable_model_summary: true                                              
│       profiler: null                                                          
│       gradient_clip_val: 0.5                                                  
│       accumulate_grad_batches: 1                                              
│       val_check_interval: 1.0                                                 
│       check_val_every_n_epoch: 1                                              
│       num_sanity_val_steps: 0                                                 
│       fast_dev_run: false                                                     
│       overfit_batches: 0                                                      
│       limit_train_batches: 1.0                                                
│       limit_val_batches: 1.0                                                  
│       limit_test_batches: 1.0                                                 
│       detect_anomaly: false                                                   
│       precision: 32                                                           
│       accelerator: auto                                                       
│       num_nodes: 1                                                            
│       deterministic: false                                                    
│       inference_mode: false                                                   
│                                                                               
├── callbacks
│   └── model_checkpoint:                                                       
│         _target_: schnetpack.train.ModelCheckpoint                            
│         monitor: val_loss                                                     
│         save_top_k: 1                                                         
│         save_last: true                                                       
│         mode: min                                                             
│         verbose: false                                                        
│         dirpath: checkpoints/                                                 
│         filename: '{epoch:02d}'                                               
│         model_path: ${globals.model_path}                                     
│       early_stopping:                                                         
│         _target_: pytorch_lightning.callbacks.EarlyStopping                   
│         monitor: val_loss                                                     
│         patience: 300                                                         
│         mode: min                                                             
│         min_delta: 0.0                                                        
│         check_on_train_epoch_end: false                                       
│         check_finite: false                                                   
│       lr_monitor:                                                             
│         _target_: pytorch_lightning.callbacks.LearningRateMonitor             
│         logging_interval: epoch                                               
│       ema:                                                                    
│         _target_: schnetpack.train.ExponentialMovingAverage                   
│         decay: 0.999                                                          
│       sampling:                                                               
│         _target_: morered.callbacks.SamplerCallback                           
│         sampler: ${sampler}                                                   
│         name: sampling                                                        
│         t: null                                                               
│         max_steps: 200                                                        
│         sample_prior: true                                                    
│         store_path: samples                                                   
│         every_n_batchs: 1                                                     
│         every_n_epochs: 100000                                                
│         start_epoch: 1                                                        
│         log_rmsd: false                                                       
│         log_validity: true                                                    
│         bonds_data_path: null                                                 
│                                                                               
├── logger
│   └── tensorboard:                                                            
│         _target_: pytorch_lightning.loggers.tensorboard.TensorBoardLogger     
│         save_dir: tensorboard/                                                
│         name: default                                                         
│                                                                               
└── seed
    └── None                                                                    
[2025-09-17 01:40:22,796][morered.train][INFO] - Setting float32 matmul precision to <medium>
[2025-09-17 01:40:22,797][morered.train][INFO] - Seed randomly...
[rank: 0] Seed set to 1230474404
[2025-09-17 01:40:22,811][morered.train][INFO] - Instantiating datamodule <morered.datasets.QM9Filtered>
[2025-09-17 01:40:23,462][morered.datasets.qm9_filtered][INFO] - Partitioning dataset with 133885 molecules and train dataset size 55000
[2025-09-17 01:40:23,471][morered.datasets.qm9_filtered][INFO] - Train dataset has 55000 molecules and is of type <class 'schnetpack.data.atoms.ASEAtomsData'>
[2025-09-17 01:40:23,472][morered.datasets.qm9_filtered][INFO] - loaded <class 'schnetpack.data.atoms.ASEAtomsData'> with 133885 molecules
[2025-09-17 01:40:23,473][morered.datasets.qm9_filtered][INFO] - in the datadloader for test
[2025-09-17 01:40:25,026][morered.train][INFO] -  keys of the batch: dict_keys(['_idx', 'energy_U0', '_n_atoms', '_atomic_numbers', '_positions', '_cell', '_pbc', 'original__positions', 'eps', 't', '_idx_i_local', '_idx_j_local', '_offsets', '_idx_m', '_idx_i', '_idx_j', 'mask', '_atomic_numbers_padded', '_positions_padded'])
[2025-09-17 01:40:25,027][morered.train][INFO] - Instantiating model <morered.model.heads_pe.PairEncoder>
[2025-09-17 01:40:26,589][morered.train][INFO] - Instantiating task <morered.DiffusionTask>
/usr/local/lib/python3.10/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'model' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['model'])`.
[2025-09-17 01:40:27,104][morered.train][INFO] - Instantiating callback <schnetpack.train.ModelCheckpoint>
[2025-09-17 01:40:27,115][morered.train][INFO] - Instantiating callback <pytorch_lightning.callbacks.EarlyStopping>
[2025-09-17 01:40:27,118][morered.train][INFO] - Instantiating callback <pytorch_lightning.callbacks.LearningRateMonitor>
[2025-09-17 01:40:27,119][morered.train][INFO] - Instantiating callback <schnetpack.train.ExponentialMovingAverage>
[2025-09-17 01:40:27,121][morered.train][INFO] - Instantiating callback <morered.callbacks.SamplerCallback>
[2025-09-17 01:40:27,145][root][INFO] - sample prior True
[2025-09-17 01:40:27,157][morered.train][INFO] - Instantiating logger <pytorch_lightning.loggers.tensorboard.TensorBoardLogger>
[2025-09-17 01:40:27,162][morered.train][INFO] - Instantiating trainer <pytorch_lightning.Trainer>
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
`Trainer(limit_train_batches=1.0)` was configured so 100% of the batches per epoch will be used..
`Trainer(limit_val_batches=1.0)` was configured so 100% of the batches will be used..
`Trainer(limit_test_batches=1.0)` was configured so 100% of the batches will be used..
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
[2025-09-17 01:40:27,190][morered.train][INFO] - run id 85edbca8-9356-11f0-b8c1-a088c2c68cbc
[2025-09-17 01:40:27,191][morered.train][INFO] - Trainer profiler: <pytorch_lightning.profilers.advanced.AdvancedProfiler object at 0x7f3005ece620>
[2025-09-17 01:40:27,192][morered.train][INFO] - Logging hyperparameters.
[2025-09-17 01:40:27,382][morered.train][INFO] - Starting training.
[2025-09-17 01:40:27,454][morered.datasets.qm9_filtered][INFO] - loaded <class 'schnetpack.data.atoms.ASEAtomsData'> with 133885 molecules
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/usr/local/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(

  | Name    | Type        | Params | Mode 
------------------------------------------------
0 | model   | PairEncoder | 6.3 M  | train
1 | outputs | ModuleList  | 0      | train
------------------------------------------------
6.3 M     Trainable params
0         Non-trainable params
6.3 M     Total params
25.342    Total estimated model params size (MB)
289       Modules in train mode
0         Modules in eval mode'''


#base base

"""
NFO:    gocryptfs not found, will not be able to use gocryptfs
sys.path: ['/home/svenelzes/MoreRedTransfomer/MoreRed/src', '/home/svenelzes/MoreRedTransfomer/MoreRed/src/scripts', '/usr/local/lib/python310.zip', '/usr/local/lib/python3.10', '/usr/local/lib/python3.10/lib-dynload', '/usr/local/lib/python3.10/site-packages']
src_path exists: True
Contents: ['scripts', 'morered']

Checking /input-data ...
input_data_dir exists: True
Contents of /input-data: ['energy_U0']
/usr/local/lib/python3.10/site-packages/hydra/_internal/config_loader_impl.py:216: UserWarning: provider=hydra.searchpath in main, path=/home/svenelzes/MoreRedTransfomer/MoreRed/configs is not available.
  warnings.warn(
/usr/local/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'train': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)

░▒▓██████████████▓▒░ ░▒▓██████▓▒ ▒▓███████▓▒░ ▒▓████████▓▒░▒▓███████▓▒░ ▒▓████████▓▒ ▒▓███████▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓███████▓▒░ ▒▓██████▓▒░ ░▒▓███████▓▒░ ▒▓██████▓▒░  ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒  ▒▓█▓ ▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░      ░▒▓█▓▒  ▒▓█▓▒ ▒▓█▓▒░       ▒▓█▓▒  ▒▓█▓▒░
░▒▓█▓▒  ▒▓█▓▒  ▒▓█▓▒ ░▒▓██████▓▒ ▒▓█▓▒  ▒▓█▓▒ ▒▓████████▓▒░▒▓█▓▒  ▒▓█▓▒ ▒▓████████▓▒ ▒▓███████▓▒░

[2025-09-17 08:44:37,861][morered.train][INFO] - Running on host: head025
⚙ Running with the following config:
├── run
│   └── work_dir: ${hydra:runtime.cwd}                                          
│       data_dir: /input-data/energy_U0                                         
│       path: ${run.work_dir}/runs                                              
│       experiment: vp_gauss_ddpm                                               
│       id: ${uuid:1}                                                           
│       ckpt_path: null                                                         
│                                                                               
├── globals
│   └── model_path: best_model                                                  
│       cutoff: 5.0                                                             
│       lr: 0.0001                                                              
│       n_atom_basis: 256                                                       
│       noise_target_key: eps                                                   
│       noise_output_key: eps_pred                                              
│       time_target_key: t                                                      
│       noise_schedule:                                                         
│         _target_: morered.noise_schedules.PolynomialSchedule                  
│         T: 1000                                                               
│         s: 1.0e-05                                                            
│         dtype: float64                                                        
│         variance_type: lower_bound                                            
│       diffusion_process:                                                      
│         _target_: morered.processes.VPGaussianDDPM                            
│         noise_schedule: ${globals.noise_schedule}                             
│         invariant: true                                                       
│         noise_key: ${globals.noise_target_key}                                
│         dtype: float64                                                        
│                                                                               
├── data
│   └── _target_: morered.datasets.QM9Filtered                                  
│       datapath: ${run.data_dir}/qm9.db                                        
│       data_workdir: null                                                      
│       batch_size: 48                                                          
│       num_train: 55000                                                        
│       num_val: 10000                                                          
│       num_test: 10000                                                         
│       num_workers: 6                                                          
│       num_val_workers: null                                                   
│       num_test_workers: null                                                  
│       remove_uncharacterized: false                                           
│       distance_unit: Ang                                                      
│       property_units:                                                         
│         energy_U0: eV                                                         
│         energy_U: eV                                                          
│         enthalpy_H: eV                                                        
│         free_energy: eV                                                       
│         homo: eV                                                              
│         lumo: eV                                                              
│         gap: eV                                                               
│         zpve: eV                                                              
│       n_atoms_allowed: null                                                   
│       shuffle_train: true                                                     
│       permute_indices: false                                                  
│       n_overfit_molecules: null                                               
│       pin_memory: true                                                        
│       indices_path: ${run.data_dir}/n_atoms_indices.pkl                       
│       load_properties:                                                        
│       - energy_U0                                                             
│       transforms:                                                             
│       - _target_: schnetpack.transform.CastTo64                               
│       - _target_: schnetpack.transform.SubtractCenterOfGeometry               
│       - _target_: morered.transform.Diffuse                                   
│         diffuse_property: _positions                                          
│         diffusion_process: ${globals.diffusion_process}                       
│         time_key: ${globals.time_target_key}                                  
│       - _target_: morered.transform.AllToAllNeighborList                      
│       - _target_: schnetpack.transform.CastTo32                               
│                                                                               
├── model
│   └── representation:                                                         
│         radial_basis:                                                         
│           _target_: schnetpack.nn.radial.GaussianRBF                          
│           n_rbf: 20                                                           
│           cutoff: ${globals.cutoff}                                           
│         _target_: schnetpack.representation.PaiNN                             
│         n_atom_basis: ${globals.n_atom_basis}                                 
│         n_interactions: 3                                                     
│         shared_interactions: false                                            
│         shared_filters: false                                                 
│         cutoff_fn:                                                            
│           _target_: schnetpack.nn.cutoff.CosineCutoff                         
│           cutoff: ${globals.cutoff}                                           
│       _target_: schnetpack.model.NeuralNetworkPotential                       
│       input_modules:                                                          
│       - _target_: schnetpack.atomistic.PairwiseDistances                      
│       output_modules:                                                         
│       - _target_: morered.model.heads.TimeAwareEquivariant                    
│         n_in: ${globals.n_atom_basis}                                         
│         n_hidden: null                                                        
│         n_layers: 3                                                           
│         output_key: ${globals.noise_output_key}                               
│         include_time: true                                                    
│         time_head: null                                                       
│         detach_time_head: false                                               
│         time_key: ${globals.time_target_key}                                  
│       do_postprocessing: true                                                 
│       postprocessors:                                                         
│       - _target_: morered.transform.BatchSubtractCenterOfMass                 
│         name: ${globals.noise_output_key}                                     
│       - _target_: schnetpack.transform.CastTo64                               
│                                                                               
├── task
│   └── optimizer_cls: torch.optim.AdamW                                        
│       optimizer_args:                                                         
│         lr: ${globals.lr}                                                     
│         weight_decay: 0.0                                                     
│       scheduler_cls: schnetpack.train.ReduceLROnPlateau                       
│       scheduler_monitor: val_loss                                             
│       scheduler_args:                                                         
│         mode: min                                                             
│         factor: 0.5                                                           
│         patience: 150                                                         
│         threshold: 0.0                                                        
│         threshold_mode: rel                                                   
│         cooldown: 10                                                          
│         min_lr: 0.0                                                           
│         smoothing_factor: 0.0                                                 
│       _target_: morered.DiffusionTask                                         
│       warmup_steps: 0                                                         
│       skip_exploding_batches: true                                            
│       include_l0: false                                                       
│       time_key: ${globals.time_target_key}                                    
│       noise_key: ${globals.noise_target_key}                                  
│       noise_pred_key: ${globals.noise_output_key}                             
│       outputs:                                                                
│       - _target_: morered.task.DiffModelOutput                                
│         name: ${globals.noise_output_key}                                     
│         target_property: ${globals.noise_target_key}                          
│         loss_fn:                                                              
│           _target_: torch.nn.MSELoss                                          
│         metrics:                                                              
│           mse:                                                                
│             _target_: torchmetrics.regression.MeanSquaredError                
│             squared: true                                                     
│         loss_weight: 1.0                                                      
│         nll_metric: null                                                      
│       diffuse_property: _positions                                            
│                                                                               
├── trainer
│   └── _target_: pytorch_lightning.Trainer                                     
│       devices: 1                                                              
│       min_epochs: null                                                        
│       max_epochs: 10000                                                       
│       enable_model_summary: true                                              
│       profiler: null                                                          
│       gradient_clip_val: 0.5                                                  
│       accumulate_grad_batches: 1                                              
│       val_check_interval: 1.0                                                 
│       check_val_every_n_epoch: 1                                              
│       num_sanity_val_steps: 0                                                 
│       fast_dev_run: false                                                     
│       overfit_batches: 0                                                      
│       limit_train_batches: 1.0                                                
│       limit_val_batches: 1.0                                                  
│       limit_test_batches: 1.0                                                 
│       detect_anomaly: false                                                   
│       precision: 32                                                           
│       accelerator: auto                                                       
│       num_nodes: 1                                                            
│       deterministic: false                                                    
│       inference_mode: false                                                   
│                                                                               
├── callbacks
│   └── model_checkpoint:                                                       
│         _target_: schnetpack.train.ModelCheckpoint                            
│         monitor: val_loss                                                     
│         save_top_k: 1                                                         
│         save_last: true                                                       
│         mode: min                                                             
│         verbose: false                                                        
│         dirpath: checkpoints/                                                 
│         filename: '{epoch:02d}'                                               
│         model_path: ${globals.model_path}                                     
│       early_stopping:                                                         
│         _target_: pytorch_lightning.callbacks.EarlyStopping                   
│         monitor: val_loss                                                     
│         patience: 300                                                         
│         mode: min                                                             
│         min_delta: 0.0                                                        
│         check_on_train_epoch_end: false                                       
│         check_finite: false                                                   
│       lr_monitor:                                                             
│         _target_: pytorch_lightning.callbacks.LearningRateMonitor             
│         logging_interval: epoch                                               
│       ema:                                                                    
│         _target_: schnetpack.train.ExponentialMovingAverage                   
│         decay: 0.999                                                          
│       sampling:                                                               
│         _target_: morered.callbacks.SamplerCallback                           
│         sampler: ${sampler}                                                   
│         name: sampling                                                        
│         t: null                                                               
│         max_steps: null                                                       
│         sample_prior: true                                                    
│         store_path: samples                                                   
│         every_n_batchs: 1                                                     
│         every_n_epochs: 100000                                                
│         start_epoch: 1                                                        
│         log_rmsd: false                                                       
│         log_validity: true                                                    
│         bonds_data_path: null                                                 
│                                                                               
├── logger
│   └── tensorboard:                                                            
│         _target_: pytorch_lightning.loggers.tensorboard.TensorBoardLogger     
│         save_dir: tensorboard/                                                
│         name: default                                                         
│                                                                               
└── seed
    └── None                                                                    
[2025-09-17 08:44:37,964][morered.train][INFO] - Setting float32 matmul precision to <medium>
[2025-09-17 08:44:37,966][morered.train][INFO] - Seed randomly...
[rank: 0] Seed set to 481892371
[2025-09-17 08:44:37,993][morered.train][INFO] - Instantiating datamodule <morered.datasets.QM9Filtered>
[2025-09-17 08:44:38,909][morered.datasets.qm9_filtered][INFO] - Partitioning dataset with 133885 molecules and train dataset size 55000
[2025-09-17 08:44:38,928][morered.datasets.qm9_filtered][INFO] - Train dataset has 55000 molecules and is of type <class 'schnetpack.data.atoms.ASEAtomsData'>
[2025-09-17 08:44:38,930][morered.datasets.qm9_filtered][INFO] - loaded <class 'schnetpack.data.atoms.ASEAtomsData'> with 133885 molecules
[2025-09-17 08:44:38,933][morered.datasets.qm9_filtered][INFO] - in the datadloader for test
[2025-09-17 08:44:40,710][morered.train][INFO] -  keys of the batch: dict_keys(['_idx', 'energy_U0', '_n_atoms', '_atomic_numbers', '_positions', '_cell', '_pbc', 'original__positions', 'eps', 't', '_idx_i_local', '_idx_j_local', '_offsets', '_idx_m', '_idx_i', '_idx_j', 'mask', '_atomic_numbers_padded', '_positions_padded'])
[2025-09-17 08:44:40,714][morered.train][INFO] - Instantiating model <schnetpack.model.NeuralNetworkPotential>
[2025-09-17 08:44:40,783][morered.train][INFO] - Instantiating task <morered.DiffusionTask>
/usr/local/lib/python3.10/site-packages/pytorch_lightning/utilities/parsing.py:210: Attribute 'model' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['model'])`.
[2025-09-17 08:44:40,816][morered.train][INFO] - Instantiating callback <schnetpack.train.ModelCheckpoint>
[2025-09-17 08:44:40,827][morered.train][INFO] - Instantiating callback <pytorch_lightning.callbacks.EarlyStopping>
[2025-09-17 08:44:40,829][morered.train][INFO] - Instantiating callback <pytorch_lightning.callbacks.LearningRateMonitor>
[2025-09-17 08:44:40,831][morered.train][INFO] - Instantiating callback <schnetpack.train.ExponentialMovingAverage>
[2025-09-17 08:44:40,833][morered.train][INFO] - Instantiating callback <morered.callbacks.SamplerCallback>
[2025-09-17 08:44:40,842][root][INFO] - sample prior True
[2025-09-17 08:44:40,857][morered.train][INFO] - Instantiating logger <pytorch_lightning.loggers.tensorboard.TensorBoardLogger>
[2025-09-17 08:44:40,862][morered.train][INFO] - Instantiating trainer <pytorch_lightning.Trainer>
GPU available: True (cuda), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
`Trainer(limit_train_batches=1.0)` was configured so 100% of the batches per epoch will be used..
`Trainer(limit_val_batches=1.0)` was configured so 100% of the batches will be used..
`Trainer(limit_test_batches=1.0)` was configured so 100% of the batches will be used..
`Trainer(val_check_interval=1.0)` was configured so validation will run at the end of the training epoch..
[2025-09-17 08:44:40,895][morered.train][INFO] - run id c984da4c-9391-11f0-9fa8-a088c251a582
[2025-09-17 08:44:40,897][morered.train][INFO] - Trainer profiler: <pytorch_lightning.profilers.advanced.AdvancedProfiler object at 0x7f4c01746950>
[2025-09-17 08:44:40,898][morered.train][INFO] - Logging hyperparameters.
[2025-09-17 08:44:41,130][morered.train][INFO] - Starting training.
[2025-09-17 08:44:41,182][morered.datasets.qm9_filtered][INFO] - loaded <class 'schnetpack.data.atoms.ASEAtomsData'> with 133885 molecules
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/usr/local/lib/python3.10/site-packages/torch/optim/lr_scheduler.py:60: UserWarning: The verbose parameter is deprecated. Please use get_last_lr() to access the learning rate.
  warnings.warn(

  | Name    | Type                   | Params | Mode 
-----------------------------------------------------------
0 | model   | NeuralNetworkPotential | 2.5 M  | train
1 | outputs | ModuleList             | 0      | train
-----------------------------------------------------------
2.5 M     Trainable params
0         Non-trainable params
2.5 M     Total params
10.140    Total estimated model params size (MB)
85        Modules in train mode
0         Modules in eval mode
[2025-09-17 08:44:41,308][morered.datasets.qm9_filtered][INFO] - instantion train dataloader with orginal batch size 48 To be augmented"""

"""
svenelzes@hydra:~/MoreRedTransfomer/MoreRed/runs$ ls -lt --group-directories-first
total 163
drwxrwxr-x 5 svenelzes svenelzes 10 Sep 19 08:38 c7b0d306-9391-11f0-9fa8-a088c251a582  #base base
drwxrwxr-x 5 svenelzes svenelzes  9 Sep 17 01:53 b53a35f4-9356-11f0-9121-e8ebd33aa110  #JT
drwxrwxr-x 5 svenelzes svenelzes  9 Sep 17 01:51 83389cb2-9356-11f0-b8c1-a088c2c68cbc 
drwxrwxr-x 5 svenelzes svenelzes  9 Sep 17 01:40 72f31328-9356-11f0-9999-0a8fc39d33ae
drwxrwxr-x 5 svenelzes svenelzes  9 Sep 16 18:35 7d4dd5ca-9319-11f0-bea4-e8ebd33aa110"""

#inspect my checkpoint files 

"""

def print_dict(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + f"{k}:")
            print_dict(v, indent + 1)
        else:
            print("  " * indent + f"{k}: {v}")

def eval():
    for ckpt_path in checkpoints:
        try:
            # Load checkpoint metadata only (CPU)
            ckpt = torch.load(ckpt_path, map_location="cpu")

            print(f"Checkpoint: {ckpt_path}")
            print("-" * 40)
            
            # Recursively print all keys and values
            if isinstance(ckpt, dict):
                print_dict(ckpt)
            else:
                print(ckpt)

            print("=" * 80)

        except Exception as e:
            print(f"Failed to load {ckpt_path}: {e}")
            print("=" * 80)

"""



import torch

# Hardcoded checkpoint paths
checkpoints = [
    "runs/c7b0d306-9391-11f0-9fa8-a088c251a582/checkpoints/last.ckpt",
    "runs/b53a35f4-9356-11f0-9121-e8ebd33aa110/checkpoints/last.ckpt",
    "runs/83389cb2-9356-11f0-b8c1-a088c2c68cbc/checkpoints/last.ckpt"
]

run_folders = [
    "/home/svenelzes/MoreRedTransfomer/MoreRed/runs/c7b0d306-9391-11f0-9fa8-a088c251a582",
    "/home/svenelzes/MoreRedTransfomer/MoreRed/runs/b53a35f4-9356-11f0-9121-e8ebd33aa110",
    "/home/svenelzes/MoreRedTransfomer/MoreRed/runs/83389cb2-9356-11f0-b8c1-a088c2c68cbc"
]


import logging
import os
import random
import socket
import tempfile
import uuid
from typing import List

import hydra
import numpy as np
import schnetpack as spk
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers.logger import Logger
from schnetpack.utils import str2class
from schnetpack.utils.script import log_hyperparameters, print_config
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from morered.utils import batch_rmsd, check_validity, generate_bonds_data



log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def eval(config: DictConfig):
    """
    General eval routine for all models defined by the provided hydra configs.

    """
 
    log.info("Running on host: " + str(socket.gethostname()))

    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            "Config incomplete! You need to specify the data directory `data_dir`."
        )
        return

    if not ("model" in config and "data" in config):
        log.error(
            """
                Config incomplete! You have to specify at least `data` and `model`!
                For an example, try one of our pre-defined experiments:
                > mrdtrain data_dir=</data/will/be/here> +experiment=<experiment_name>
            """
        )
        return

    if os.path.exists("config.yaml"):
        log.info(
            f"Config already exists in given directory {os.path.abspath('.')}."
            + " Attempting to continue training."
        )

        # save old config
        old_config = OmegaConf.load("config.yaml")
        count = 1
        while os.path.exists(f"config.old.{count}.yaml"):
            count += 1
        with open(f"config.old.{count}.yaml", "w") as f:
            OmegaConf.save(old_config, f, resolve=False)

        # resume from latest checkpoint
        if config.run.ckpt_path is None:
            if os.path.exists("checkpoints/last.ckpt"):
                config.run.ckpt_path = "checkpoints/last.ckpt"

        if config.run.ckpt_path is not None:
            log.info(
                f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
            )
    else:
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f, resolve=False)

    if config.get("print_config"):
        print_config(config, resolve=False)
    if "matmul_precision" in config and config.matmul_precision is not None:
        log.info(f"Setting float32 matmul precision to <{config.matmul_precision}>")
        torch.set_float32_matmul_precision(config.matmul_precision)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.info("Seed randomly...")
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
        seed_everything(seed, workers=True)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)
    
    
    datamodule.prepare_data()
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    log.info(f" keys of the batch: {batch.keys()} and some values {batch['_positions'][0][:5]}")

    base_dataset= datamodule.test_dataloader().dataset



    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)


    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )

    task: spk.AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    best_model_path = os.path.join(config.globals.checkpoint_path, "best_model")
    task.model = torch.load(best_model_path)
    log.info(f"best task model is {task.model}")
    log.info(f"some emdedding weights {task.model.embedding.nuclear_embedding.embedding.weight[:5]}")
    # Init Lightning callbacks
    log.info(f"model in eval is {task.model}")
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    
    profiler = AdvancedProfiler(dirpath="/home/svenelzes/MoreRedTransfomer/MoreRed/profiler", filename="profiler")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        profiler= profiler,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial",
    )
    log.info(f"run id {config.run.id}")
    log.info(f"Trainer profiler: {trainer.profiler}")


    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=task, trainer=trainer)

    # Train the model
    log.info("Skip training in eval")
    #trainer.fit(model=task, datamodule=datamodule, ckpt_path=config.run.ckpt_path)


    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test(model=task, datamodule=datamodule)

    log.info("log RMSD ratio explicitly.")
    sampled_dataset = model(base_dataset)
    rmsd = batch_rmsd(sampled_dataset, base_dataset)
    log.info(f"RMSD is {rmsd}")






