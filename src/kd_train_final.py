import os
import torch
import toml
import warnings
import logging
import argparse
from utils import *

from models.unet import UNet64, UNet32
from models.generator import TSCNet
from time import gmtime, strftime
from data import dataloader2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer
from torchinfo import summary
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tools import KDLoss, PearsonCorrelation, AFD

warnings.filterwarnings('ignore')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# global
logger = logging.getLogger(__name__)

def cleanup():
    dist.destroy_process_group()

def setup(rank, world_size):
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.distributed.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank)

def load_state_dict_from_checkpoint(checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        new_state_dict[name] = v
    return new_state_dict

def entry(rank, world_size, config, args):
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    setup(rank, world_size)
    if rank == 0:
        # This should be needed to be reproducible https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
        set_seed(config["main"]["seed"])

    #============================ load config
    epochs = config["main"]["epochs"]
    batch_size = config['dataset_train']['dataloader']['batchsize']
    max_clip_grad_norm = config["main"]["max_clip_grad_norm"]
    interval_eval = config["main"]["interval_eval"]
    resume = config['main']['resume']
    num_prints = config["main"]["num_prints"]
    gradient_accumulation_steps = config["main"]["gradient_accumulation_steps"]
    data_test_dir = config['dataset_test']['path']
    init_lr = config['optimizer']['init_lr']
    gamma = config["scheduler"]["gamma"]
    decay_epoch = config['scheduler']['decay_epoch']
    num_channel = config['model']['num_channel']
    remix = config["main"]["augment"]["remix"]
    remix_snr = config["main"]["augment"]["remix_snr"]
    n_fft = config["feature"]["n_fft"]
    hop = config["feature"]["hop"]
    cut_len = int(config["main"]["cut_len"])
    save_model_dir = os.path.join(config["main"]["save_model_dir"], config["main"]['name'] + '/checkpoints')

    if rank == 0:
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        # Store config file
        config_name = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(' ', '_') + '.toml'
        with open(os.path.join(config["main"]["save_model_dir"], config["main"]['name'] + '/' + config_name), 'w+') as f:
            toml.dump(config, f)
            f.close()

    log_dir = save_model_dir
    logger = get_logger(log_dir=log_dir, log_name="train.log", resume=resume, is_rank0=rank==0)
    logger.info(f"Argument: {args}")

    # Create train dataloader
    train_ds, test_ds = dataloader2.load_data(    
                                        config['dataset_train']['path'], 
                                        batch_size = config['dataset_train']['dataloader']['batchsize'], 
                                        n_cpu = config['dataset_train']['dataloader']['n_worker'], 
                                        cut_len = cut_len, 
                                        rank = rank, 
                                        world_size = world_size,
                                        shuffle = config['dataset_train']['sampler']['shuffle']
                                    )

    logger.info(f"Total iteration through trainset: {len(train_ds)}")
    logger.info(f"Total iteration through testset: {len(test_ds)}")

    # ----- Load student model and teacher model
    model = UNet32(n_channels=num_channel, bilinear=True)
    model = DistributedDataParallel(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    # optimizer
    if config['main']['use_ZeRo']:
        optimizer = ZeroRedundancyOptimizer( 
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=init_lr
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=init_lr)

    if config['main']['criterion']['AFDLoss']: # need to forward through the teacher backbone to get the feature extractor
        state_dict = load_state_dict_from_checkpoint(config['main']['teacher_checkpoint'])
        teacher_model = TSCNet(num_channel=64, num_features=n_fft//2+1).cuda()
        teacher_model.load_state_dict(state_dict)
        teacher_model = DistributedDataParallel(teacher_model.to(rank), device_ids=[rank], find_unused_parameters=True)
    else:
        teacher_model = None

    if rank == 0:
        logger.info(f"---------- Summary for student model")
        summary(model, [(1, 2, cut_len//hop+1, int(n_fft/2)+1)])

    # tensorboard writer
    writer = SummaryWriter(os.path.join(save_model_dir, "tsb_log"))

    loss_weights = []
    loss_weights.append(config["main"]['loss_weights']["ri"])
    loss_weights.append(config["main"]['loss_weights']["mag"])
    loss_weights.append(config["main"]['loss_weights']["time"])
    loss_weights.append(config["main"]['loss_weights']["gan"])

    # criterion
    kd_weight = list(config['main']['criterion']['kd_weight'])

    criterion_kd_list = nn.ModuleList([])
    if config['main']['criterion']['KDLoss']:
        criterion_kd_list.append(KDLoss(weight_ri=config["main"]['loss_weights']["ri"],
                                        weight_mag=config["main"]['loss_weights']["mag"],
                                        weight_time=config["main"]['loss_weights']["time"],
                                        weight_gan=config["main"]['loss_weights']["gan"]))
    
    if config['main']['criterion']['AFDLoss']:
        # student: layer [x1,x2,x3,x4], teacher: layer [x2, x3, x4, x5]
        s_shapes = [(batch_size, 32, 321, 210), (batch_size, 64, 160, 100), (batch_size, 128, 80, 50), (batch_size, 256, 40, 25)]
        t_shapes = [(batch_size, 64, 321, 101), (batch_size, 64, 321, 101), (batch_size, 64, 321, 101), (batch_size, 64, 321, 101)]
        qk_dim = 512
        criterion_kd_list.append(AFD(t_shapes=t_shapes, 
                                    s_shapes=s_shapes, 
                                    qk_dim=qk_dim))

    criterion_kd_list = criterion_kd_list.cuda()
    
    if len(criterion_kd_list) == 0 or (len(criterion_kd_list) == 1 and isinstance(criterion_kd_list[0], KDLoss)):
        kd_optimizer = None
    else:
        kd_optimizer = torch.optim.AdamW(criterion_kd_list.parameters(), lr=init_lr)

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)

    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=gamma)
    
    trainer = trainer_class(
        dist = dist,
        rank = rank,
        resume = resume,
        n_gpus = world_size,
        epochs = epochs,
        batch_size = batch_size,
        model = model,
        teacher_model = teacher_model,
        train_ds = train_ds,
        test_ds = test_ds,
        scheduler = scheduler,
        optimizer = optimizer,
        kd_optimizer = kd_optimizer,
        loss_weights = loss_weights,
        hop = hop,
        n_fft = n_fft,
        interval_eval = interval_eval,
        max_clip_grad_norm = max_clip_grad_norm,
        gradient_accumulation_steps = gradient_accumulation_steps,
        remix = remix,
        remix_snr = remix_snr,
        save_model_dir = save_model_dir,
        data_test_dir = data_test_dir,
        tsb_writer = writer,
        num_prints = num_prints,
        logger = logger,
        kd_weight = kd_weight,
        criterion_kd_list = criterion_kd_list,
    )

    trainer.train()

    cleanup()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, 
                                        type=str, 
                                        help="config file path (defaul: None)", 
                                        default="/home/minhkhanh/Desktop/work/denoiser/CMGAN/src/config.toml")


    args = parser.parse_args()
    config = toml.load(args.config)

    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("GPU list:", available_gpus)
    args.n_gpus = len(available_gpus)
    print("Number of gpu:", args.n_gpus)

    try: 
        mp.spawn(entry,
                args=(args.n_gpus, config, args),
                nprocs=args.n_gpus,
                join=True)
    except KeyboardInterrupt:
        print('Interrupted')
        try: 
            dist.destroy_process_group()  
        except KeyboardInterrupt: 
            os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
    
