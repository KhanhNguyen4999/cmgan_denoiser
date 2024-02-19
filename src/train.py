from models.generator import TSCNet
from models.unet import UNet
from models.demucs import Demucs
from models import discriminator
import os
from time import gmtime, strftime
from data import dataloader
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer
from utils import *
import toml
from torchinfo import summary
import argparse
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import warnings
import logging
from trainer.trainer import Trainer
from trainer.unet_trainer import UTrainer
warnings.filterwarnings('ignore')

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

def entry(rank, world_size, config):
    # init distributed training
    # os.environ["CUDA_VISIBLE_DEVICES"] = config["main"]["device_ids"]
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    setup(rank, world_size)

    ## load config
    # train
    epochs = config["main"]["epochs"]
    batch_size = config['dataset_train']['dataloader']['batchsize']
    use_amp = config["main"]["use_amp"]
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
    # augment
    remix = config["main"]["augment"]["remix"]


    # feature
    n_fft = config["feature"]["n_fft"]
    hop = config["feature"]["hop"]
    ndf = config["feature"]["ndf"]

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

    # This should be needed to be reproducible https://discuss.pytorch.org/t/setting-seed-in-torch-ddp/126638
    config["main"]["seed"] += rank 
    set_seed(config["main"]["seed"])

    # Create train dataloader
    train_ds = dataloader.load_data(    
                                        True,
                                        config['dataset_train']['path'], 
                                        batch_size = config['dataset_train']['dataloader']['batchsize'], 
                                        n_cpu = config['dataset_train']['dataloader']['n_worker'], 
                                        cut_len = cut_len, 
                                        rank = rank, 
                                        world_size = world_size,
                                        shuffle = config['dataset_train']['sampler']['shuffle']
                                    )

    test_ds = dataloader.load_data (    
                                        False,
                                        config['dataset_valid']['path'], 
                                        batch_size = config['dataset_valid']['dataloader']['batchsize'], 
                                        n_cpu = config['dataset_valid']['dataloader']['n_worker'], 
                                        cut_len = cut_len, 
                                        rank = rank, 
                                        world_size = world_size,
                                        shuffle = config['dataset_valid']['sampler']['shuffle']
                                    )

    logger.info(f"Total iteration through trainset: {len(train_ds)}")
    logger.info(f"Total iteration through testset: {len(test_ds)}")

    # model
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if config['model']['type'] == "unet":
        model = UNet(n_channels=3, bilinear=True)
    else:
        model = TSCNet(num_channel=num_channel, num_features=n_fft // 2 + 1)
        model_discriminator = discriminator.Discriminator(ndf=ndf)
        model_discriminator = DistributedDataParallel(model_discriminator.to(rank), device_ids=[rank])
    
    # Distributed model
    model = DistributedDataParallel(model.to(rank), device_ids=[rank], find_unused_parameters=True)


    if rank == 0:
        summary(model, [(1, 2, cut_len//hop+1, int(n_fft/2)+1)])
        
        if config['model'] == 'tscnet':
            summary(model, [(2, 2, cut_len//hop+1, int(n_fft/2)+1)])
            summary(model_discriminator, [(1, 1, int(n_fft/2)+1, cut_len//hop+1),
                                        (1, 1, int(n_fft/2)+1, cut_len//hop+1)])

    
    # optimizer
    if config['main']['use_ZeRo']:
        optimizer = ZeroRedundancyOptimizer( 
            model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=init_lr
        )
        if config['model']['type'] == 'tscnet':
            optimizer_disc = ZeroRedundancyOptimizer(
                model_discriminator.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=2*init_lr
            )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=init_lr)
        if config['model']['type'] == 'tscnet':
            optimizer_disc = torch.optim.AdamW(model_discriminator.parameters(), 
                                        lr=2*init_lr)

    # tensorboard writer
    writer = SummaryWriter(os.path.join(save_model_dir, "tsb_log"))

    loss_weights = []
    loss_weights.append(config["main"]['loss_weights']["ri"])
    loss_weights.append(config["main"]['loss_weights']["mag"])
    loss_weights.append(config["main"]['loss_weights']["time"])
    loss_weights.append(config["main"]['loss_weights']["gan"])

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)

    if isinstance(trainer_class, Trainer):
        # scheduler
        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=gamma)
        scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_size=decay_epoch, gamma=gamma)

        trainer = trainer_class(
            dist = dist,
            rank = rank,
            resume = resume,
            n_gpus = world_size,
            epochs = epochs,
            batch_size = batch_size,
            model = model,
            discriminator = model_discriminator,
            train_ds = train_ds,
            test_ds = test_ds,
            scheduler_D = scheduler_D,
            scheduler_G = scheduler_G,
            optimizer = optimizer,
            optimizer_disc = optimizer_disc,
            
            loss_weights = loss_weights,

            hop = hop,
            n_fft = n_fft,
            
            scaler = scaler,
            use_amp = use_amp,
            interval_eval = interval_eval,
            max_clip_grad_norm = max_clip_grad_norm,
            gradient_accumulation_steps = gradient_accumulation_steps,

            remix = remix,
            
            save_model_dir = save_model_dir,
            data_test_dir = data_test_dir,
            tsb_writer = writer,
            num_prints = num_prints,
            logger = logger
        )
    else:
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
            train_ds = train_ds,
            test_ds = test_ds,
            scheduler = scheduler,
            optimizer = optimizer,
            
            loss_weights = loss_weights,

            hop = hop,
            n_fft = n_fft,
            
            scaler = scaler,
            use_amp = use_amp,
            interval_eval = interval_eval,
            max_clip_grad_norm = max_clip_grad_norm,
            gradient_accumulation_steps = gradient_accumulation_steps,

            remix = remix,
            
            save_model_dir = save_model_dir,
            data_test_dir = data_test_dir,
            tsb_writer = writer,
            num_prints = num_prints,
            logger = logger
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


    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("GPU list:", available_gpus)
    args.n_gpus = len(available_gpus)
    print("Number of gpu:", args.n_gpus)

    try: 
        mp.spawn(entry,
                args=(args.n_gpus, config),
                nprocs=args.n_gpus,
                join=True)
    except KeyboardInterrupt:
        print('Interrupted')
        try: 
            dist.destroy_process_group()  
        except KeyboardInterrupt: 
            os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
    