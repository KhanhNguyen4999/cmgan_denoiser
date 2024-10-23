import os
import torch
import toml
import warnings
import logging
import argparse
from utils import *

from models.unet import UNet16, UNet32, UNet64
from models.generator import TSCNet
from models import discriminator
from models.distiller import Distiller
from time import gmtime, strftime
from data import dataloader2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchinfo import summary
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tools import KDLoss, PearsonCorrelation
from tools.AFD_kullback_leiber import AFD

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

def load_student_model(model_type, n_channels):
    if model_type == 'Unet16':
        model = UNet16(n_channels=n_channels, bilinear=True)
    elif model_type == 'Unet32':
        model = UNet32(n_channels=n_channels, bilinear=True)
    else:
        model = UNet64(n_channels=n_channels, bilinear=True)
    return model

def load_teacher_model(checkpoint_path, n_fft):
    state_dict = load_state_dict_from_checkpoint(checkpoint_path)
    teacher_model = TSCNet(num_channel=64, num_features=n_fft//2+1).cuda()
    teacher_model.load_state_dict(state_dict)
    return teacher_model

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
    forward_teacher = config['main']['criterion']['AKD'] or config['main']['criterion']['AFDLoss'] or config['main']['criterion']['FAKD']
    
    model = load_student_model(model_type=config["main"]["student_model"], n_channels=num_channel)
    model = DistributedDataParallel(model.to(rank), device_ids=[rank], find_unused_parameters=True)

    teacher_model = load_teacher_model(checkpoint_path=config['main']['teacher_checkpoint'], n_fft=n_fft)
    teacher_model = DistributedDataParallel(teacher_model.to(rank), device_ids=[rank], find_unused_parameters=True)

    discriminator_model = discriminator.Discriminator(ndf=16).cuda()
    discriminator_model = DistributedDataParallel(discriminator_model.to(rank), device_ids=[rank], find_unused_parameters=True)

    snr_mixing_discriminator_model = discriminator.SNRMixingDiscriminator(ndf=16).cuda()
    snr_mixing_discriminator_model = DistributedDataParallel(snr_mixing_discriminator_model.to(rank), device_ids=[rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    disc_optimizer = torch.optim.AdamW(discriminator_model.parameters(), lr=2 * init_lr)
    snr_mixing_disc_optimizer = torch.optim.AdamW(snr_mixing_discriminator_model.parameters(), lr=2 * init_lr)
    
    if forward_teacher:
        distiller = Distiller(config)
        distiller = DistributedDataParallel(distiller.to(rank), device_ids=[rank], find_unused_parameters=True)
        distiller_optimizer = torch.optim.AdamW(distiller.parameters(), lr=init_lr)
    elif len(list(config['main']['criterion']['kd_weight'])) > 0:
        distiller = Distiller(config)
        distiller_optimizer = None
    else:
        distiller = None
        distiller_optimizer = None


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

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)
    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=gamma)
    scheduler_D = torch.optim.lr_scheduler.StepLR(disc_optimizer, step_size=decay_epoch, gamma=gamma)
    scheduler_snr_mixing_D = torch.optim.lr_scheduler.StepLR(snr_mixing_disc_optimizer, step_size=decay_epoch, gamma=gamma)
    if forward_teacher:
        scheduler_distiller = torch.optim.lr_scheduler.StepLR(distiller_optimizer, step_size=decay_epoch, gamma=gamma)
    else:
        scheduler_distiller = None
        
    trainer = trainer_class(
        dist = dist,
        rank = rank,
        resume = resume,
        n_gpus = world_size,
        epochs = epochs,
        batch_size = batch_size,
        model = model,
        teacher_model = teacher_model,
        discriminator_model = discriminator_model,
        snr_mixing_discriminator_model = snr_mixing_discriminator_model,

        distiller = distiller,
        train_ds = train_ds,
        test_ds = test_ds,

        scheduler = scheduler,
        scheduler_D = scheduler_D,
        scheduler_distiller = scheduler_distiller,
        scheduler_snr_mixing_D = scheduler_snr_mixing_D,

        optimizer = optimizer,
        distiller_optimizer = distiller_optimizer,
        discriminator_optimizer = disc_optimizer,
        snr_mixing_disc_optimizer = snr_mixing_disc_optimizer,
        
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
        forward_teacher = forward_teacher
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
    
