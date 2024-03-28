from models.generator import TSCNet
from models.unet import UNet
from models.demucs import Demucs
from models import discriminator
import os
from time import gmtime, strftime
from data import dataloader2
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

warnings.filterwarnings('ignore')

# global
logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


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

def entry(world_size, config, args):
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
    teacher_checkpoint = "/home/duynk29/khanhnnm/cmgan_denoiser/src/best_ckpt/ckpt"

    # feature
    n_fft = config["feature"]["n_fft"]
    hop = config["feature"]["hop"]
    ndf = config["feature"]["ndf"]

    cut_len = int(config["main"]["cut_len"])
    save_model_dir = os.path.join(config["main"]["save_model_dir"], config["main"]['name'] + '/checkpoints')

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    # Store config file
    config_name = strftime("%Y-%m-%d %H:%M:%S", gmtime()).replace(' ', '_') + '.toml'
    with open(os.path.join(config["main"]["save_model_dir"], config["main"]['name'] + '/' + config_name), 'w+') as f:
        toml.dump(config, f)
        f.close()

    log_dir = save_model_dir
    logger = get_logger(log_dir=log_dir, log_name="train.log", resume=resume, is_rank0=True)

    # Create train dataloader
    train_ds = dataloader2.load_data(    
                                        True,
                                        config['dataset_train']['path'], 
                                        batch_size = config['dataset_train']['dataloader']['batchsize'], 
                                        n_cpu = config['dataset_train']['dataloader']['n_worker'], 
                                        cut_len = cut_len, 
                                        world_size = world_size,
                                        shuffle = config['dataset_train']['sampler']['shuffle']
                                    )

    test_ds = dataloader2.load_data (    
                                        False,
                                        config['dataset_valid']['path'], 
                                        batch_size = config['dataset_valid']['dataloader']['batchsize'], 
                                        n_cpu = config['dataset_valid']['dataloader']['n_worker'], 
                                        cut_len = cut_len, 
                                        world_size = world_size,
                                        shuffle = config['dataset_valid']['sampler']['shuffle']
                                    )

    logger.info(f"Total iteration through trainset: {len(train_ds)}")
    logger.info(f"Total iteration through testset: {len(test_ds)}")

    # model
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ----- Load student model and teacher model
    model = UNet(n_channels=num_channel, bilinear=True)
    state_dict = load_state_dict_from_checkpoint(teacher_checkpoint)
    teacher_model = TSCNet(num_channel=64, num_features=n_fft//2+1).cuda()
    teacher_model.load_state_dict(state_dict)
    
    logger.info(f"---------- Summary for student model")
    summary(model, [(1, 2, cut_len//hop+1, int(n_fft/2)+1)])
    logger.info(f"----------Summary for Teacher model")
    summary(teacher_model, [(2, 2, cut_len//hop+1, int(n_fft/2)+1)])
    
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
    
    trainer = trainer_class(
        dist = dist,
        rank = 0,
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
        logger = logger,
        kd_args = args
    )

    trainer.train()

    cleanup()



if __name__ == '__main__':

    seed = 42  # Adjust seed value as needed
    set_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=False, 
                                        type=str, 
                                        help="config file path (defaul: None)", 
                                        default="/home/minhkhanh/Desktop/work/denoiser/CMGAN/src/config.toml")


    args = parser.parse_args()
    config = toml.load(args.config)

    # kd argument
    
    args.s_shapes = [(16, 256, 40, 25), (16, 128, 80, 50), (16, 64, 160, 100), (16, 32, 321, 201)]
    args.t_shapes = [(16, 64, 321, 101), (16, 64, 321, 101), (16, 64, 321, 101), (16, 64, 321, 101)]
    args.qk_dim = 512

    print(args)
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("GPU list:", available_gpus)
    args.n_gpus = len(available_gpus)
    print("Number of gpu:", args.n_gpus)

    entry(args.n_gpus, config, args)
    # try: 
    #     mp.spawn(entry,
    #             args=(args.n_gpus, config, args),
    #             nprocs=args.n_gpus,
    #             join=True)
    # except KeyboardInterrupt:
    #     print('Interrupted')
    #     try: 
    #         dist.destroy_process_group()  
    #     except KeyboardInterrupt: 
    #         os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
    