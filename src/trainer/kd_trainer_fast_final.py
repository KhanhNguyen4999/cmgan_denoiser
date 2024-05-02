from base.base_trainer import BaseTrainer
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
from utils import *
from typing import Any
import numpy as np
import os
from evaluation import evaluation_model
from tools.compute_metrics import stoi
from augment import Remix
from data.dataloader import load_audio
from tools.contrastive_loss_base_huggingface import ContrastiveLoss


class KDTrainer(BaseTrainer):
    def __init__ (
                self,
                dist,
                rank,
                resume,
                n_gpus,
                epochs,
                batch_size,
                model,
                teacher_model,
                train_ds,
                test_ds,
                scheduler,
                optimizer,
                loss_weights,
                hop,
                n_fft,
                scaler,
                use_amp,
                interval_eval,
                max_clip_grad_norm,
                gradient_accumulation_steps,
                remix,
                save_model_dir,
                data_test_dir,
                tsb_writer,
                num_prints,
                logger,
                kd_weight,
                criterion_kd_list,
            ):

        super(KDTrainer, self).__init__(
                                dist,
                                rank,
                                resume,
                                model,
                                train_ds,
                                test_ds,
                                epochs,
                                use_amp,
                                interval_eval,
                                max_clip_grad_norm,
                                save_model_dir
                            )
        
        self.teacher_model = teacher_model
        self.kd_weight = kd_weight
        self.criterion_kd_list = criterion_kd_list
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.scheduler = scheduler

        self.loss_weights = loss_weights
        self.tsb_writer = tsb_writer
        self.n_gpus = n_gpus

        self.n_fft = n_fft
        self.hop = hop
        self.scaler = scaler

        self.best_loss = float('inf')
        self.best_state = None
        self.epoch_start = 0
        self.save_enhanced_dir = self.save_model_dir + "/enhanced_sample"
        self.data_test_dir = data_test_dir
        self.num_prints = num_prints
        self.logger = logger
        self.model_type = "unet"

        # data augment
        augments = []
        if remix:
            augments.append(Remix())
        self.augment = torch.nn.Sequential(*augments)

        if not os.path.exists(self.save_enhanced_dir):
            os.makedirs(self.save_enhanced_dir)

        if self.resume:
            self.reset()

    def gather(self, value: torch.tensor) -> Any:
        # gather value across devices - https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather
        if value.ndim == 0:
            value = value.clone()[None]

        output_tensors = [value.clone() for _ in range(self.dist.get_world_size())]
        self.dist.all_gather(output_tensors, value)
        return torch.cat(output_tensors, dim=0)

    def serialize(self, epoch):
        '''
        function help to save new general checkpoint
        '''
        package = {}
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            package["model"] = self.model.module.state_dict()
        else:
            package["model"] = self.model.state_dict()
        
        if isinstance(self.optimizer, ZeroRedundancyOptimizer):
            package['optimizer'] = self.optimizer.consolidate_state_dict()
        else:
            package['optimizer'] = self.optimizer.state_dict()
        
        package['best_state'] = self.best_state
        package['loss'] = self.best_loss
        package['epoch'] = epoch
        package['scaler'] = self.scaler
        tmp_path = os.path.join(self.save_model_dir, "checkpoint.tar")
        torch.save(package, tmp_path)

        # Save only the best model, pay attention that don't need to save best discriminator
        # because when infer time, you only need model to predict, and if better the discriminator
        # the worse the model ((:
        model = package['best_state']
        tmp_path = os.path.join(self.save_model_dir, "best.th")
        torch.save(model, tmp_path)

    def load_best(self):
        self.logger.info("<<<<<<<<<<<<<<<<<< Load best loss >>>>>>>>>>>>>>>>>>")
        state_dict = torch.load(self.save_model_dir + "/best.th")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            name = k
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict)

    def reset(self):
        # self.dist.barrier()
        if os.path.exists(self.save_model_dir) and os.path.isfile(self.save_model_dir + "/checkpoint.tar"):
            if self.rank == 0:
                self.logger.info("<<<<<<<<<<<<<<<<<< Load pretrain >>>>>>>>>>>>>>>>>>")
                self.logger.info("Loading last state for resuming training")

            map_location='cuda:{}'.format(self.rank)
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            package = torch.load(self.save_model_dir + "/checkpoint.tar", map_location = map_location)
            
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(package['model'])
            else:
                self.model.load_state_dict(package['model'])
                
            self.optimizer.load_state_dict(package['optimizer'])
            self.epoch_start = package['epoch'] + 1
            self.best_loss = package['loss']
            self.best_state = package['best_state']
            self.scaler = package['scaler']
            if self.rank == 0:
                self.logger.info(f"Model checkpoint loaded. Training will begin at {self.epoch_start} epoch.")
                self.logger.info(f"Load pretrained info: ")
                self.logger.info(f"Best loss: {self.best_loss}")

    def forward_generator_step(self, clean, noisy):

        if len(self.augment) > 0:
            sources = torch.stack([noisy - clean, clean])
            sources = self.augment(sources)
            noise, clean = sources
            noisy = noise + clean

        noisy_spec = torch.view_as_real(torch.stft(noisy, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True,
                                return_complex=True))
        clean_spec = torch.view_as_real(torch.stft(clean, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(),
                                onesided=True,
                                return_complex=True))
        
        noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
        clean_spec = power_compress(clean_spec)
        clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
        clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

    
        est_real, est_imag, _ = self.model(noisy_spec)

        est_real, est_imag = est_real.permute(0, 1, 3, 2), est_imag.permute(0, 1, 3, 2)
        est_mag = torch.sqrt(est_real**2 + est_imag**2)
        clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)

        est_spec_uncompress = power_uncompress(est_real, est_imag).squeeze(1)
        est_audio = torch.istft(est_spec_uncompress, self.n_fft, self.hop,
                                    window=torch.hamming_window(self.n_fft).cuda(), onesided=True)

        return {
            "est_real": est_real,
            "est_imag": est_imag,
            "est_mag": est_mag,
            "clean_real": clean_real,
            "clean_imag": clean_imag,
            "clean_mag": clean_mag,
            "est_audio": est_audio,
        }

    def calculate_generator_loss(self, generator_outputs):

        loss_mag = F.mse_loss(
            generator_outputs["est_mag"], generator_outputs["clean_mag"]
        )
        loss_ri = F.mse_loss(
            generator_outputs["est_real"], generator_outputs["clean_real"]
        ) + F.mse_loss(generator_outputs["est_imag"], generator_outputs["clean_imag"])

        time_loss = torch.mean(
            torch.abs(generator_outputs["est_audio"] - generator_outputs["clean"])
        )

        loss = (
            self.loss_weights[0] * loss_ri
            + self.loss_weights[1] * loss_mag
            + self.loss_weights[2] * time_loss
        )
        return loss
    
    def forward_generator_teacher_step(self, clean, enhance_audio):
        enhance_spec = torch.view_as_real(torch.stft(enhance_audio, self.n_fft, self.hop, window=torch.hamming_window(self.n_fft).cuda(), onesided=True, return_complex=True))
        enhance_spec = power_compress(enhance_spec)
        enhance_real = enhance_spec[:, 0, :, :].unsqueeze(1)
        enhance_imag = enhance_spec[:, 1, :, :].unsqueeze(1)
        enhance_mag = torch.sqrt(enhance_real**2 + enhance_imag**2)

        return {
            "est_real": enhance_real,
            "est_imag": enhance_imag,
            "est_mag": enhance_mag,
            "est_audio": enhance_audio,
        }
    
    def calculate_kd_loss(self, student_generator_outputs, teacher_generator_outputs):
        loss = torch.zeros(1, requires_grad=True).cuda()
        for citerion, weight in zip(self.criterion_kd_list, self.kd_weight):
            loss += citerion(student_generator_outputs, teacher_generator_outputs) * weight
        
        return loss

    def train_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        teacher_enhance = batch[2].cuda()

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean, teacher_enhance = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1), torch.transpose(teacher_enhance, 0, 1)
        noisy, clean, teacher_enhance = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1), torch.transpose(teacher_enhance * c, 0, 1) 

        # Runs the forward pass under autocast.
        with autocast(enabled = self.use_amp):
            student_generator_outputs = self.forward_generator_step(clean, noisy)
            teacher_generator_outputs = self.forward_generator_teacher_step(clean, teacher_enhance)

            student_generator_outputs["clean"] = clean
            student_generator_outputs["noisy"] = noisy

            loss = self.calculate_generator_loss(student_generator_outputs)
            kd_loss = self.calculate_kd_loss(student_generator_outputs, teacher_generator_outputs)

        # loss is float32 because mse_loss layers autocast to float32.
        assert loss.dtype is torch.float32, f"loss's dtype is not torch.float32 but {loss.dtype}"

        total_loss = loss + kd_loss
        self.scaler.scale(total_loss).backward(retain_graph=True)

        # Logging
        # average over devices in ddp
        if self.n_gpus > 1:
            loss = self.gather(loss).mean()

        return loss.item(), kd_loss.item()


    def train_epoch(self, epoch) -> None:
        self.model.train()

        loss_train = []
        se_loss_train = []
        kd_loss_train = []

        self.logger.info('\n <Epoch>: {} -- Start training '.format(epoch))
        name = f"Train | Epoch {epoch}"
        logprog = LogProgress(self.logger, self.train_ds, updates=self.num_prints, name=name)

        for idx, batch in enumerate(logprog):
            se_loss, kd_loss = self.train_step(batch)
            loss = se_loss + kd_loss
            loss_train.append(loss)
            kd_loss_train.append(kd_loss)
            se_loss_train.append(se_loss)
        
            if self.rank  == 0:
                logprog.update(gen_loss=format(loss, ".5f"))
            
            # Optimize step
            if (idx + 1) % self.gradient_accumulation_steps == 0 or idx == len(self.train_ds) - 1:
                
                #gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_clip_grad_norm)

                # update parameters
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                

        loss_train = np.mean(loss_train)
        kd_loss_train = np.mean(kd_loss_train)
        se_loss_train = np.mean(se_loss_train)

        template = 'Train loss: {} - kd loss: {} - se loss: {}'
        info = template.format(loss_train, kd_loss_train, se_loss_train)
        if self.rank == 0:
            # print("Done epoch {} - {}".format(epoch, info))
            self.logger.info('-' * 70)
            self.logger.info(bold(f"     Epoch {epoch} - Overall Summary Training | {info}"))
            self.tsb_writer.add_scalar("Loss/train", loss_train, epoch)
            self.tsb_writer.add_scalar("Loss/train_kd_loss", kd_loss_train, epoch)
            self.tsb_writer.add_scalar("Loss/train_se_loss", se_loss_train, epoch)    
            
        # ------------ Valid stage -------------------
        loss_valid = self.valid_epoch(epoch)

         # save best checkpoint
        if self.rank == 0:
            template = 'Valid loss: {}'
            info = template.format(loss_valid)
            self.logger.info(bold(f"             - Overall Summary Validation | {info}"))
            self.logger.info('-' * 70)
            self.tsb_writer.add_scalar("Loss/valid", loss_valid, epoch)

            self.best_loss = min(self.best_loss, loss_valid)
            if loss_valid == self.best_loss:
                self.best_state = copy_state(self.model.state_dict())

            if epoch % self.interval_eval == 0:
                metrics_avg = evaluation_model(self.model, 
                                self.data_test_dir + "/noisy", 
                                self.data_test_dir + "/clean",
                                True, 
                                self.save_enhanced_dir)

                for metric_type, value in metrics_avg.items():
                    self.tsb_writer.add_scalar(f"metric/{metric_type}", value, epoch)

                info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics_avg.items())
                # print("Evaluation epoch {} -- {}".format(epoch, info))
                self.logger.info(bold(f"     Evaluation Summary:  | Epoch {epoch} | {info}"))

            # Save checkpoint
            self.serialize(epoch)

        # self.dist.barrier() # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
        self.scheduler.step()

    @torch.no_grad()
    def test_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        teacher_enhance = batch[2].cuda()

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean, teacher_enhance = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1), torch.transpose(teacher_enhance, 0, 1)
        noisy, clean, teacher_enhance = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1), torch.transpose(teacher_enhance * c, 0, 1) 

        # Runs the forward pass under autocast.
        with autocast(enabled = self.use_amp):
            student_generator_outputs = self.forward_generator_step(clean, noisy)

            student_generator_outputs["clean"] = clean
            student_generator_outputs["noisy"] = noisy

            loss = self.calculate_generator_loss(student_generator_outputs)

        # loss is float32 because mse_loss layers autocast to float32.
        assert loss.dtype is torch.float32, f"loss's dtype is not torch.float32 but {loss.dtype}"

        # Logging
        # average over devices in ddp
        if self.n_gpus > 1:
            loss = self.gather(loss).mean()

        return loss.item()

    def valid_epoch(self, epoch):

        self.model.eval()

        loss_total = 0.
        for idx, batch in enumerate(self.test_ds):
            step = idx + 1
            loss = self.test_step(batch)
            loss_total += loss

        loss_avg = loss_total / step

        return loss_avg


    