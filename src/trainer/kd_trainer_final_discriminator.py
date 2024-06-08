from base.base_trainer import BaseTrainer
from torch.cuda.amp import autocast
import torch.nn.functional as F
from models import discriminator
from torch.distributed.optim import ZeroRedundancyOptimizer
from utils import *
from typing import Any
import numpy as np
from evaluation import evaluation_model
from tools.compute_metrics import stoi
from augment import Remix, SNRScale
from data.dataloader import load_audio
from tools import AFD
import os
import json


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
                scheduler_D,
                optimizer,
                kd_optimizer,
                loss_weights,
                hop,
                n_fft,
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
                discriminator_model,
                discriminator_optimizer
            ):

        super(KDTrainer, self).__init__(
                                dist,
                                rank,
                                resume,
                                model,
                                train_ds,
                                test_ds,
                                epochs,
                                interval_eval,
                                max_clip_grad_norm,
                                save_model_dir
                            )
        
        self.kd_optimizer = kd_optimizer
        self.teacher_model = teacher_model
        self.kd_weight = kd_weight
        self.criterion_kd_list = criterion_kd_list
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        self.scheduler = scheduler
        self.scheduler_D = scheduler_D

        self.loss_weights = loss_weights
        self.tsb_writer = tsb_writer
        self.n_gpus = n_gpus

        self.n_fft = n_fft
        self.hop = hop

        self.best_loss = float('inf')
        self.best_state = None
        self.epoch_start = 0
        self.save_enhanced_dir = self.save_model_dir + "/enhanced_sample"
        self.data_test_dir = data_test_dir
        self.num_prints = num_prints
        self.logger = logger
        self.model_type = "unet"

        self.discriminator_model = discriminator_model
        self.discriminator_model_optimizer = discriminator_optimizer

        # data augment
        self.remix = remix
        augments = []
        if remix:
            augments.append(Remix())
            self.snr_scaler = SNRScale()
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
            package["disc_model"] = self.discriminator_model.module.state_dict()
        else:
            package["model"] = self.model.state_dict()
        
        if isinstance(self.optimizer, ZeroRedundancyOptimizer):
            package['optimizer'] = self.optimizer.consolidate_state_dict()
            package["disc_optimizer"] = self.discriminator_model_optimizer.consolidate_state_dict()
        else:
            package['optimizer'] = self.optimizer.state_dict()
            package["disc_optimizer"] = self.discriminator_model_optimizer.state_dict()
        
        package['best_state'] = self.best_state
        package['loss'] = self.best_loss
        package['epoch'] = epoch
        tmp_path = os.path.join(self.save_model_dir, "checkpoint.tar")
        torch.save(package, tmp_path)

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
                self.discriminator_model.module.load_state_dict(package['disc_model'])
            else:
                self.model.load_state_dict(package['model'])
                self.discriminator_model.load_state_dict(package(['disc_model']))

                
            self.optimizer.load_state_dict(package['optimizer'])
            self.discriminator_model_optimizer.load_state_dict(package['disc_optimizer'])

            self.epoch_start = package['epoch'] + 1
            self.best_loss = package['loss']
            self.best_state = package['best_state']
    
            if self.rank == 0:
                self.logger.info(f"Model checkpoint loaded. Training will begin at {self.epoch_start} epoch.")
                self.logger.info(f"Load pretrained info: ")
                self.logger.info(f"Best loss: {self.best_loss}")

    def forward_step(self, model, clean, noisy, type):

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
        if type == "teacher":
            with torch.no_grad():
                est_real, est_imag, features = model(noisy_spec)
        else:
            est_real, est_imag, features = model(noisy_spec)

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
            "features": features
        }

    def calculate_se_loss(self, generator_outputs):
        predict_fake_metric = self.discriminator_model(
            generator_outputs["clean_mag"], generator_outputs["est_mag"]
        )
        gen_loss_GAN = F.mse_loss(
            predict_fake_metric.flatten(), generator_outputs["one_labels"].float()
        )
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
            + self.loss_weights[3] * gen_loss_GAN
        )
        return loss   
    
    def calculate_kd_loss(self, student_generator_outputs, teacher_generator_outputs):
        loss = torch.zeros(1, requires_grad=True).cuda()
        for criterion, weight in zip(self.criterion_kd_list, self.kd_weight):
            if isinstance(criterion, AFD):
                path_dir = f'{self.save_model_dir}/attn_weight'
                if not os.path.isdir(path_dir):
                    os.makedirs(path_dir)

                loss_afd, attn = criterion(student_generator_outputs['features'], teacher_generator_outputs['features'])
                loss += loss_afd * weight
                if self.idx_step == 0:
                    file_path = f'{path_dir}/attn_weight_{self.epoch}.json'
                    with open(file_path, 'w') as f:
                        json.dump(attn.tolist(), f)
            else:
                loss += criterion(student_generator_outputs, teacher_generator_outputs) * weight
        
        return loss
    
    def calculate_discriminator_loss(self, generator_outputs):

        length = generator_outputs["est_audio"].size(-1)
        # pesq_score = generator_outputs['pesq_label'] / 4.5
        est_audio_list = list(generator_outputs["est_audio"].detach().cpu().numpy())
        clean_audio_list = list(generator_outputs["clean"].cpu().numpy()[:, :length])
        pesq_score = discriminator.batch_pesq(clean_audio_list, est_audio_list)

        # The calculation of PESQ can be None due to silent part
        if pesq_score is not None:
            predict_enhance_metric = self.discriminator_model(
                generator_outputs["clean_mag"], generator_outputs["est_mag"].detach()
            )
            predict_max_metric = self.discriminator_model(
                generator_outputs["clean_mag"], generator_outputs["clean_mag"]
            )
            discrim_loss_metric = F.mse_loss(predict_max_metric.flatten(), generator_outputs['one_labels']) \
                                + F.mse_loss(predict_enhance_metric.flatten(), pesq_score)
        else:
            discrim_loss_metric = None
        
        return discrim_loss_metric

    def forward_only_teacher_step(self, enhance_audio):
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
    
    def train_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).cuda()
        teacher_pesq_label = batch[4].cuda()
        teacher_pesq_label = teacher_pesq_label.type(torch.float32)

        if any([isinstance(x, AFD) for x in self.criterion_kd_list]):
            # Normalization
            c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
            noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
            noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

            # Runs the forward pass under autocast.
            student_generator_outputs = self.forward_step(self.model, clean, noisy, "student")
            teacher_generator_outputs = self.forward_step(self.teacher_model, clean, noisy, "teacher")
            student_generator_outputs["clean"] = clean
            student_generator_outputs["noisy"] = noisy
            student_generator_outputs["one_labels"] = one_labels
            se_loss = self.calculate_se_loss(student_generator_outputs)
            kd_loss = self.calculate_kd_loss(student_generator_outputs, teacher_generator_outputs)
        else:
            teacher_enhance = batch[2].cuda()
            # Normalization
            c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
            noisy, clean, teacher_enhance = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1), torch.transpose(teacher_enhance, 0, 1)
            noisy, clean, teacher_enhance = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1), torch.transpose(teacher_enhance * c, 0, 1) 

            if self.remix:
                sources = torch.stack([noisy - clean, clean])
                sources = self.augment(sources)
                noise, clean = sources
                noisy = noise + clean

            # teacher_generator_outputs = self.forward_only_teacher_step(teacher_enhance)
            teacher_generator_outputs = self.forward_step(self.teacher_model, clean, noisy, "teacher")

            if self.remix:
                sources = self.snr_scaler([noisy, clean], snr_scale=1.2)
                scaled_noise, clean = sources
                noisy = scaled_noise + clean

            student_generator_outputs = self.forward_step(self.model, clean, noisy, "student")
            
            student_generator_outputs["clean"] = clean
            student_generator_outputs["noisy"] = noisy
            student_generator_outputs["one_labels"] = one_labels
            se_loss = self.calculate_se_loss(student_generator_outputs)
            kd_loss = self.calculate_kd_loss(student_generator_outputs, teacher_generator_outputs)

        # loss is float32 because mse_loss layers autocast to float32.
        assert se_loss.dtype is torch.float32, f"loss's dtype is not torch.float32 but {se_loss.dtype}"

        
        loss = se_loss + kd_loss 

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        if self.kd_optimizer is not None:
            self.kd_optimizer.step()
            self.kd_optimizer.zero_grad()

        # Train Discriminator
        student_generator_outputs['pesq_label'] = teacher_pesq_label
        discriminator_loss = self.calculate_discriminator_loss(student_generator_outputs)
        
        if discriminator_loss is not None:
            discriminator_loss.backward()
            self.discriminator_model_optimizer.step()
            self.discriminator_model_optimizer.zero_grad()
        else:
            discriminator_loss = torch.tensor([0.0])

        # Logging
        # average over devices in ddp
        if self.n_gpus > 1:
            se_loss = self.gather(se_loss).mean()
            kd_loss = self.gather(kd_loss).mean()
            discriminator_loss = self.gather(discriminator_loss).mean()

        return se_loss.item(), kd_loss.item(), discriminator_loss.item()


    def train_epoch(self, epoch) -> None:
        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()

        loss_train = []
        se_loss_train = []
        kd_loss_train = []
        discriminator_loss_train = []
        self.epoch = epoch

        self.logger.info('\n <Epoch>: {} -- Start training '.format(epoch))
        name = f"Train | Epoch {epoch}"
        logprog = LogProgress(self.logger, self.train_ds, updates=self.num_prints, name=name)

        for idx, batch in enumerate(logprog):
            self.idx_step = idx
            se_loss, kd_loss, discriminator_loss = self.train_step(batch)

            total_loss = se_loss + kd_loss
            loss_train.append(total_loss)
            kd_loss_train.append(kd_loss)
            se_loss_train.append(se_loss)
            discriminator_loss_train.append(discriminator_loss)
        
            if self.rank  == 0:
                logprog.update(gen_loss=format(total_loss, ".5f"))


        loss_train = np.mean(loss_train)
        kd_loss_train = np.mean(kd_loss_train)
        se_loss_train = np.mean(se_loss_train)
        discriminator_loss_train = np.mean(discriminator_loss_train)

        template = 'Train loss: {} - kd loss: {} - se loss: {} - discriminator loss: {}'
        info = template.format(loss_train, kd_loss_train, se_loss_train, discriminator_loss_train)
        if self.rank == 0:
            # print("Done epoch {} - {}".format(epoch, info))
            self.logger.info('-' * 70)
            self.logger.info(bold(f"     Epoch {epoch} - Overall Summary Training | {info}"))
            self.tsb_writer.add_scalar("Loss/train", loss_train, epoch)
            self.tsb_writer.add_scalar("Loss/train_kd_loss", kd_loss_train, epoch)
            self.tsb_writer.add_scalar("Loss/train_se_loss", se_loss_train, epoch)  
            self.tsb_writer.add_scalar("Loss/discriminator_loss_train", discriminator_loss_train, epoch)    
            
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

        self.dist.barrier() # see https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
        self.scheduler.step()
        self.scheduler_D.step()


    @torch.no_grad()
    def test_step(self, batch):
        clean = batch[0].cuda()
        noisy = batch[1].cuda()
        one_labels = torch.ones(clean.size(0)).to(self.rank)

        # Normalization
        c = torch.sqrt(noisy.size(-1) / torch.sum((noisy ** 2.0), dim=-1))
        noisy, clean = torch.transpose(noisy, 0, 1), torch.transpose(clean, 0, 1)
        noisy, clean = torch.transpose(noisy * c, 0, 1), torch.transpose(clean * c, 0, 1)

        student_generator_outputs = self.forward_step(self.model, clean, noisy, "student")
        student_generator_outputs["clean"] = clean
        student_generator_outputs["noisy"] = noisy
        student_generator_outputs["one_labels"] = one_labels
        loss = self.calculate_se_loss(student_generator_outputs)

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


    
