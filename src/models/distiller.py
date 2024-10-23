import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import KDLoss
# from tools.AFD_kullback_leiber import AFD
from tools.AFD_classic import AFD
from tools.AFD_classic_enc_dec import AFDEncDec
from tools.AKD import AKD
from tools.Fusion_AKD import FAKD
from tools.UCLFWPKD import UCLFWPKD

class Distiller(nn.Module):
    def __init__(self, config):
        super(Distiller, self).__init__()

        batch_size = config['dataset_train']['dataloader']['batchsize']
        student_model = config["main"]["student_model"]

        criterion_kd_list = nn.ModuleList([])
        if config['main']['criterion']['KDLoss']:
            criterion_kd_list.append(KDLoss(weight_ri=config["main"]['loss_weights']["ri"],
                                            weight_mag=config["main"]['loss_weights']["mag"],
                                            weight_time=config["main"]['loss_weights']["time"],
                                            weight_gan=config["main"]['loss_weights']["gan"]))
        
        if config['main']['criterion']['AFDLoss'] \
            or config['main']['criterion']['AKD'] \
            or config['main']['criterion']['FAKD']:

            # student: layer [x1,x2,x3,x4], teacher: layer [x2, x3, x4, x5]
            if student_model == 'Unet16':
                s_shapes = [(batch_size, 16, 321, 201), (batch_size, 32, 160, 100), (batch_size, 64, 80, 50), (batch_size, 128, 40, 25)] 
            elif student_model == 'Unet32':
                s_shapes = [(batch_size, 32, 321, 201), (batch_size, 64, 160, 100), (batch_size, 128, 80, 50), (batch_size, 256, 40, 25)]
            else:
                s_shapes = [(batch_size, 64, 321, 201), (batch_size, 128, 160, 100), (batch_size, 256, 80, 50), (batch_size, 512, 40, 25)]

            t_shapes = [(batch_size, 64, 321, 101), (batch_size, 64, 321, 101), (batch_size, 64, 321, 101), (batch_size, 64, 321, 101)]
            
        if config['main']['criterion']['AFDLoss']:
            criterion_kd_list.append(AFD(s_shapes=s_shapes, t_shapes=t_shapes))
        
        if config['main']['criterion']['AKD']:
            criterion_kd_list.append(AKD(s_shapes=s_shapes, t_shapes=t_shapes))

        if config['main']['criterion']['FAKD']:
            criterion_kd_list.append(FAKD(s_shapes=s_shapes, t_shapes=t_shapes))


        self.criterion_kd_list = criterion_kd_list.cuda()
        self.kd_weight = list(config['main']['criterion']['kd_weight'])

    def forward(self, student_generator_outputs, teacher_generator_outputs):
        loss = torch.zeros(1, requires_grad=True).cuda()
        for criterion, weight in zip(self.criterion_kd_list, self.kd_weight):
            if isinstance(criterion, AFD) or isinstance(criterion, AKD) or isinstance(criterion, FAKD):
                loss += criterion(student_generator_outputs['features'], teacher_generator_outputs['features']) * weight
            else:
                loss += criterion(student_generator_outputs, teacher_generator_outputs) * weight
        
        return loss
    

class DistillerStage2(nn.Module):
    def __init__(self, config):
        super(DistillerStage2, self).__init__()

        batch_size = config['dataset_train']['dataloader']['batchsize']
        student_model = config["main"]["student_model"]

        criterion_kd_list = nn.ModuleList([])
        if config['main']['criterion']['KDLoss']:
            criterion_kd_list.append(KDLoss(weight_ri=config["main"]['loss_weights']["ri"],
                                            weight_mag=config["main"]['loss_weights']["mag"],
                                            weight_time=config["main"]['loss_weights']["time"],
                                            weight_gan=config["main"]['loss_weights']["gan"]))
        
        if config['main']['criterion']['AFDLoss']:
            if student_model == 'Unet16':
                s_shapes = [(batch_size, 16, 321, 210), (batch_size, 32, 160, 100), (batch_size, 64, 80, 50), (batch_size, 128, 40, 25), (batch_size, 128, 20, 12)]
            else:
                s_shapes = [(batch_size, 32, 321, 210), (batch_size, 64, 160, 100), (batch_size, 128, 80, 50), (batch_size, 256, 40, 25), (batch_size, 256, 20, 12)]
            
            t_shapes = [(batch_size, 32, 321, 201), (batch_size, 64, 160, 100), (batch_size, 128, 80, 50), (batch_size, 256, 40, 25), (batch_size, 256, 20, 12)]
            # t_shapes = [(batch_size, 64, 321, 201), (batch_size, 128, 160, 100), (batch_size, 256, 80, 50), (batch_size, 512, 40, 25), (batch_size, 512, 20, 12)]

        if config['main']['criterion']['AFDLoss']:
            criterion_kd_list.append(AFD(s_shapes=s_shapes, t_shapes=t_shapes))
        
        if config['main']['criterion']['AKD']:
            criterion_kd_list.append(AKD(s_shapes=s_shapes, t_shapes=t_shapes))

        if config['main']['criterion']['FAKD']:
            criterion_kd_list.append(FAKD(s_shapes=s_shapes, t_shapes=t_shapes))

        if config['main']['criterion']['UCLFWPKD']:
            # student Unet16, teacher Unet64
            if student_model == 'Unet16':
                s_shapes_enc = [(batch_size, 16, 321, 201), (batch_size, 32, 160, 100), (batch_size, 64, 80, 50), (batch_size, 128, 40, 25), (batch_size, 128, 20, 12)]
                s_shapes_dec = [(batch_size, 64, 40, 25), (batch_size, 32, 80, 50), (batch_size, 16, 160, 100), (batch_size, 16, 321, 201)]
            else:
                s_shapes_enc = [(batch_size, 32, 321, 201), (batch_size, 64, 160, 100), (batch_size, 128, 80, 50), (batch_size, 256, 40, 25), (batch_size, 256, 20, 12)]
                s_shapes_dec = [(batch_size, 128, 40, 25), (batch_size, 64, 80, 50), (batch_size, 32, 160, 100), (batch_size, 32, 321, 201)]

            t_shapes_enc = [(batch_size, 64, 321, 201), (batch_size, 128, 160, 100), (batch_size, 256, 80, 50), (batch_size, 512, 40, 25), (batch_size, 512, 20, 12)]
            t_shapes_dec = [(batch_size, 256, 40, 25), (batch_size, 128, 80, 50), (batch_size, 64, 160, 100), (batch_size, 64, 321, 201)]
            criterion_kd_list.append(UCLFWPKD(t_shapes_enc, s_shapes_enc, t_shapes_dec, s_shapes_dec))

        self.criterion_kd_list = criterion_kd_list.cuda()
        self.kd_weight = list(config['main']['criterion']['kd_weight'])

    def forward(self, student_generator_outputs, teacher_generator_outputs):
        loss = torch.zeros(1, requires_grad=True).cuda()
        for criterion, weight in zip(self.criterion_kd_list, self.kd_weight):
            
            if isinstance(criterion, AFD) or isinstance(criterion, AKD) or isinstance(criterion, FAKD):
                loss += criterion(student_generator_outputs['features_enc'], teacher_generator_outputs['features_enc']) * weight
            elif isinstance(criterion, UCLFWPKD):
                enc_stu_fea_list = student_generator_outputs['features_enc']
                dec_stu_fea_list = student_generator_outputs['features_dec']
                enc_tea_fea_list = teacher_generator_outputs['features_enc']
                dec_tea_fea_list = teacher_generator_outputs['features_dec']
                uclfwpkd_loss = criterion(enc_stu_fea_list, enc_tea_fea_list, dec_stu_fea_list, dec_tea_fea_list) * weight
                loss += uclfwpkd_loss
                # print(" --- UCLFWPKD loss: ", uclfwpkd_loss)
            else:
                logit_loss = criterion(student_generator_outputs, teacher_generator_outputs) * weight
                loss += logit_loss
                # print(" ---logit loss: ", logit_loss)
        return loss


class DistillerStagev3(nn.Module):
    def __init__(self, config):
        super(DistillerStagev3, self).__init__()

        batch_size = config['dataset_train']['dataloader']['batchsize']
        student_model = config["main"]["student_model"]

        criterion_kd_list = nn.ModuleList([])
        if config['main']['criterion']['KDLoss']:
            criterion_kd_list.append(KDLoss(weight_ri=config["main"]['loss_weights']["ri"],
                                            weight_mag=config["main"]['loss_weights']["mag"],
                                            weight_time=config["main"]['loss_weights']["time"],
                                            weight_gan=config["main"]['loss_weights']["gan"]))
        
        if config['main']['criterion']['AFDLoss'] or config['main']['criterion']['UCLFWPKD']:
            if student_model == 'Unet16':
                s_shapes_enc = [(batch_size, 16, 321, 201), (batch_size, 32, 160, 100), (batch_size, 64, 80, 50), (batch_size, 128, 40, 25), (batch_size, 128, 20, 12)]
                s_shapes_dec = [(batch_size, 64, 40, 25), (batch_size, 32, 80, 50), (batch_size, 16, 160, 100), (batch_size, 16, 321, 201)]
            else:
                s_shapes_enc = [(batch_size, 32, 321, 201), (batch_size, 64, 160, 100), (batch_size, 128, 80, 50), (batch_size, 256, 40, 25), (batch_size, 256, 20, 12)]
                s_shapes_dec = [(batch_size, 128, 40, 25), (batch_size, 64, 80, 50), (batch_size, 32, 160, 100), (batch_size, 32, 321, 201)]

            t_shapes_enc = [(batch_size, 64, 321, 201), (batch_size, 128, 160, 100), (batch_size, 256, 80, 50), (batch_size, 512, 40, 25), (batch_size, 512, 20, 12)]
            t_shapes_dec = [(batch_size, 256, 40, 25), (batch_size, 128, 80, 50), (batch_size, 64, 160, 100), (batch_size, 64, 321, 201)]
            # t_shapes_enc = [(batch_size, 32, 321, 201), (batch_size, 64, 160, 100), (batch_size, 128, 80, 50), (batch_size, 256, 40, 25), (batch_size, 256, 20, 12)]
            # t_shapes_dec = [(batch_size, 128, 40, 25), (batch_size, 64, 80, 50), (batch_size, 32, 160, 100), (batch_size, 32, 321, 201)]

        if config['main']['criterion']['AFDLoss']:
            criterion_kd_list.append(AFDEncDec(t_shapes_enc, t_shapes_dec, s_shapes_enc, s_shapes_dec))

        if config['main']['criterion']['UCLFWPKD']:
            criterion_kd_list.append(UCLFWPKD(t_shapes_enc, s_shapes_enc, t_shapes_dec, s_shapes_dec))

        self.criterion_kd_list = criterion_kd_list.cuda()
        self.kd_weight = list(config['main']['criterion']['kd_weight'])

    def forward(self, student_generator_outputs, teacher_generator_outputs):
        loss = torch.zeros(1, requires_grad=True).cuda()
        for criterion, weight in zip(self.criterion_kd_list, self.kd_weight):
            
            if isinstance(criterion, AFDEncDec):
                loss += criterion(  teacher_generator_outputs['features_enc'], teacher_generator_outputs['features_dec'],
                                    student_generator_outputs['features_enc'], student_generator_outputs['features_dec']) * weight
            elif isinstance(criterion, UCLFWPKD):
                enc_stu_fea_list = student_generator_outputs['features_enc']
                dec_stu_fea_list = student_generator_outputs['features_dec']
                enc_tea_fea_list = teacher_generator_outputs['features_enc']
                dec_tea_fea_list = teacher_generator_outputs['features_dec']
                uclfwpkd_loss = criterion(enc_stu_fea_list, enc_tea_fea_list, dec_stu_fea_list, dec_tea_fea_list) * weight
                loss += uclfwpkd_loss
                # print(" --- UCLFWPKD loss: ", uclfwpkd_loss)
            else:
                logit_loss = criterion(student_generator_outputs, teacher_generator_outputs) * weight
                loss += logit_loss
                # print(" ---logit loss: ", logit_loss)
        return loss
    

class DistillerStage2Version2(nn.Module):
    def __init__(self, config):
        super(DistillerStage2Version2, self).__init__()

        criterion_kd_list = nn.ModuleList([])
        if config['main']['criterion']['KDLoss']:
            criterion_kd_list.append(KDLoss(weight_ri=config["main"]['loss_weights']["ri"],
                                            weight_mag=config["main"]['loss_weights']["mag"],
                                            weight_time=config["main"]['loss_weights']["time"],
                                            weight_gan=config["main"]['loss_weights']["gan"]))

        self.criterion_kd_list = criterion_kd_list.cuda()
        self.kd_weight = list(config['main']['criterion']['kd_weight'])

    def forward(self, student_generator_outputs, auxiliary_teacher_generator_outputs, teacher_generator_outputs):
        loss = torch.zeros(1, requires_grad=True).cuda()
        for criterion, weight in zip(self.criterion_kd_list, self.kd_weight):
            loss += criterion(student_generator_outputs, teacher_generator_outputs) + criterion(student_generator_outputs, auxiliary_teacher_generator_outputs) 
                # print(" ---logit loss: ", logit_loss)
        return loss