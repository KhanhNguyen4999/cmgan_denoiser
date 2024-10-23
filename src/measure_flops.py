
import torch
from models.unet import UNet16, UNet32, UNet64
from fvcore.nn import FlopCountAnalysis
from models.generator import TSCNet


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
    teacher_model = TSCNet(num_channel=64, num_features=n_fft//2+1)
    teacher_model.load_state_dict(state_dict)
    return teacher_model


if __name__ == "__main__":

    nfft = 400
    hop = 100
    cut_len = 32000
    # model = load_teacher_model("/root/khanhnnm/se/cmgan_denoiser/src/best_ckpt/ckpt", nfft)
    model = load_student_model(model_type="Unet16", n_channels=3)
    input = torch.randn(1, 2, cut_len//hop+1, int(nfft/2)+1)
    flops = FlopCountAnalysis(model, input)
    print(f"FLOPs: {flops.total()/10**9} GB")