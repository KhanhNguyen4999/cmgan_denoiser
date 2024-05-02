import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel.distributed import DistributedDataParallel
import torchaudio
from torch.nn import functional as F
import os
from utils import *
import random
from natsort import natsorted

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def load_audio(audio_path, offset, num_frames):
    if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
        out, sr = torchaudio.load(audio_path,
                                frame_offset=offset,
                                num_frames=num_frames or -1)
    else:
        out, sr = torchaudio.load(audio_path, 
                                frame_offset=offset, 
                                num_frames=num_frames)
    return out

class DemandDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, cut_len=16000*2):
        self.cut_len = cut_len
        self.clean_dir = os.path.join(data_dir, 'clean')
        self.noisy_dir = os.path.join(data_dir, 'noisy')
        self.clean_wav_name = os.listdir(self.clean_dir)
        self.clean_wav_name = natsorted(self.clean_wav_name)

    def __len__(self):
        return len(self.clean_wav_name)

    def __getitem__(self, idx):
        clean_file = os.path.join(self.clean_dir, self.clean_wav_name[idx])
        noisy_file = os.path.join(self.noisy_dir, self.clean_wav_name[idx])

        clean_ds, _ = torchaudio.load(clean_file)
        noisy_ds, _ = torchaudio.load(noisy_file)
        clean_ds = clean_ds.squeeze()
        noisy_ds = noisy_ds.squeeze()
        length = len(clean_ds)
        assert length == len(noisy_ds)
        if length < self.cut_len:
            units = self.cut_len // length
            clean_ds_final = []
            noisy_ds_final = []
            for i in range(units):
                clean_ds_final.append(clean_ds)
                noisy_ds_final.append(noisy_ds)
            clean_ds_final.append(clean_ds[: self.cut_len%length])
            noisy_ds_final.append(noisy_ds[: self.cut_len%length])
            clean_ds = torch.cat(clean_ds_final, dim=-1)
            noisy_ds = torch.cat(noisy_ds_final, dim=-1)
        else:
            # randomly cut 2 seconds segment
            wav_start = random.randint(0, length - self.cut_len)
            noisy_ds = noisy_ds[wav_start:wav_start + self.cut_len]
            clean_ds = clean_ds[wav_start:wav_start + self.cut_len]

        return clean_ds, noisy_ds, length

class Audioset:
    def __init__(self, data_dir, cut_len=16000*2):
        """
        files should be a list [(file, length)]
        """
        self.cut_len = cut_len
        self.data_dir = data_dir
        self.clean_dir = os.path.join(data_dir, 'clean')
        self.wav_name = os.listdir(self.clean_dir)
        self.wav_name = natsorted(self.wav_name)
        self.num_examples = []

        examples = 1
        for file in self.wav_name:
            clean_ds, _ = torchaudio.load(os.path.join(data_dir, "clean", file))
            file_length = clean_ds.shape[0]
            if file_length < cut_len:
                examples = 1
            else:
                examples = (file_length-cut_len) // cut_len + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for file, examples in zip(self.wav_name, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.cut_len is not None:
                offset = self.cut_len * index
                num_frames = self.cut_len
            
            clean_audio = load_audio(os.path.join(self.data_dir, "clean", str(file)), offset, num_frames)
            noisy_audio = load_audio(os.path.join(self.data_dir, "noisy", str(file)), offset, num_frames)

            length = clean_audio.shape[-1]
            if length < self.cut_len:
                units = self.cut_len // length
                clean_ds_final = []
                noisy_ds_final = []
                for i in range(units):
                    clean_ds_final.append(clean_audio)
                    noisy_ds_final.append(noisy_audio)
                clean_ds_final.append(clean_audio[:, :self.cut_len%length])
                noisy_ds_final.append(noisy_audio[:, :self.cut_len%length])

                clean_audio = torch.cat(clean_ds_final, dim=-1)
                noisy_audio = torch.cat(noisy_ds_final, dim=-1)
            

            assert clean_audio.shape[-1] == noisy_audio.shape[-1], "dimension difference between clean audio and noisy audio"
            return clean_audio.squeeze(), noisy_audio.squeeze(), len(clean_audio)

def load_data(ds_dir, batch_size, n_cpu, cut_len, rank, world_size):
    torchaudio.set_audio_backend("sox_io")         # in linux
    train_dir = os.path.join(ds_dir, 'train')
    test_dir = os.path.join(ds_dir, 'test')

    train_ds = DemandDataset(train_dir, cut_len)
    test_ds = DemandDataset(test_dir, cut_len)

    sampler = DistributedSampler(dataset=train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataset = torch.utils.data.DataLoader(dataset=train_ds, 
                                                batch_size=batch_size, 
                                                shuffle=False,
                                                drop_last=True, 
                                                num_workers=n_cpu,
                                                sampler=sampler)

    test_dataset = torch.utils.data.DataLoader(dataset=test_ds, 
                                                batch_size=batch_size, 
                                                shuffle=False,
                                                drop_last=False, 
                                                num_workers=n_cpu)

    return train_dataset, test_dataset

def load_data(is_train, data_dir, batch_size, n_cpu, cut_len, rank, world_size, shuffle):
    torchaudio.set_audio_backend("sox_io")         # in linux
    if is_train:
        dataset_ds = DemandDataset(data_dir, cut_len)
    else:
        dataset_ds = Audioset(data_dir, cut_len)

    sampler = DistributedSampler(dataset=dataset_ds, num_replicas=world_size, rank=rank, shuffle=shuffle)

    train_dataset = torch.utils.data.DataLoader(dataset=dataset_ds, 
                                                batch_size=batch_size, 
                                                shuffle= False,
                                                drop_last=False, 
                                                num_workers=n_cpu,
                                                sampler=sampler
                                            )

    return train_dataset