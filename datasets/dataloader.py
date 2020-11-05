import os
import glob
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

from utils.audio import Audio


def create_dataloader(hp, train):
    def train_collate_fn(batch):
        target_mag_list = list()
        mixed_mag_list = list()

        for target_mag, mixed_mag in batch:
            target_mag_list.append(target_mag)
            mixed_mag_list.append(mixed_mag)
        target_mag_list = torch.stack(target_mag_list, dim=0)
        mixed_mag_list = torch.stack(mixed_mag_list, dim=0)

        return target_mag_list, mixed_mag_list

    def test_collate_fn(batch):
        return batch

    if train:
        return DataLoader(dataset=VFWSDataset(hp, True),
                          batch_size=hp.train.batch_size,
                          shuffle=True,
                          num_workers=hp.train.num_workers,
                          collate_fn=train_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=None)
    else:
        return DataLoader(dataset=VFWSDataset(hp, False),
                          collate_fn=test_collate_fn,
                          batch_size=1, shuffle=False, num_workers=0)


class VFWSDataset(Dataset):
    def __init__(self, hp, train):
        def find_all(data_dir,file_format):
            return sorted(glob.glob(os.path.join(data_dir, file_format)))
        self.hp = hp
        self.train = train

        self.mixed_dir = hp.data.vfws_dir + 'mixed/train/' if train else hp.data.vfws_dir + 'mixed/test/'
        self.clean_dir = hp.data.vfws_dir + 'clean/train/' if train else hp.data.vfws_dir + 'clean/test/'

        self.target_wav_list = find_all(self.clean_dir, hp.form.target.wav)
        self.mixed_wav_list  = find_all(self.mixed_dir, hp.form.mixed.wav)
        self.target_mag_list = find_all(self.clean_dir, hp.form.target.mag)
        self.mixed_mag_list  = find_all(self.mixed_dir, hp.form.mixed.mag)

        assert len(self.target_wav_list) == len(self.mixed_wav_list) == \
            len(self.target_mag_list) == len(self.mixed_mag_list), "number of training files must match"

        self.audio = Audio(hp)

    def __len__(self):
        return len(self.target_mag_list)

    def __getitem__(self, idx):
        if self.train :  # need to be fast
            target_mag = torch.load(self.target_mag_list[idx])
            mixed_mag = torch.load(self.mixed_mag_list[idx])
            return target_mag, mixed_mag
        else:
            target_wav, _ = librosa.load(self.target_wav_list[idx], self.hp.audio.sample_rate)
            mixed_wav, _ = librosa.load(self.mixed_wav_list[idx], self.hp.audio.sample_rate)
            target_mag, _ = self.wav2magphase(self.target_wav_list[idx])
            mixed_mag, mixed_phase = self.wav2magphase(self.mixed_wav_list[idx])
            target_mag = torch.from_numpy(target_mag)
            mixed_mag = torch.from_numpy(mixed_mag)
            # mixed_phase = torch.from_numpy(mixed_phase)
            return target_wav, mixed_wav, target_mag, mixed_mag, mixed_phase

    def wav2magphase(self, path):
        wav, _ = librosa.load(path, self.hp.audio.sample_rate)
        mag, phase = self.audio.wav2spec(wav)
        return mag, phase
