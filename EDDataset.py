import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import pathlib
import enum
import pandas as pd
import torchvision.transforms as transforms
import speech2spikes as s2s
import torchaudio.functional as F
import random

def exp_filter(spikes: torch.Tensor, tau: float, dt: float = 1.0):
    B, T = spikes.shape

    alpha = torch.exp(
        torch.tensor(
            -dt / tau,
            device=spikes.device,
            dtype=spikes.dtype
        )
    )

    y = torch.zeros_like(spikes)
    y_prev = torch.zeros(B, device=spikes.device, dtype=spikes.dtype)

    for t in range(spikes.shape[1]):
        y_prev = alpha * y_prev + spikes[:, t]
        y[:, t] = y_prev

    return y

def van_rossum_3d(
        spike_pred: torch.Tensor,
        spike_target: torch.Tensor,
        tau: float = 20.0,
        dt: float = 1.0,
        reduction: str = 'mean',
        normalise: str = "tau"
) -> torch.Tensor:
    assert spike_pred.shape == spike_target.shape

    B, T, N = spike_pred.shape

    f_pred = exp_filter(spike_pred.view(B * N, T), tau, dt).view(B, T, N)
    # f_target = exp_filter(spike_target.view(B * N, T), tau, dt).view(B, T, N)
    f_target = exp_filter(spike_target.reshape(B * N, T), tau, dt).view(B, T, N)

    sq = (f_pred - f_target).pow(2)
    dist = sq.mean(dim=1)

    if normalise == "tau":
        dist = (dt / tau) * dist
    elif normalise == "dt":
        dist = dt * dist
    elif normalise == "none":
        pass
    else:
        raise ValueError("normalise must be 'tau', 'dt', or 'none'")

    if reduction == 'mean':
        return dist.mean()
    elif reduction == 'sum':
        return dist.sum()
    elif reduction == 'none':
        return dist
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")

def van_rossum_loss(
        spike_pred: torch.Tensor,
        spike_target: torch.Tensor,
        tau: float = 20.0,
        dt: float = 1.0,
        reduction: str = 'mean',
        normalise: str = "tau"
) -> torch.Tensor:
    assert spike_pred.shape == spike_target.shape

    if spike_pred.dim() == 3:
        return van_rossum_3d(spike_pred, spike_target, tau, dt, reduction, normalise)

    f_pred = exp_filter(spike_pred, tau, dt)
    f_target = exp_filter(spike_target, tau, dt)

    sq = (f_pred - f_target).pow(2)
    dist = sq.mean(dim=1)

    if normalise == "tau":
        dist = (dt / tau) * dist
    elif normalise == "dt":
        dist = dt * dist
    elif normalise == "none":
        pass
    else:
        raise ValueError("normalise must be 'tau', 'dt', or 'none'")

    if reduction == 'mean':
        return dist.mean()
    elif reduction == 'sum':
        return dist.sum()
    elif reduction == 'none':
        return dist
    else:
        raise ValueError("reduction must be 'mean', 'sum', or 'none'")


class TUTSection(enum.Enum):
    HOME = 'home'
    RESIDENTIAL = 'residential'
    BOTH = 'both'

    # Convert enum to inherent string
    def __str__(self):
        return self.value

class DatasetSplit(enum.Enum):
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    ALL = 'all'

    def __str__(self):
        return self.value

class TUTDataset(Dataset):
    def __init__(self, root, transform=None, sec: TUTSection=TUTSection.HOME, split: DatasetSplit=DatasetSplit.TRAIN):
        self.root = pathlib.Path(root)
        self.transform = transform
        # if sec == TUTSection.ALL:
        #     path = [self.root / "audio" / "home", self.root / "audio" / "residential"]
        #     self.ann_path = [self.root / "meta" / "home", self.root / "meta" / "residential"]
        # else:
        self.audio_path = self.root / "audio" / str(sec)
        self.ann_path = self.root / "meta_full" / str(sec)
        self.file_list = []
        for idx, file in enumerate(sorted(os.listdir(self.audio_path))):
            if split == DatasetSplit.TRAIN and idx % 5 == 0:
                continue
            if split == DatasetSplit.TEST and idx % 5 != 0:
                continue
            if file.endswith('.wav'):
                self.file_list.append(self.audio_path / file)
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_file = self.file_list[idx]
        ann_file = self.ann_path / (audio_file.stem + "_full.ann")

        waveform, sr = torchaudio.load(audio_file)
        if waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono by averaging channels
        waveform = waveform.squeeze(0)  # Assuming mono audio

        ann_df = pd.read_csv(ann_file, sep='\t', header=None, names=['onset', 'offset', 'event_label'],
                             dtype={'onset': float, 'offset': float, 'event_label': str})

        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, torchaudio.transforms.Resample):
                    sr = t.new_freq
                    # print(f"Resampled to {sr} Hz")

        if self.transform:
            waveform = self.transform(waveform)

            # if waveform.shape[1] > 5650:
            #     waveform = waveform[:, :5650]
            # elif waveform.shape[1] < 5650:
            #     pad_amount = 5650 - waveform.shape[1]
            #     waveform = nn.functional.pad(waveform, (0, pad_amount))

            spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True if self.transform else False)
            waveform = waveform.permute(1, 0)
        else:
            spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True if self.transform else False)

        return waveform, spikes


def spike_conversion(ann_df, audio, window_length=0.025, hop_length=0.011608, sr=16000, spectrogram=False):
    if spectrogram:
        num_frames = audio.shape[1]
    else:
        num_frames = 1
    spikes = torch.zeros(num_frames)
    for _, row in ann_df.iterrows():
        onset = row['onset']
        offset = row['offset']
        # Check if either is a NaN
        if pd.isna(onset) or pd.isna(offset):
            continue
        # print(f"Processing event with onset: {onset}, offset: {offset}")
        onset_frame = int((onset / hop_length))
        offset_frame = int((offset / hop_length))
        # if offset_frame - onset_frame > 150:
        #     continue
        # print(onset_frame, offset_frame)
        # if offset_frame - onset_frame > 60:
        #     offset_frame = onset_frame + 60  # Limit max duration to 60 frames
        spikes[onset_frame:offset_frame] = 1.0
        # print(f"Onset: {onset}, Offset: {offset}, Onset frame: {onset_frame}, Offset frame: {offset_frame}")
    return spikes


def spike_conversion_s2s(ann_df, audio, sr=16000):
    spikes = torch.zeros((audio.shape[1]))
    # print(audio.shape)
    for _, row in ann_df.iterrows():
        onset = row['onset']
        offset = row['offset']
        # print(onset, offset)
        if pd.isna(onset) or pd.isna(offset):
            continue
        onset_sample = int(onset * 200)
        offset_sample = int(offset * 200)
        # print(onset_sample, offset_sample)
        spikes[onset_sample:offset_sample] = 1.0
    return spikes

class URBANDataset(Dataset):
    def __init__(self, root, split: DatasetSplit=DatasetSplit.TRAIN, transform=None, random_chunking=False, only_background=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.split = split
        self.audio_path = self.root / "audio" / str(split)
        self.ann_path = self.root / "annotations" / str(split)
        self.file_list = []
        for file in os.listdir(self.audio_path):
            if file.endswith('.wav'):
                self.file_list.append(self.audio_path / file)
        self.file_list.sort()
        self.file_list = self.file_list[:1000]
        self.random_chunking = random_chunking
        self.only_background = only_background

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx, t_idx=False):
        audio_file = self.file_list[idx]
        ann_file = self.ann_path / (audio_file.stem + ".txt")

        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.squeeze(0)  # Assuming mono audio

        ann_df = pd.read_csv(ann_file, sep='\t', header=None, names=['onset', 'offset', 'event_label'], dtype={'onset': float, 'offset': float, 'event_label': str})

        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, torchaudio.transforms.Resample):
                    sr = t.new_freq
                    # print(f"Resampled to {sr} Hz")

        if self.transform:
            # print(waveform.shape)
            if any(isinstance(t, ReshapeTransform) for t in self.transform.transforms):
                for t in self.transform.transforms:
                    if not isinstance(t, ReshapeTransform):
                        waveform = t(waveform)
            else:
                waveform = self.transform(waveform)
            # print(waveform.shape)

            # If there is a MelSpectrogram in the transform
            if any(isinstance(t, torchaudio.transforms.MelSpectrogram) for t in self.transform.transforms):
                if waveform.shape[1] > 313:
                    waveform = waveform[:, :313]
                elif waveform.shape[1] < 313:
                    pad_amount = 313 - waveform.shape[1]
                    waveform = nn.functional.pad(waveform, (0, pad_amount))
            elif any(isinstance(t, ToSpikeTransform) for t in self.transform.transforms):
                cap_amount = int((sr * 10) / 80)
                if waveform.shape[1] > cap_amount:
                    waveform = waveform[:, :cap_amount]
                elif waveform.shape[1] < cap_amount:
                    pad_amount = cap_amount - waveform.shape[1]
                    waveform = nn.functional.pad(waveform, (0, pad_amount))
                spikes = spike_conversion_s2s(ann_df, waveform, sr=sr)
                waveform = waveform.permute(1, 0)
                if not self.random_chunking and not self.only_background:
                    if any(isinstance(t, ReshapeTransform) for t in self.transform.transforms):
                        for t in self.transform.transforms:
                            if isinstance(t, ReshapeTransform):
                                waveform = t(waveform)
                    return waveform, spikes

            spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True, hop_length=512/16000)
            waveform = waveform.permute(1, 0)
        else:
            spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=False)

        waveform_tensor = torch.zeros([10, waveform.shape[0], 100], dtype=waveform.dtype)
        if self.random_chunking:
            for i in range(10):
                lb = random.randint(0, waveform.shape[1]-100)
                waveform_tensor[i, :] = waveform[:, lb:lb+100]

            waveform_tensor = waveform_tensor.permute(0, 2, 1)
            return waveform_tensor, torch.empty([1])

        if self.only_background:
            on_off_pairs = []
            for _, row in ann_df.iterrows():
                on_off_pairs.append((row['onset'], row['offset']))

            on_off_indices = []
            for on, off in on_off_pairs:
                on_idx = int(on * 200)
                off_idx = int(off * 200)
                on_off_indices.append((on_idx, off_idx))

            for on, off in on_off_indices:
                waveform = torch.concat((waveform[:, :on], waveform[:, off:]), dim=1)

            while waveform.shape[1] < 500:
                waveform = torch.concat((waveform, waveform), dim=1)
            waveform = waveform[:, :500]

            if waveform.shape[1] > 500:
                waveform = waveform[:, :500]

            waveform = waveform.permute(1, 0)

        if any(isinstance(t, ReshapeTransform) for t in self.transform.transforms):
            for t in self.transform.transforms:
                if isinstance(t, ReshapeTransform):
                    waveform = t(waveform)

        return waveform, spikes

    def get_file_by_num(self, file_num):
        if file_num < 0 or file_num >= len(self.file_list):
            raise IndexError("File number out of range")
        return self.file_list.index(self.audio_path / f"soundscape_train_bimodal{str(file_num)}.wav")

class DataSEDDataset(Dataset):
    def __init__(self, root, split: DatasetSplit=DatasetSplit.TRAIN, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.split = split
        self.audio_path = self.root / "SED_wav"
        self.ann_path = self.root / "SED_ground_truth" / "Monophonic_sound_detection.csv"
        self.file_list = []
        for idx, file in enumerate(sorted(os.listdir(self.audio_path))):
            if split == DatasetSplit.TRAIN and idx % 5 == 0:
                continue
            if split == DatasetSplit.TEST and idx % 5 != 0:
                continue
            if file.endswith('.wav'):
                self.file_list.append(self.audio_path / file)
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_file = self.file_list[idx]
        # print(audio_file)

        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.squeeze(0)  # Assuming mono audio

        # Load csv skipping row 1
        ann_df = pd.read_csv(
            self.ann_path,
            sep=',',
            names=['sound_name', 'class_name', 'start_perc', 'end_perc', 'start_time', 'end_time', 'event_length'],
            dtype={'sound_name': str, 'class_name': str, 'start_perc': float, 'end_perc': float, 'start_time': float, 'end_time': float, 'event_length': float},
            header=0
        )

        ann_df.drop(columns=['class_name', 'start_perc', 'end_perc', 'event_length'], inplace=True)
        # select records where class_name is the audiofile.wav without self.audio_path
        ann_df = ann_df[ann_df['sound_name'] == audio_file.name]
        ann_df = ann_df[['start_time', 'end_time']]
        ann_df.columns = ['onset', 'offset']

        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, torchaudio.transforms.Resample):
                    sr = t.new_freq

        if self.transform:
            waveform = self.transform(waveform)

            # if waveform.shape[1] > 313:
            #     waveform = waveform[:, :313]
            # elif waveform.shape[1] < 313:
            #     pad_amount = 313 - waveform.shape[1]
            #     waveform = nn.functional.pad(waveform, (0, pad_amount))

            if waveform.dim() == 3:
                waveform = waveform.mean(dim=0)
                if waveform.dim() == 3:
                    waveform = waveform.squeeze(0)

            spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True if self.transform else False)
            waveform = waveform.permute(1, 0)
        else:
            spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True if self.transform else False)

        # print(waveform.shape)

        return waveform, spikes

    def get_file_by_num(self, file_num):
        if file_num < 0 or file_num >= len(self.file_list):
            raise IndexError("File number out of range")
        return self.file_list.index(self.audio_path / f"S-{file_num:04}.wav")


class TUT2017Dataset(Dataset):
    def __init__(self, root, split: DatasetSplit=DatasetSplit.TRAIN, transform=None, spk_args=None, only_class=None):
        self.label_strings = ["babycry", "glassbreak", "gunshot"]

        self.root = pathlib.Path(root)
        self.transform = transform
        self.split = split
        self.audio_path = self.root / ("dev"+str(split)) / "audio"
        self.ann_path = self.root / ("dev"+str(split)) / "meta"
        bc_df = pd.read_csv(self.ann_path / f"event_list_dev{str(split)}_babycry.csv", dtype={"filename": str, "onset": float, "offset": float, "label": str}, delimiter="\t", header=None, names=["filename", "onset", "offset", "label"])
        gb_df = pd.read_csv(self.ann_path / f"event_list_dev{str(split)}_glassbreak.csv", dtype={"filename": str, "onset": float, "offset": float, "label": str}, delimiter="\t", header=None, names=["filename", "onset", "offset", "label"])
        gs_df = pd.read_csv(self.ann_path / f"event_list_dev{str(split)}_gunshot.csv", dtype={"filename": str, "onset": float, "offset": float, "label": str}, delimiter="\t", header=None, names=["filename", "onset", "offset", "label"])
        self.label_df = pd.concat([bc_df, gb_df, gs_df], ignore_index=True)

        if only_class is not None:
            if only_class == 0:
                self.label_df = bc_df
            elif only_class == 1:
                self.label_df = gb_df
            elif only_class == 2:
                self.label_df = gs_df

        self.file_list = []
        for idx, file in enumerate(sorted(os.listdir(self.audio_path))):
            if file.endswith('.wav'):
                if only_class is None:
                    self.file_list.append(self.audio_path / file)
                else:
                    if self.label_strings[only_class] in file:
                        self.file_list.append(self.audio_path / file)
        self.file_list.sort()
        self.spk_args = spk_args

        self.only_class = only_class


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_file = self.file_list[idx]
        ann_df = self.label_df[self.label_df['filename'] == audio_file.name][['onset', 'offset']]
        # print(audio_file.name)

        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.mean(dim=0).squeeze(0)  # Assuming mono audio

        if self.transform is not None:
            for t in self.transform.transforms:
                if isinstance(t, torchaudio.transforms.Resample):
                    sr = t.new_freq

        if self.transform:
            waveform = self.transform(waveform)
            if self.only_class is not None and self.label_strings[self.only_class] not in audio_file.name:
                spikes = torch.zeros(waveform.shape[1])
            else:
                if self.spk_args:
                    spikes = spike_conversion(ann_df, waveform,
                                              sr=sr,
                                              spectrogram=True if self.transform else False,
                                              hop_length=self.spk_args['hop_length'],
                                              window_length=self.spk_args['window_length']
                                              )
                else:
                    spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True if self.transform else False)
            waveform = waveform.permute(1, 0)
        else:
            if self.only_class is not None and self.label_strings[self.only_class] not in audio_file.name:
                spikes = torch.zeros(waveform.shape[1])
            else:
                spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True if self.transform else False)

        return waveform, spikes


class DangMFCC:
    def __init__(self,
                 sample_rate: int,
                 win_sec: float = 0.04,
                 hop_sec: float = 0.02,
                 n_mfcc: int = 20,
                 n_mels: int = 40,
                 n_fft = None
                 ):
        assert n_fft is None or isinstance(n_fft, int)

        self.sr = sample_rate
        self.win_length = int(round(win_sec * sample_rate))
        self.hop_length = int(round(hop_sec * sample_rate))
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft

        if self.n_fft is None:
            self.n_fft = 1
            while self.n_fft < self.win_length:
                self.n_fft *= 2

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sr,
            n_mfcc=self.n_mfcc,
            melkwargs=dict(
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                center=True,
                power=2.0
            )
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mfcc = self.mfcc_transform(waveform)
        delta = F.compute_deltas(mfcc)
        deltadelta = F.compute_deltas(delta)

        feat60 = torch.cat([mfcc, delta, deltadelta], dim=1)

        return feat60


class TUT2017Classes(Dataset):
    def __init__(self, root, split: DatasetSplit=DatasetSplit.TRAIN, transform=None, transform_2=None, no_neg=False, cut_to_label=True, to_100=True):
        self.root = pathlib.Path(root)
        self.transform_1 = transform
        self.transform_2 = transform_2
        self.split = split
        self.audio_path = self.root / ("dev"+str(split)) / "audio"
        self.ann_path = self.root / ("dev"+str(split)) / "meta"
        bc_df = pd.read_csv(self.ann_path / f"event_list_dev{str(split)}_babycry.csv", dtype={"filename": str, "onset": float, "offset": float, "label": str}, delimiter="\t", header=None, names=["filename", "onset", "offset", "label"])
        gb_df = pd.read_csv(self.ann_path / f"event_list_dev{str(split)}_glassbreak.csv", dtype={"filename": str, "onset": float, "offset": float, "label": str}, delimiter="\t", header=None, names=["filename", "onset", "offset", "label"])
        gs_df = pd.read_csv(self.ann_path / f"event_list_dev{str(split)}_gunshot.csv", dtype={"filename": str, "onset": float, "offset": float, "label": str}, delimiter="\t", header=None, names=["filename", "onset", "offset", "label"])
        self.label_df = pd.concat([bc_df, gb_df, gs_df], ignore_index=True)
        self.file_list = []
        for idx, file in enumerate(sorted(os.listdir(self.audio_path))):
            if file.endswith('.wav'):
                self.file_list.append(self.audio_path / file)
        self.file_list.sort()
        # self.file_list = [self.file_list[1], self.file_list[501], self.file_list[1002]]
        self.cutting_down = cut_to_label


        if no_neg:
            # Remove all files from file_list that do not have an event
            # Iterate through label_df and if onset is NaN, then that file does not have an event
            temp_list = []
            for i in range(len(self.file_list)):
                if pd.isna(self.label_df[self.label_df['filename'] == self.file_list[i].name]['onset'].min()):
                    continue
                else:
                    temp_list.append(self.file_list[i])

            self.file_list = list(set(temp_list))

        self.to_100 = to_100


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_file = self.file_list[idx]
        ann_df = self.label_df[self.label_df['filename'] == audio_file.name][['onset', 'offset']]

        waveform, sr = torchaudio.load(audio_file)
        waveform = waveform.mean(dim=0).squeeze(0)  # Assuming mono audio

        def cut_to_label(waveform, ann_df, sr):
            if ann_df.empty:
                return waveform

            hop_length = 0.02

            # if min_sample is NaN, return min_sample and max_sample as -1, and return the entire waveform
            if pd.isna(ann_df['onset'].min()):
                return waveform, -1, -1

            min_sample = int((ann_df['onset'].min() / hop_length)) - 5
            max_sample = int((ann_df['offset'].max() / hop_length))+ 5
            # print(ann_df['onset'].min(), ann_df['offset'].min(), sr, hop_length)
            # print(min_sample, max_sample)
            if min_sample < 0:
                min_sample = 0
            if max_sample > waveform.shape[0]:
                max_sample = waveform.shape[0]

            return waveform[min_sample:max_sample], min_sample, max_sample


        if self.transform_1 is not None:
            for t in self.transform_1.transforms:
                if isinstance(t, torchaudio.transforms.Resample):
                    sr = t.new_freq

        base = waveform.clone()

        if self.transform_1:
            waveform = self.transform_1(base)
            # spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True if self.transform else False)
            waveform = waveform.permute(1, 0)

        if self.transform_2:
            waveform_2 = self.transform_2(base)
            waveform_2 = waveform_2.permute(1, 0)
        else:
            waveform_2 = waveform

        # print(waveform.shape, waveform_2.shape)

        if self.cutting_down:
            waveform_c, onset, offset = cut_to_label(waveform, ann_df, sr)
            waveform_2_c, onset_2, offset_2 = cut_to_label(waveform_2, ann_df, sr)
        else:
            waveform_c = waveform
            waveform_2_c = waveform_2
            onset = 1
            offset = 1

        # print(waveform_c.shape, waveform_2_c.shape)

        if onset == -1:
            spikes = torch.tensor([3])
            rand_sample = random.randint(0, waveform.shape[0] - 100)
            waveform_c = waveform[rand_sample:rand_sample + 100]
            waveform_2_c = waveform_2[rand_sample:rand_sample + 100]
        elif "babycry" in audio_file.name:
            spikes = torch.tensor([0])  # babycry
        elif "glassbreak" in audio_file.name:
            spikes = torch.tensor([1])  # glassbreak
        else:
            spikes = torch.tensor([2])  # gunshot

        # Pad waveform (both sides) to 100 or trim to 100 (from first)
        if self.cutting_down and self.to_100:
            if waveform_c.shape[0] < 100:
                # print("Padding")
                # print(waveform_c.shape)
                diff = 100 - waveform_c.shape[0]
                each_end = diff // 2
                # print(onset)
                # print(each_end)
                if onset-each_end < 0:
                    onset = each_end
                if onset - each_end + 100 > waveform.shape[0]:
                    onset = waveform.shape[0] + each_end - 100
                waveform_c = waveform[onset-each_end:onset-each_end+100]
                waveform_2_c = waveform_2[onset-each_end:onset-each_end+100]

                # assert waveform_c.shape[0] == 100
                # assert waveform_2_c.shape[0] == 100
            elif waveform_c.shape[0] > 100:
                # print("Cutting")
                # print(waveform_c.shape)
                # print(onset)

                if onset + 100 > waveform.shape[0]:
                    temp_onset = waveform.shape[0] - 100
                else:
                    temp_onset = onset

                waveform_c = waveform[temp_onset:temp_onset+100]
                waveform_2_c = waveform_2[temp_onset:temp_onset+100]

            # assert waveform_c.shape[0] == 100
            if waveform_c.shape[0] != 100:
                print(onset, offset)
                print(waveform_c.shape)
            # print(waveform_c.shape, waveform_2_c.shape)
            assert waveform_2_c.shape[0] == 100

        return (waveform_c, waveform_2_c), (spikes, onset, offset)


class UnsqueezeTransform:
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, x):
        return x.unsqueeze(self.dim)

class ToSpikeTransform:
    def __init__(self, num_channels=20):
        self.spk_transform = s2s.S2S()
        self.spk_transform.configure(n_mels=num_channels)

    def __call__(self, x):
        transformed = self.spk_transform([(x, torch.zeros(x.shape[1]))])[0]
        # Convert the [T, 20] output to [T, 40] by splitting the negative and positive parts of the output
        # positive part goes to the first 20 channels, negative part goes to the next 20 channels
        pos_part = torch.clamp(transformed, min=0)
        neg_part = torch.clamp(transformed, max=0).abs()
        catted = torch.cat([pos_part, neg_part], dim=2)
        return catted

    def transform(self, x):
        return self.__call__(x)

class SqueezeTransform:
    def __init__(self, dim=None):
        self.dim = dim

    def __call__(self, x):
        return x.squeeze(self.dim)

    def transform(self, x):
        return self.__call__(x)

class MonofyTransform:
    def __init__(self, dim=0):
        self.dim = dim

    def __call__(self, x):
        # Make a [2, length] waveform mono by averaging the two channels into a [length] waveform
        if x.shape[0] == 1:
            return x.squeeze(0)
        elif x.shape[0] == 2:
            return x.mean(dim=0)
        else:
            raise ValueError("Input waveform must have 1 or 2 channels")

    def transform(self, x):
        return self.__call__(x)

class ReshapeTransform:
    def __init__(self, *shape):
        # *shape is a list of integers
        self.shape = shape

    def __call__(self, x):
        return x.permute(*self.shape)

    def transform(self, x):
        return self.__call__(x)


class BrownianNoise(Dataset):
    def __init__(self, audio_path, transform=None, transform_2=None):
        self.file_list = []
        for idx, file in enumerate(sorted(os.listdir(audio_path))):
            if file.endswith('.wav'):
                self.file_list.append(audio_path / file)
        self.transform_1 = transform
        self.transform_2 = transform_2

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # print(idx)
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        audio_path = self.file_list[idx]
        waveform, sr = torchaudio.load(audio_path)

        if self.transform_1 is not None:
            for t in self.transform_1.transforms:
                if isinstance(t, torchaudio.transforms.Resample):
                    sr = t.new_freq

        base = waveform.clone()

        if self.transform_1:
            waveform = self.transform_1(base)
            # spikes = spike_conversion(ann_df, waveform, sr=sr, spectrogram=True if self.transform else False)
            waveform = waveform.permute(1, 0)

        if self.transform_2:
            waveform_2 = self.transform_2(base)
            waveform_2 = waveform_2.permute(1, 0)
        else:
            waveform_2 = waveform

        return waveform, waveform_2




if __name__ == "__main__":
    home = pathlib.Path("~").expanduser()

    ds_loc = home / "data" / "URBAN-SED"

    a_transform = transforms.Compose([
        torchaudio.transforms.Resample(44100, 16000),
        UnsqueezeTransform(dim=0),
        ToSpikeTransform(num_channels=64),
        SqueezeTransform(0),
        SqueezeTransform(0),
    ])

    ds = URBANDataset(ds_loc, split=DatasetSplit.TRAIN, transform=a_transform, only_background=True)

    sample, _ = ds[0]

    dl = DataLoader(ds, batch_size=2, shuffle=True)

    inputs, targets = next(iter(dl))
    print(sample.shape)
    print(inputs.shape)

    # Combine dimensions 0 and 1
    inputs = inputs.view(inputs.shape[0] * inputs.shape[1], inputs.shape[2], inputs.shape[3])

    print(inputs.shape)

    # Plot raster for each
    import matplotlib.pyplot as plt
    for i in range(len(inputs)):
        plt.imshow(inputs[i].T)
        plt.show()
