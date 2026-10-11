import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import math

class SHDDataset(Dataset):
    def __init__(self, root, transform=None, train=True, t_bin_size=0.001, transpose=True):
        super(SHDDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train

        # Load data from directory
        if train:
            filename = "shd_train.h5"
        else:
            filename = "shd_test.h5"

        # Load h5s
        with h5py.File(root / filename, "r") as f:
            print("Keys: %s" % f.keys())
            self.times = [np.array(t) for t in f["spikes"]["times"]]
            self.units = [np.array(u) for u in f["spikes"]["units"]]
            self.labels = torch.as_tensor(np.array(list(f["labels"])))

        print(len(self.times), len(self.units), len(self.labels))

        if self.train:
            tensor_time = math.ceil((1.369 / t_bin_size) + 1)
        else:
            tensor_time = math.ceil((1.17 / t_bin_size) + 1)

        self.data = torch.zeros((len(self.times), tensor_time, 700), dtype=torch.uint8)

        for i, (times, units) in enumerate(zip(self.times, self.units)):
            time_bins = np.floor(times / t_bin_size).astype(np.int64)

            # Ensure indices fall inside the tensor
            time_bins = np.clip(time_bins, 0, tensor_time - 1)

            self.data[i, time_bins, units.astype(np.int64)] = 1

        if transpose:
            self.data = self.data.permute(0, 2, 1)

        '''
        self.times is an array of floats containing the time in seconds of their corresponding spike
        self.units is an array of integers containing the input neuron that the spike detailed in timings is in
        '''

        '''
        Structure:
          1. ["extra", "labels", "spikes"]
          2. [], [int(class)], ["times", "units"]
          3. [], _, , [float(time)] [int(inp_neuron)]
        '''

    def __getitem__(self, idx):
        spikes = self.data[idx]
        label = self.labels[idx]

        if self.transform is not None:
            spikes = self.transform(spikes)

        return spikes.to(torch.float32), label

    def __len__(self):
        return self.data.shape[0]

    def shape(self):
        return self.data.shape

if __name__ == "__main__":
    import pathlib
    home = pathlib.Path("~").expanduser()
    local_dir = home / "data" / "shd" / "SHD"

    ds = SHDDataset(local_dir, train=True)
    print(ds.shape())

    print(torch.sum(next(iter(ds))[0]))