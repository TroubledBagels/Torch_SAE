import torch
import torch.utils.data as data

class SinWaveDS(data.Dataset):
    def __init__(self, channels, time_steps, frequency=20, num_samples=1000):
        '''
        :param channels: number of channels in the generated data (i.e. width)
        :param time_steps: length of the data
        :param frequency: how many timesteps for a full sine wave cycle
        :param num_samples:
        '''
        self.channels = channels
        self.time_steps = time_steps
        self.frequency = frequency
        self.num_samples = num_samples
        self.data = self.generate_data()