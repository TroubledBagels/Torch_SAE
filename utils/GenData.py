import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

class SinWaveDS(data.Dataset):
    def __init__(self, channels, timesteps, wavelength=20, num_samples=1000):
        '''
        :param channels: number of channels in the generated data (i.e. width)
        :param timesteps: length of the data
        :param frequency: number of timesteps for one full sine wave cycle
        :param num_samples: number of samples to generate
        '''
        self.channels = channels
        self.timesteps = timesteps
        self.wavelength = wavelength
        self.num_samples = num_samples
        self.data = self.generate_data()

    def generate_data(self):
        data = torch.zeros((self.num_samples, self.channels, self.timesteps))

        for i in range(self.num_samples):
            phase_shift = torch.rand(1) * 2 * torch.pi

            for t in range(self.timesteps):
                angle = 2 * torch.pi * t / self.wavelength + phase_shift
                s = torch.sin(angle)
                spike_c = ((s + 1) / 2 * (self.channels - 1)).long()
                data[i, spike_c, t] = 1.0

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], 0

if __name__ == '__main__':
    dataset = SinWaveDS(channels=20, timesteps=1000, frequency=50, num_samples=1)

    # Plot raster of the first sample
    plt.figure(figsize=(10, 6))
    plt.imshow(dataset.data[0].numpy(), aspect='auto', cmap='gray', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.title('Raster Plot of First Sample')
    plt.xlabel('Time Steps')
    plt.ylabel('Channels')
    plt.show()