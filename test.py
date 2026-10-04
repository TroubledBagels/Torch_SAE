import copy

import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

torch.manual_seed(42)

NUM_INPUTS = 10
NUM_HIDDEN_1 = 32
NUM_HIDDEN_2 = 32
NUM_OUTPUTS = 10

NUM_STEPS = 50
BATCH_SIZE = 1

BETA = 0.99
THRESHOLD = 1.0


# ============================================================
# Common SNN definition
#
# Both networks will contain exactly the same:
#   - Linear weights
#   - Linear biases
#   - LIF parameters
#
# Only the forward timing will be different.
# ============================================================

class BaseSNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(NUM_INPUTS, NUM_HIDDEN_1, bias=True)
        self.lif1 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
        )

        self.fc2 = nn.Linear(NUM_HIDDEN_1, NUM_HIDDEN_2, bias=True)
        self.lif2 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
        )

        self.fc3 = nn.Linear(NUM_HIDDEN_2, NUM_OUTPUTS, bias=True)
        self.lif3 = snn.Leaky(
            beta=BETA,
            threshold=THRESHOLD,
        )


# ============================================================
# Network 1
# Instantaneous / snnTorch-style propagation
#
# A spike can propagate through every layer during the same
# simulation timestep.
# ============================================================

class InstantaneousSNN(BaseSNN):

    def forward(self, x_sequence):

        batch_size = x_sequence.shape[1]

        mem1 = torch.zeros(
            batch_size,
            NUM_HIDDEN_1,
            device=x_sequence.device
        )

        mem2 = torch.zeros(
            batch_size,
            NUM_HIDDEN_2,
            device=x_sequence.device
        )

        mem3 = torch.zeros(
            batch_size,
            NUM_OUTPUTS,
            device=x_sequence.device
        )

        spk1_history = []
        spk2_history = []
        spk3_history = []

        for t in range(x_sequence.shape[0]):

            x = x_sequence[t]

            # -------------------------
            # Layer 1
            # -------------------------
            current1 = self.fc1(x)
            spk1, mem1 = self.lif1(current1, mem1)

            # Layer 1 spike at t is used immediately
            # by layer 2 at t.
            # -------------------------
            # Layer 2
            # -------------------------
            current2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(current2, mem2)

            # Layer 2 spike at t is also immediately
            # used by layer 3 at t.
            # -------------------------
            # Layer 3
            # -------------------------
            current3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(current3, mem3)

            spk1_history.append(spk1)
            spk2_history.append(spk2)
            spk3_history.append(spk3)

        return (
            torch.stack(spk1_history),
            torch.stack(spk2_history),
            torch.stack(spk3_history),
        )


# ============================================================
# Network 2
# Explicit 1-timestep delay between layers
#
# L1 spike at t enters L2 at t+1.
# L2 spike at t enters L3 at t+1.
# ============================================================

class DelayedSNN(BaseSNN):

    def forward(self, x_sequence):

        batch_size = x_sequence.shape[1]
        device = x_sequence.device

        mem1 = torch.zeros(
            batch_size, NUM_HIDDEN_1, device=device
        )

        mem2 = torch.zeros(
            batch_size, NUM_HIDDEN_2, device=device
        )

        mem3 = torch.zeros(
            batch_size, NUM_OUTPUTS, device=device
        )

        delay_1_to_2 = torch.zeros(
            batch_size, NUM_HIDDEN_1, device=device
        )

        delay_2_to_3 = torch.zeros(
            batch_size, NUM_HIDDEN_2, device=device
        )

        spk1_history = []
        spk2_history = []
        spk3_history = []

        for t in range(x_sequence.shape[0]):

            # ==========================================
            # LAYER 1
            # Always has valid input
            # ==========================================

            x = x_sequence[t]

            current1 = self.fc1(x)
            spk1, mem1 = self.lif1(current1, mem1)


            # ==========================================
            # LAYER 2
            #
            # First valid input arrives at t = 1.
            #
            # DO NOT update its membrane at t = 0.
            # ==========================================

            if t >= 1:

                current2 = self.fc2(delay_1_to_2)
                spk2, mem2 = self.lif2(current2, mem2)

            else:

                spk2 = torch.zeros(
                    batch_size,
                    NUM_HIDDEN_2,
                    device=device
                )


            # ==========================================
            # LAYER 3
            #
            # First valid input arrives at t = 2.
            #
            # DO NOT update its membrane before then.
            # ==========================================

            if t >= 2:

                current3 = self.fc3(delay_2_to_3)
                spk3, mem3 = self.lif3(current3, mem3)

            else:

                spk3 = torch.zeros(
                    batch_size,
                    NUM_OUTPUTS,
                    device=device
                )


            # ==========================================
            # Register / delay update
            # ==========================================

            delay_1_to_2 = spk1

            # Only store a VALID layer-2 result.
            if t >= 1:
                delay_2_to_3 = spk2


            spk1_history.append(spk1)
            spk2_history.append(spk2)
            spk3_history.append(spk3)


        return (
            torch.stack(spk1_history),
            torch.stack(spk2_history),
            torch.stack(spk3_history),
        )

# ============================================================
# Create the two networks
# ============================================================

instant_net = InstantaneousSNN()

delayed_net = DelayedSNN()


# ============================================================
# Give delayed_net EXACTLY the same parameters
# ============================================================


delayed_net.fc1.weight = torch.nn.Parameter((delayed_net.fc1.weight + 1) * 10)
delayed_net.fc2.weight = torch.nn.Parameter((delayed_net.fc2.weight + 1) * 10)
delayed_net.fc3.weight = torch.nn.Parameter((delayed_net.fc3.weight + 1) * 10)

delayed_net.load_state_dict(
    copy.deepcopy(instant_net.state_dict())
)

loss_fn = nn.MSELoss()


# Confirm that every parameter is identical.
for (name1, p1), (name2, p2) in zip(
    instant_net.named_parameters(),
    delayed_net.named_parameters(),
):
    assert name1 == name2
    assert torch.equal(p1, p2)

print("Networks have identical parameters.")


# ============================================================
# Generate identical input spike sequence
#
# Shape:
#
# [time, batch, input_neurons]
# ============================================================

torch.manual_seed(124)

input_spikes = (
    torch.rand(
        NUM_STEPS,
        BATCH_SIZE,
        NUM_INPUTS
    ) < 0.4
).float()


# ============================================================
# Simulate both
# ============================================================

with torch.no_grad():

    instant_s1, instant_s2, instant_s3 = instant_net(
        input_spikes
    )

    delayed_s1, delayed_s2, delayed_s3 = delayed_net(
        input_spikes
    )


# Remove batch dimension because batch_size = 1
instant_s1 = instant_s1[:, 0]
instant_s2 = instant_s2[:, 0]
instant_s3 = instant_s3[:, 0]

delayed_s1 = delayed_s1[:, 0]
delayed_s2 = delayed_s2[:, 0]
delayed_s3 = delayed_s3[:, 0]

delayed_loss = loss_fn(torch.zeros_like(delayed_s3), delayed_s3).item()
instant_loss = loss_fn(torch.zeros_like(instant_s3), instant_s3).item()


# ============================================================
# Basic comparison
# ============================================================

print()
print("=" * 60)
print("RAW SPIKE COUNTS")
print("=" * 60)

print(
    "Instantaneous output spikes:",
    instant_s3.sum().item()
)

print(
    "Delayed output spikes:      ",
    delayed_s3.sum().item()
)


# ============================================================
# Temporal alignment
#
# L1:
#     same timing
#
# L2:
#     delayed by 1 timestep
#
# L3:
#     delayed by 2 timesteps
# ============================================================

layer1_equal = torch.equal(
    instant_s1,
    delayed_s1
)

layer2_equal_after_shift = torch.equal(
    instant_s2[:-1],
    delayed_s2[1:]
)

layer3_equal_after_shift = torch.equal(
    instant_s3[:-2],
    delayed_s3[2:]
)


print()
print("=" * 60)
print("TEMPORAL ALIGNMENT")
print("=" * 60)

print(
    f"Layer 1 identical:                {layer1_equal}"
)

print(
    f"Layer 2 identical after +1 shift: {layer2_equal_after_shift}"
)

print(
    f"Layer 3 identical after +2 shift: {layer3_equal_after_shift}"
)

print(f"Instantaneous output loss: {instant_loss:.6f}")
print(f"Delayed output loss: {delayed_loss:.6f}")


# ============================================================
# Difference metric
# ============================================================

raw_difference = torch.abs(
    instant_s3 - delayed_s3
).sum()

aligned_difference = torch.abs(
    instant_s3[:-2] - delayed_s3[2:]
).sum()


print()
print("=" * 60)
print("OUTPUT DIFFERENCE")
print("=" * 60)

print(
    "Difference without correcting for delay:",
    raw_difference.item()
)

print(
    "Difference after correcting for 2-step delay:",
    aligned_difference.item()
)


# ============================================================
# Visualisation
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(13, 9),
    sharex=True
)


def plot_spikes(ax, spikes, title):

    time, neurons = torch.where(spikes > 0)

    ax.scatter(
        time.cpu(),
        neurons.cpu(),
        marker="|",
        s=100
    )

    ax.set_title(title)
    ax.set_ylabel("Neuron")
    ax.grid(alpha=0.2)


# Layer 1
plot_spikes(
    axes[0, 0],
    instant_s1,
    "Instantaneous - Layer 1"
)

plot_spikes(
    axes[0, 1],
    delayed_s1,
    "Delayed - Layer 1"
)


# Layer 2
plot_spikes(
    axes[1, 0],
    instant_s2,
    "Instantaneous - Layer 2"
)

plot_spikes(
    axes[1, 1],
    delayed_s2,
    "Delayed - Layer 2"
)


# Layer 3
plot_spikes(
    axes[2, 0],
    instant_s3,
    "Instantaneous - Output"
)

plot_spikes(
    axes[2, 1],
    delayed_s3,
    "Delayed - Output"
)


axes[2, 0].set_xlabel("Timestep")
axes[2, 1].set_xlabel("Timestep")

plt.tight_layout()
plt.show()


# ============================================================
# Plot output spike count per timestep
# ============================================================

instant_count = instant_s3.sum(dim=1)
delayed_count = delayed_s3.sum(dim=1)

plt.figure(figsize=(11, 4))

plt.step(
    range(NUM_STEPS),
    instant_count.cpu(),
    where="post",
    label="Instantaneous"
)

plt.step(
    range(NUM_STEPS),
    delayed_count.cpu(),
    where="post",
    label="1-step/layer delay"
)

plt.xlabel("Simulation timestep")
plt.ylabel("Output spikes")
plt.title("Output spike timing comparison")
plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()
plt.show()