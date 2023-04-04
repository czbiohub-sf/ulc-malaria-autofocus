#! /usr/bin/env python3


import numpy as np

from autofocus.infer import choose_device


def get_confusion_data(net, dataset, sample_size=100):
    device = choose_device()
    prev_net_state = net.training

    net.eval()

    outputs, std_devs = [], []
    classes_in_order = sorted([int(c) for c in dataset.classes])
    for clss in classes_in_order:
        samples = dataset.sample_from_class(int(clss), sample_size).to(device)
        with torch.no_grad():
            out = net(samples)
            output_stddev, output_mean = torch.std_mean(out, unbiased=True)
        outputs.append(output_mean.item())
        std_devs.append(output_stddev.item())

    net.train(prev_net_state)

    return classes_in_order, outputs, std_devs


def get_allan_deviation(data):
    # Number of output datapoints in log space
    nlog2 = np.floor(np.log(len(data), 2))

    # Vector containing the numbers of re-sampled raw datapoints at each
    # downsampling interval
    n_bins = 2 ** np.arange(1, nlog2 + 1)

    allen_deviation = np.zeros(nlog2)

    # Vector of integers from 1 to the number of raw datapoints
    ints = np.arange(1, len(data) + 1)

    # Loop through each downsampling interval
    for i in range(nlog2):
        # subs are indices into the raw data. All raw data sharing the same
        # subs value will be averaged together.
        subs = np.ceil((n_bins[i] / n) * ints)

        # accumarray sums all raw datapoints with the same subs values. We
        # divide by the number of samples at each subs value to generate a mean
        # rather than a sum. The sqrt, diff, and .^2 computes the RMS of the
        # resulting downsampled data.
        allen_deviation[i] = np.sqrt(
            0.5 * np.mean(diff(accumarray(subs, data) / accumarray(subs, 1)) ** 2)
        )
