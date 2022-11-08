#! /usr/bin/env python3


from infer import choose_device


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
