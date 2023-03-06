#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runs MNIST training with differential privacy.

"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset

from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm



# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

def get_smaller_train(data, train_size, seed= 42, class_to_remove: int = 8, no_to_remove: int = 5000):

    targets = data.targets
    num_train = len(targets)
    train_idx = np.arange(num_train)
    indices = train_idx

    split = int(np.floor(train_size * num_train))
    np.random.seed(seed)
    np.random.shuffle(indices)
    subtrain_idx, subtargets =  train_idx[indices[:split]], targets[indices[:split]]

    if class_to_remove is None or no_to_remove == -1:
        return subtrain_idx

    total_inclass = len(subtrain_idx[subtargets == class_to_remove])
    print(total_inclass)
    idx_to_remove = np.random.choice(subtrain_idx[subtargets == class_to_remove], total_inclass - no_to_remove, replace=False)
    print(len(idx_to_remove))
    mask1, mask2 = np.zeros(train_idx.shape,dtype=bool), np.zeros(train_idx.shape,dtype=bool)
    mask1[idx_to_remove] = True
    mask2[subtrain_idx] = True
    return train_idx[~mask1 & mask2]

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim = 784, output_dim = 10):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        o = self.linear(x.view(-1,28*28))
        outputs = torch.sigmoid(o)
        #outputs = torch.sigmoid(self.linear(x))
        return outputs

class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"



def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
        return epsilon
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(model, device, test_loader, disparate_target: int = None, test_target = 2):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    target_correct, no_target = 0, 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            ).squeeze()  # get the index of the max log-probability
            if disparate_target is not None:
                wrong_target = - torch.ones(pred.shape).to(device)
                no_target_batch = torch.where(target == test_target, 1, 0).sum().item()
                no_target += no_target_batch
                pred_target = torch.where( pred == test_target, pred, wrong_target)
                target_correct += pred_target.eq(target.view_as(pred_target)).sum().item()
            
            
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    if disparate_target is not None:
        print(
        "\nTest set (disparate): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            target_correct,
            no_target,
            100.0 * target_correct / no_target,
        )
    )
        return target_correct / no_target
    else:
        return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Opacus MNIST Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=512,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "-r",
        "--n-runs",
        type=int,
        default=1,
        metavar="R",
        help="number of runs to average on",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--disparate",
        type= int,
        default= 8,
        help="Disparate target",
    )
    parser.add_argument(
        "--no-to-remove",
        type = int,
        default= 1,
        help= "no of samples to remove"
    )
    parser.add_argument(
        "--seed",
        type = int,
        default= 42,
        help= "random seed number"
    )
    parser.add_argument(
        "--train-size",
        type = float,
        default= 0.01,
        help= "train percentage"
    )
    
    args = parser.parse_args()
    device = torch.device( torch.device(args.device) if torch.cuda.is_available() else 'cpu')

    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         args.data_root,
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    #             ]
    #         ),
    #     ),
    #     batch_size=args.batch_size,
    #     num_workers=0,
    #     pin_memory=True,
    # )

    mnist_train = datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        )


    small = get_smaller_train(mnist_train, args.train_size, 
                            seed= args.seed,
                            class_to_remove= args.disparate, no_to_remove= args.no_to_remove)
    new_set = Subset(mnist_train, small)

    train_loader = torch.utils.data.DataLoader(new_set,
            batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = []
    for _ in range(args.n_runs):
        torch.manual_seed(args.seed)
        model = LogisticRegression().to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng, accountant = 'rdp')
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

        for epoch in range(1, args.epochs + 1):
            eps = train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
            run_results.append((eps, test(model, device, test_loader, disparate_target= args.disparate)))


    repro_str = (
        f"mnist_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_disp_{args.no_to_remove}_trainsize_{args.train_size}_seed_{args.seed}"
    )
    # torch.save(run_results, f"run_results_{repro_str}.pt")
    with open(f"cnn_results_{repro_str}.txt", 'w') as f:
        for i in run_results:
            f.write(f"{i[0]} {i[1]}\n")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()
