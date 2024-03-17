import argparse
import json
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import utils
from model import Net
from sagemaker.experiments import load_run

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py
def _get_data_loader(batch_size, dir, data_name, **kwargs):
    logger.info("Get test data loader")
    data_tensor = torch.load(os.path.join(dir, data_name))
    dataset = torch.utils.data.TensorDataset(data_tensor[0], data_tensor[1])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )


def prepare_dataloader(args):
    kwargs = (
        {"num_workers": args.num_workers, "pin_memory": True} if args.use_cuda else {}
    )
    train_loader = _get_data_loader(
        args.batch_size, args.data_dir, "training.pt", **kwargs
    )
    test_loader = _get_data_loader(
        args.test_batch_size, args.data_dir, "test.pt", **kwargs
    )
    logger.info(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.info(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )
    return train_loader, test_loader


def train(args, run=None):
    # set the device to be used for training
    args.use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)

    # set the loader, model, and optimizer
    train_loader, test_loader = prepare_dataloader(args)
    model = Net().to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.sampler),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader, args.device, epoch, run)
        save_model(model, args.out_dir, args.device, "model_epoch_{}.pth".format(epoch))
    save_model(model, args.model_dir, args.device, "model_last.pth")


def test(model, test_loader, device, epoch, run=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, size_average=False
            ).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[
                1
            ]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # if run:
            run.log_confusion_matrix(
                target.cpu(), pred.cpu(), "Confusion-Matrix-Test-Data"
            )

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * accuracy,
        )
    )
    if run:
        run.log_metric(name="test:loss", value=test_loss, step=epoch)
        run.log_metric(name="test:accuracy", value=accuracy, step=epoch)
    else:
        utils.log(
            {"epoch": epoch, "test_loss": test_loss, "accuracy": accuracy},
            os.path.join(args.out_dir, "metrics.csv"),
        )


def save_model(model, model_dir, device, model_name="model.pth"):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, model_name)
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    model.to(args.device)


def get_args():
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="S",
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count(),
    )

    # Container environment
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"],
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"],
    )

    # Sagemaker Experiments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="run name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.run_name is None and args.experiment_name is None:
        run = None
        train(args, run)
    else:
        with load_run(
            experiment_name=args.experiment_name, run_name=args.run_name
        ) as run:
            train(args, run)
