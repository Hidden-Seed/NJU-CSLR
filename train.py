import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from nnet.blstm import blstm
from utils.logger import *
from utils.config.train_config import *


def create_logger(config):
    logger = Logger(config["data"]["log_dir"], config["data"]["log_name"])
    logger.info(config["model"])
    logger.info(config["data"])

    return logger


def create_dataloader(data_path, label_path, data_class):
    # Read the data and convert to a Tensor
    np_data_x = np.load(data_path, allow_pickle=True)
    np_data_y = np.load(label_path, allow_pickle=True)
    data_x = torch.from_numpy(np_data_x)
    data_y = torch.from_numpy(np_data_y)

    # Outermost layer is a list, second layer is a tuple, innermost layers are ndarrays.
    data = list(data_x.numpy().reshape(1, -1, time_step, input_size))
    data.append(list(data_y.numpy().reshape(-1, 1)))
    data = list(zip(*data))

    sampler = DistributedSampler(data)

    # Create dataLoader
    dataloader = DataLoader(data, batch_size=batch_size, num_workers=cpu_nums,
                            sampler=sampler, pin_memory=True)
    return dataloader


def data_split(config, logger):
    # Check if the data file exists
    dataset_dir = config["data"]["dataset_processed_dir"]
    data_file_name = config["data"]["data_file_name"]
    label_file_name = config["data"]["label_file_name"]

    train_data_path = os.path.join(dataset_dir, f"train_{data_file_name}")
    valid_data_path = os.path.join(dataset_dir, f"valid_{data_file_name}")
    test_data_path = os.path.join(dataset_dir, f"test_{data_file_name}")

    train_label_path = os.path.join(dataset_dir, f"train_{label_file_name}")
    valid_label_path = os.path.join(dataset_dir, f"valid_{label_file_name}")
    test_label_path = os.path.join(dataset_dir, f"test_{label_file_name}")

    data_label_list = [train_data_path, train_data_path, test_data_path,
                       train_label_path, valid_label_path, test_label_path]
    try:
        if not all(os.path.exists(file) for file in data_label_list):
            logger.error(
                "Data has not been properly processed and split.")
            raise FileNotFoundError("Data files are missing.")

    except RuntimeError as e:
        logger.error(f"Error occurred: {e}")
        # Notify other processes in the distributed system to exit.
        dist.barrier()
        sys.exit(1)

    train_loader = create_dataloader(
        train_data_path, train_label_path, "train")
    valid_loader = create_dataloader(
        valid_data_path, valid_label_path, "valid")
    test_loader = create_dataloader(
        test_data_path, test_label_path, "test")

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # NCCL is the fastest and most recommended backend on GPU devices
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    # Set the device for this process to the corresponding GPU
    torch.cuda.set_device(local_rank)

    options = create_parser()
    config = read_config(options)
    logger = create_logger(config)

    # Model config
    time_step = int(config["model"]["TIME_STEP"])
    input_size = int(config["model"]["INPUT_SIZE"])
    hidden_size = int(config["model"]["HIDDEN_SIZE"])
    output_size = int(config["model"]["OUTPUT_SIZE"])
    batch_size = int(config["model"]["BATCH_SIZE"])
    epoch = int(config["model"]["EPOCH"])
    lr = float(config["model"]["LEARNING_RATE"])
    drop_rate = float(config["model"]["DROP_RATE"])
    layers = int(config["model"]["LAYERS"])
    cpu_nums = int(config["model"]["CPU_NUMS"])

    # Model storage config
    model_save_dir = config["data"]["model_save_dir"]
    os.makedirs(model_save_dir, exist_ok=True)
    iteration = options.train_iteration
    model_save_name = f"{options.model_type}_output{output_size}_input{time_step}x{input_size}_{iteration}.pkl"

    train_loader, valid_loader, test_loader = data_split(config, logger)

    # Set the random seed
    torch.manual_seed(int(config["model"]["SEED"]))

    # Create the model
    model = blstm(input_size, hidden_size, output_size,
                  layers, drop_rate).to(local_rank)
    # Load the model before constructing the DDP model, and it only needs to be loaded on the master node.
    ckpt_path = None
    if dist.get_rank() == 0 and ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path))
    # Create DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # The optimizer can only be initialized using the model after constructing the DDP model.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Create loss function
    loss_func = nn.CrossEntropyLoss().to(local_rank)
    # Define learning rate decay points
    # Reduce the learning rate to 1/10 of its original value at 50% and 75% of the training progress.
    # Epoch must not be less than 4.
    mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epoch // 2, epoch // 4 * 3], gamma=0.1)

    train_loss, valid_loss = [], []
    min_valid_loss = np.inf

    # iterator = tqdm(range(epoch))
    for cur_epoch in range(epoch):
        total_train_loss = []
        # Set training mode
        model.train()

        # Set the epoch for the sampler,
        # DistributedSampler requires this to specify the shuffle method,
        # By maintaining the same random seed across different processes,
        # It ensures that all processes have the same shuffle effect.
        train_loader.sampler.set_epoch(cur_epoch)

        for step, (b_x, b_y) in enumerate(train_loader):
            # The target for CrossEntropy should be a LongTensor, must be 1-D, not in one-hot encoding format.
            b_x = b_x.type(torch.FloatTensor).to(local_rank)
            b_y = b_y.type(torch.long).to(local_rank)
            prediction = model(b_x)  # rnn output
            # h_s = h_s.data  # repack the hidden state, break the connection from last iteration
            # h_c = h_c.data  # repack the hidden state, break the connection from last iteration

            # To calculate the loss, the target must be converted to 1-D
            # b_y is not in one-hot encoding format.
            loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            total_train_loss.append(loss.item())
        # Store the average cross-entropy.
        train_loss.append(np.mean(total_train_loss))

        total_valid_loss = []
        # Set validation mode
        model.eval()
        valid_loader.sampler.set_epoch(cur_epoch)
        for step, (b_x, b_y) in enumerate(valid_loader):
            b_x = b_x.type(torch.FloatTensor).to(local_rank)
            b_y = b_y.type(torch.long).to(local_rank)
            with torch.no_grad():
                prediction = model(b_x)  # rnn output
            # h_s = h_s.data  # repack the hidden state, break the connection from last iteration
            # h_c = h_c.data  # repack the hidden state, break the connection from last iteration
            loss = loss_func(prediction[:, -1, :], b_y.view(b_y.size()[0]))
            total_valid_loss.append(loss.item())
        valid_loss.append(np.mean(total_valid_loss))

        # Save checkpoint if conditions are met
        # Save model.module instead of model
        # This is because model is actually a DDP model, and its parameters are wrapped by model = DDP(model).
        # Only save once on process 0 to avoid saving duplicates multiple times.

        if valid_loss[-1] < min_valid_loss and dist.get_rank() == 0:
            checkpoint = {
                "epoch": cur_epoch,
                "model": model.module,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
            }
            # Save the model
            torch.save(checkpoint, os.path.join(
                model_save_dir, model_save_name))
            min_valid_loss = valid_loss[-1]

        log_string = (
            "iter: [{:d}/{:d}], train_loss: {:0.6f}, valid_loss: {:0.6f}, "
            "best_valid_loss: {:0.6f}, lr: {:0.7f}"
        ).format((cur_epoch + 1), epoch, train_loss[-1],
                 valid_loss[-1], min_valid_loss, optimizer.param_groups[0]["lr"])

        # Update learning rate
        mult_step_scheduler.step()
        logger.info(log_string)

    final_predict = []
    ground_truth = []

    for step, (b_x, b_y) in enumerate(test_loader):
        b_x = b_x.type(torch.FloatTensor).to(local_rank)
        b_y = b_y.type(torch.long).to(local_rank)
        with torch.no_grad():
            prediction = model(b_x)  # rnn output
        # h_s = h_s.data        # repack the hidden state, break the connection from last iteration
        # h_c = h_c.data        # repack the hidden state, break the connection from last iteration

        ground_truth = ground_truth + \
            b_y.view(b_y.size()[0]).cpu().numpy().tolist()
        pre_result = torch.max(F.softmax(prediction[:, -1, :], dim=1), 1)
        pre_class = pre_result[1].cpu().data.numpy().tolist()
        # pre_prob = pre_result[0].cpu().data.numpy().tolist()
        # print(pre_class, pre_prob)
        final_predict = final_predict + pre_class

    ground_truth = np.asarray(ground_truth)
    final_predict = np.asarray(final_predict)

    accuracy = float((ground_truth == final_predict).astype(
        int).sum()) / float(final_predict.size)
    logger.info("Test accuracy: " + str(accuracy))
