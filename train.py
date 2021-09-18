import torch
import torch.nn as nn
from model.las_model import Listener, Speller, mmWavoice, mmWavoiceNet,rescrossSE, interattention, ECABasicBlock
from utils.data import mmWavoiceDataset, mmWavoiceLoader
from torch.utils.tensorboard import SummaryWriter
from solver.solver import batch_iterator,mmWavoice_batch_iterator
import numpy as np
import yaml
import os
import random
import argparse
from tqdm import tqdm

#Set cuda device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Training script for LAS on mmWave_voice.")
parser.add_argument(
    "--config_path", metavar="config_path", type=str, help = " Path to config file for training.", required=False, default="./config/Wavoice.yaml"
)
parser.add_argument(
    "--experiment_name", metavar="experiment_name", type=str, help = "Name for tensorborad logs", default= "Wavoice",
)

def main(args):
    writer = SummaryWriter(comment=args.experiment_name)
    #Fix seed
    seed = 17
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("------------------------------------------------")
    print("Loading Config", flush=True)
    #load config file 
    config_path = args.config_path
    print("Loading configure at", config_path)
    with open(config_path,"r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    data_name = params["data"]["name"]

    tf_rate_upperbound = params["training"]["tf_rate_upperbound"] ## teacher forcing rate during training will be linearly in las
    tf_rate_lowerbound = params["training"]["tf_rate_lowerbound"]  # decaying from upperbound to lower bound for each epoch in las
    tf_decay_step = params["training"]["tf_decay_step"]
    epochs = params["training"]["epochs"]

    #Load datasets
    print("------------------------------------------------")
    print("Processing datasets...",flush=True)
    train_dataset = mmWavoiceDataset(params, "train")#AudioDataset(params, "train")
    train_loader = mmWavoiceLoader(train_dataset, shuffle=True, num_workers=params["data"]["num_works"]).loader
    dev_dataset = mmWavoiceDataset(params, "test") #AudioDataset(params,"test")
    dev_loader = mmWavoiceLoader(dev_dataset, num_workers=params["data"]["num_works"]).loader

    print("------------------------------------------------")
    print("Creating model architecture...", flush=True)

    listener = Listener(**params["model"]["listener"])
    speller = Speller(**params["model"]["speller"])
    mmwavoicenet = mmWavoiceNet(ECABasicBlock, rescrossSE, interattention, [1,1,1,1,1])
    mmWavoice_model = mmWavoice(listener, speller, mmwavoicenet)

    print(mmWavoice_model)
    mmWavoice_model.cuda()
    #Create optimizer
    optimizer = torch.optim.Adam(params = mmWavoice_model.parameters(), lr=params["training"]["lr"])
    if params["training"]["continue_from"]:
        print("Loading checkpoint  model %s" % params["training"]["continue_from"])
        package = torch.load(params["training"]["continue_from"])
        mmWavoice_model.load_state_dict(package["state_dict"])
        optimizer.load_state_dict(package["optim_dict"])
        start_epoch = int(package.get("epoch",1))
    else:
        start_epoch  = 0

    print("------------------------------------------------")
    print("Training...", flush=True)

    global_step = 0 + (len(train_loader) * start_epoch)
    best_cv_loss = 10e5
    my_fields = {"loss": 0}
    for epoch in tqdm(range(start_epoch, epochs), desc="Epoch training"):
        epoch_step = 0
        train_loss = []
        train_ler = []
        batch_loss = 0

        for i, (data) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False, desc=f"Epoch number {epoch}"):
            my_fields["loss"] = batch_loss
            tf_rate = tf_rate_upperbound - (tf_rate_upperbound - tf_rate_lowerbound)* min(
                (float(global_step) / tf_decay_step),1
            ) # adjust learning 

            voice_inputs = data[1]["voice_inputs"].cuda()
            mmwave_inputs = data[2]["mmwave_inputs"].cuda()
            labels = data[3]["targets"].cuda()


            batch_loss, batch_ler, true_y,pred_y,raw_pred_seq, = mmWavoice_batch_iterator(
                voice_batch_data=voice_inputs,
                mmwave_batch_data=mmwave_inputs,
                batch_label=labels,
                mmWavoice_model=mmWavoice_model,
                optimizer = optimizer,
                tf_rate = tf_rate,
                is_training=True,
                max_label_len=params["model"]["speller"]["max_label_len"],
                label_smoothing=params["training"]["label_smoothing"],
                vocab_dict=train_dataset.char2idx,
            )
            torch.cuda.empty_cache()
            train_loss.append(batch_loss)
            train_ler.extend(batch_ler)
            global_step += 1
            epoch_step += 1
            writer.add_scalar("loss/train_step", batch_loss, global_step)
            writer.add_scalar("ler/train_step",np.array([sum(train_ler) / len(train_ler)]), global_step)
        train_loss = np.array([sum(train_loss) / len(train_loss)])
        train_ler = np.array([sum(train_ler) / len(train_ler)])
        writer.add_scalar("loss/train-epoch", train_loss, epoch)
        writer.add_scalar("loss/train_epoch",train_ler, epoch)
        #valiation
        val_loss = []
        val_ler = []
        val_step  = 0
        for i, (data) in tqdm(enumerate(dev_loader), total=len(dev_loader), leave=False, desc="Validation"):
            with torch.no_grad():
                voice_inputs = data[1]["voice_inputs"].cuda()
                mmwave_inputs = data[2]["mmwave_inputs"].cuda()
                labels = data[3]["targets"].cuda()

                batch_loss, batch_ler, true_y, pred_y, raw_pred_seq = mmWavoice_batch_iterator(
                    voice_batch_data=voice_inputs,
                    mmwave_batch_data=mmwave_inputs,
                    batch_label=labels,
                    mmWavoice_model=mmWavoice_model,
                    optimizer=optimizer,
                    tf_rate=tf_rate,
                    is_training=False,
                    max_label_len=params["model"]["speller"]["max_label_len"],
                    label_smoothing=params["training"]["label_smoothing"],
                    vocab_dict=dev_dataset.char2idx,
                )
                torch.cuda.empty_cache()
                val_loss.append(batch_loss)
                val_ler.extend(batch_ler)
                val_step += 1
        val_loss = np.array([sum(val_loss) / len(val_loss)])
        val_ler = np.array([sum(val_ler) / len(val_ler)])
        writer.add_scalar("loss/dev", val_loss, epoch)
        writer.add_scalar("ler/dev",val_ler, epoch)
        if params["training"]["checkpoint"]:
            file_path_old = os.path.join(params["training"]["save_folder"], f"{data_name}-epoch{epoch - 10}.pth.tar")
            if os.path.exists(file_path_old):
                os.remove(file_path_old)

            file_path = os.path.join(params["training"]["save_folder"],  f"{data_name}-epoch{epoch}.pth.tar")
            torch.save(
                mmWavoice_model.serialize(optimizer=optimizer, epoch=epoch, tr_loss=train_loss, val_loss=val_loss), file_path,
            )
            print()
            print("Saving  checkpoint model to %s" % file_path)

        if val_loss < best_cv_loss:
            file_path = os.path.join(params["training"]["save_folder"], f"{data_name}-BEST_LOSS-epoch{epoch}.pth.tar")
            torch.save(
                mmWavoice_model.serialize(optimizer=optimizer, epoch=epoch, tr_loss=train_loss, val_loss = val_loss),file_path
            )
            print("Saving BEST model to %s" % file_path)
        print()
















if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
