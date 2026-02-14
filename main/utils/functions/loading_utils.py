import os
import re 
import glob
import pickle
import torch

from alphazoo.network_manager import Network_Manager




def load_network_checkpoint(game_name, network_name, checkpoint_number):
    game_folder = "Games/" + game_name + "/"
    cp_network_folder = game_folder + "models/" + network_name + "/"
    if not os.path.exists(cp_network_folder):
        raise Exception("Could not find a network with that name.")
    
    buffer_path = cp_network_folder + "replay_buffer.cp"
    plot_path = cp_network_folder + "plot_data.pkl"

    if checkpoint_number == "auto":
        cp_paths = glob.glob(cp_network_folder + "*_cp")
        # In each filename: finds all numbers in filename -> gets the last one -> converts to int. Then orders all the ints extracted -> gets the last one
        checkpoint_number = sorted(list(map(lambda str: int(re.findall('\d+',  str)[-1]), cp_paths)))[-1]    

    checkpoint_path =  cp_network_folder + network_name + "_" + str(checkpoint_number) + "_cp"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model_pickle_path =  cp_network_folder + "base_model.pkl"
    model = load_pickle(model_pickle_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_pickle_path =  cp_network_folder + "base_optimizer.pkl"
    base_optimizer = load_pickle(optimizer_pickle_path)
    optimizer_dict = checkpoint["optimizer_state_dict"]

    scheduler_pickle_path =  cp_network_folder + "base_scheduler.pkl"
    base_scheduler = load_pickle(scheduler_pickle_path)
    scheduler_dict = checkpoint["scheduler_state_dict"]

    nn = Network_Manager(model)
    return nn, base_optimizer, optimizer_dict, base_scheduler, scheduler_dict, buffer_path, plot_path, checkpoint_number

def save_checkpoint(save_path, network, optimizer, scheduler):
    checkpoint = \
    {
    'model_state_dict': network.get_model().state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)