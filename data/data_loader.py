# %%
import numpy as np
import torch
import os


# %%
def extract_in_out_from_folder(folder_path):
    in_list = []
    out_list = []

    # Get all txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, "r") as file:
            content = file.read()

            # Split by lines and process each line
            for line in content.splitlines():
                if "IN:" in line and "OUT:" in line:
                    # Extract the parts following "IN:" and "OUT:"
                    in_part = line.split("IN:")[1].split("OUT:")[0].strip()
                    out_part = line.split("OUT:")[1].strip()

                    # Append to respective lists
                    in_list.append(in_part.split(" "))
                    out_list.append(out_part.split(" "))
    return in_list, out_list


def one_hot_enc(len, idx):  # onehot encoding function
    arr = np.zeros(len)
    arr[idx] = 1
    return arr


def conver_str_to_embed_tensor(data, pad_length):
    masks = []
    embed_data = np.zeros((len(data), pad_length))  # create data we want to fill into

    for i, dat in enumerate(data):  # pad data
        ### Pad data ###
        add_pad_len = pad_length - len(dat)  # find diff to max
        data[i] = dat + ["None"] * add_pad_len  # add that amount of "None"s
        masks.append(add_pad_len)

    ### Create embedding ###
    unique = np.unique(data)  # find unique
    # N_unique = len(unique) # find number of unique
    # make a dict with onehot encodings based on the unique words
    embed_dict = {str(word): i for i, word in enumerate(unique)}

    ### Apply embedding to the data ###
    for i, dat in enumerate(data):  # pad data
        embed_dat = np.array(
            [embed_dict[word] for word in data[i]]
        )  # apply embedding to each word
        embed_data[i] = embed_dat  # save data

    # convert to tensor
    embed_data = torch.tensor(embed_data)
    # add start and end of sequence token
    return embed_data, embed_dict


# %%

# Example usage
path = "C:/Users/skrok/Documents/GitHub/AT-NLP-KU/group_project/data/SCAN-master/add_prim_split/"
in_data, out_data = extract_in_out_from_folder(path)

pad_length = 12
pad_length_out = 48
in_data_tens, embed_in = conver_str_to_embed_tensor(in_data, pad_length)
out_data_tens, embed_out = conver_str_to_embed_tensor(out_data, pad_length_out)
# %%
# fish
