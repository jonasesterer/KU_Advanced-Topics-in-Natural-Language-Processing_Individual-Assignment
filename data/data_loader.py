# %%
import numpy as np
import torch
import os


# %%


def reverse_dict(dict):
    reversed_dict = {value: key for key, value in dict.items()}
    return reversed_dict


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


def extract_in_out_from_file(file_path):
    in_list = []
    out_list = []

    # Get all txt files in the folder

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


def extract_data_to_tens(path, input_max_len=None, out_max_len=None):
    in_data, out_data = extract_in_out_from_file(path)

    if input_max_len is None:
        input_max_len = max([len(dat) for dat in in_data])

    if out_max_len is None:
        out_max_len = max([len(dat) for dat in out_data])

    in_data_tens, embed_in = conver_str_to_embed_tensor(in_data, input_max_len)
    out_data_tens, embed_out = conver_str_to_embed_tensor(out_data, out_max_len)

    return in_data_tens, out_data_tens, embed_in, embed_out


def re_view_for_batch(in_data, out_data, batch_size):
    N = in_data.shape[0]
    # Calculate the number of additional rows needed to make N divisible by B
    remainder = N % batch_size

    if remainder != 0:
        padding_needed = batch_size - remainder
        # Randomly sample rows from the tensor to fill the missing batch
        indices = torch.randint(0, N, (padding_needed,))  # Random indices

        # Concatenate the original tensor with the sampled rows
        in_data = torch.cat([in_data, in_data[indices]], dim=0)
        out_data = torch.cat([out_data, out_data[indices]], dim=0)

    in_data_batched = in_data.view(
        in_data.shape[0] // batch_size, batch_size, in_data.shape[-1]
    )
    out_data_batched = out_data.view(
        out_data.shape[0] // batch_size, batch_size, out_data.shape[-1]
    )

    return in_data_batched, out_data_batched


def get_split_data_from_file_path(
    train_path: str,
    test_path: str,
    batch_size: int = 16,
    input_max_len: int = None,
    out_max_len: int = None,
):
    """
    Inputs:\n
    - train_path: path to train file\n
    - test_path: path to test file\n
    - batch_size: batch size\n
    - input_max_len: the len that the feature dim for the "IN:" should have\n
    - out_max_len: the len that the feature dim for the "OUT:" should have\n
    Output:\n
    - Batched in_tst, out_tst, in_train, out_train, embeds. (size of data is N x B x F)
    \n
    OBS: If "len_commands % B != 0", the function randomly samples the data until it is divisible, so we can do the split without strange sizes.\n
    OBS: If max_len is not given, then the code findes the minimal max len and uses that.
    \n
    where:\n
    - N is number of batches, \n
    - B is batch size, and \n
    - F is feature size\n
    """

    in_data_tst, out_data_tst, embed_in_tst, embed_out_tst = extract_data_to_tens(
        test_path, input_max_len=input_max_len, out_max_len=out_max_len
    )

    (
        in_data_train,
        out_data_train,
        embed_in_train,
        embed_out_train,
    ) = extract_data_to_tens(
        train_path, input_max_len=input_max_len, out_max_len=out_max_len
    )

    in_data_tst, out_data_tst = re_view_for_batch(in_data_tst, out_data_tst, batch_size)
    in_data_train, out_data_train = re_view_for_batch(
        in_data_train, out_data_train, batch_size
    )

    return (
        in_data_tst,
        out_data_tst,
        in_data_train,
        out_data_train,
        {
            "tst": {"in": embed_in_tst, "out": embed_out_tst},
            "train": {"in": embed_in_train, "out": embed_out_train},
        },
    )


if __name__ == "__main__":
    ### Example Load in data from files ###
    # load path if run from ./data/
    path_tst = "datafiles/simple_split/tasks_test_simple.txt"
    path_train = "datafiles/simple_split/tasks_train_simple.txt"

    # Alternative paths, if run from root
    # path_tst = "./data/datafiles/simple_split/tasks_test_simple.txt"
    # path_train = "./data/datafiles/simple_split/tasks_train_simple.txt"
    #
    in_tst, out_tst, in_train, out_train, embeds_dict = get_split_data_from_file_path(
        train_path=path_train,
        test_path=path_tst,
        batch_size=16,
        input_max_len=None,
        out_max_len=48,
    )

    ### Test if in and out match after loading ###
    # print("in: ",[reverse_dict(embeds_dict["train"]["in"])[int(val)] for val in in_train[0,0]])
    # print("out: ",[reverse_dict(embeds_dict["train"]["out"])[int(val)] for val in out_train[0,0]])
    # print("t o: ","I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT".split(" "))

    ### Example Loops ###

    # Train data
    for i, (in_batch_train, out_batch_train) in enumerate(zip(in_train, out_train)):
        ## model train and update ##
        # Bla bla bal
        break  # remove to loop over all data

    # test data
    for i, (in_batch_test, out_batch_test) in enumerate(zip(in_train, out_train)):
        ## model train and update ##
        # Bla bla bal
        break  # remove to loop over all data

    ### Finding word sequence from "embeddings" ###
    # the embeddings are a dict
    # the code: embeds_dict["train"]["in"], gives the embedding for in_data_train
    # the code: embeds_dict["test"]["out"], gives the embedding for out_data_test

    # Examples:
    print("in: ", [int(val) for val in in_train[0, 0]])
    print(
        "in: ",
        [reverse_dict(embeds_dict["train"]["in"])[int(val)] for val in in_train[0, 0]],
    )
    print("out: ", [int(val) for val in out_train[0, 0]])
    print(
        "out: ",
        [
            reverse_dict(embeds_dict["train"]["out"])[int(val)]
            for val in out_train[0, 0]
        ],
    )
