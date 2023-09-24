import os
import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import optax

class LoadDataset(Dataset):

    def __init__(self, csv_path, has_label=True):

        self.has_label = has_label
        dataset = pd.read_csv(csv_path, index_col=0)
        self.index = dataset.index

        features = dataset.copy()
        if has_label:
            labels = features.pop('booking_status')

        self.features = jnp.array(features.values)
        if has_label:
            self.labels = jnp.array(labels.values)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        data_features = self.features[index]
        
        if self.has_label:
            data_labels = self.labels[index]

            return data_features, data_labels
        
        return data_features


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def compute_loss_acc_train(state, params, batch):
    input, labels = batch

    logits = state.apply_fn(params, input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)

    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()

    return loss, acc


def prediction_test(state, params, batch):
    input = batch

    logits = state.apply_fn(params, input).squeeze(axis=-1)
    pred_labels = (logits > 0).astype(jnp.float32)

    return pred_labels


def plot_log(dir, info_dict):
    os.makedirs(dir, exist_ok=True)

    for key, val in info_dict.items():
        if key != 'epoch':
            plt.title(f"training_{key}")
            plt.plot(info_dict['epoch'], val)
            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.savefig(f"{dir}/{key}.png")
            plt.close()


def prediction2csv(dir, label, index):
    df = pd.DataFrame(index, columns=['id'])
    df['booking_status'] = np.asarray(label, dtype=np.int32)
    
    df.to_csv(f"{dir}/prediction.csv", index=False)
