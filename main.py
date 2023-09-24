import datetime
from absl import app, flags

import jax
from torch.utils.data import DataLoader
import numpy as np
import tqdm

from utils import LoadDataset, numpy_collate, plot_log, prediction2csv
from learner import Learner


FLAGS = flags.FLAGS
flags.DEFINE_string('alg', 'baseline', 'algorithm for run')
flags.DEFINE_float('lr', 3e-4, 'learning rate')
flags.DEFINE_multi_integer('hidden_dims', [128, 256, 128, 1], 'dimensions of hidden layers')
flags.DEFINE_integer('batch_size', 256, 'mini batch size')
flags.DEFINE_integer('num_epoch', 10, 'epoch')
flags.DEFINE_integer('seed', 0, 'seed')


def main(_):

    # make dataset
    train_dataset = LoadDataset('dataset/train.csv', has_label=True)
    test_dataset = LoadDataset('dataset/test.csv', has_label=False)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, collate_fn=numpy_collate)

    # initialize model with data example
    kwargs = dict()
    kwargs['seed'] = FLAGS.seed
    kwargs['hidden_dims'] = FLAGS.hidden_dims
    kwargs['input_example'] = test_dataset[0][np.newaxis]
    kwargs['lr'] = FLAGS.lr
    
    # set learner
    learner = Learner(**kwargs)

    # create logger
    now = datetime.datetime.today().strftime('%Y%m%d%H%M%S')
    log_dir = f"run_log/{FLAGS.alg}_seed{FLAGS.seed}_{now}"
    training_info = {
        'epoch': [],
        'accuracy': [],
        'loss': []
    }

    # train and save log
    for epoch in range(FLAGS.num_epoch):
        for batch in tqdm.tqdm(train_dataloader,
                               smoothing=0.1,
                               disable=False,
                               ncols=90):
            loss, acc = learner.learner_train(batch)
        
        print("\n=========================")
        print(f"epoch ({epoch+1}/{FLAGS.num_epoch})")
        print("train accuracy: ", acc*100)
        print("train loss: ", loss)
        print("=========================\n")

        training_info['epoch'].append(epoch+1)
        training_info['accuracy'].append(acc*100)
        training_info['loss'].append(loss)
        plot_log(log_dir, training_info)
    
    # test and save prediction
    predicted_label = learner.learner_test(test_dataset[:])
    prediction2csv(log_dir, predicted_label, test_dataset.index)

    print("All Done!")


if __name__ == '__main__':
    app.run(main)