from absl import app, flags

import jax
from torch.utils.data import DataLoader
import numpy as np
import tqdm

from utils import LoadDataset, numpy_collate
from learner import Learner


FLAGS = flags.FLAGS
flags.DEFINE_float('lr', 3e-4, 'learning rate')
flags.DEFINE_multi_integer('hidden_dims', [128, 256, 128, 1], 'dimensions of hidden layers')
flags.DEFINE_integer('batch_size', 256, 'mini batch size')
flags.DEFINE_integer('num_epoch', 10, 'epoch')
flags.DEFINE_integer('seed', 0, 'seed')


def main(_):
    # generate PRNG key for JAX
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, inp_rng, init_rng = jax.random.split(rng, 3)

    # make dataset
    train_dataset = LoadDataset('dataset/train.csv', has_label=True)
    test_dataset = LoadDataset('dataset/test.csv', has_label=False)
    train_dataloader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, collate_fn=numpy_collate)
    test_dataloader = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=True, collate_fn=numpy_collate)

    # initialize model with data example
    kwargs = dict()
    kwargs['seed'] = FLAGS.seed
    kwargs['hidden_dims'] = FLAGS.hidden_dims
    kwargs['input_example'] = test_dataset[0][np.newaxis]
    kwargs['lr'] = FLAGS.lr
    
    # set learner
    learner = Learner(**kwargs)

    # train
    for epoch in range(FLAGS.num_epoch):
        for batch in tqdm.tqdm(train_dataloader,
                               smoothing=0.1,
                               disable=False,
                               ncols=90):
            loss, acc = learner.learner_train(batch)
        
        print('\n=========================')
        print(f'epoch ({epoch+1}/{FLAGS.num_epoch})')
        print('accuracy: ', acc*100)
        print('loss: ', loss)
        print('=========================\n')


if __name__ == '__main__':
    app.run(main)