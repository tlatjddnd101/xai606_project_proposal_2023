import numpy as np
import tqdm
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 3e-4, 'learning rate')
flags.DEFINE_multi_integer('hidden_dims', [128, 256, 128, 1], 'dimensions of hidden layers')
flags.DEFINE_integer('epoch', 100000, 'epoch')
flags.DEFINE_integer('seed')


def main(_):
    pass

if __name__ == '__main__':
    app.run(main)