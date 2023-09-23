import jax
from flax.training import train_state
import optax

from model import BaselineClassifier
from utils import compute_loss_acc_train, prediction_test

@jax.jit
def train_step(state, batch):
    
    grad_fn = jax.value_and_grad(compute_loss_acc_train,
                                 argnums=1,
                                 has_aux=True)
    
    (loss, acc), grads = grad_fn(state, state.params, batch)
    new_state = state.apply_gradients(grads=grads)
    
    return new_state, loss, acc

@jax.jit
def test_step(state, batch):
    pred_labels = prediction_test(state, state.params, batch)
    return pred_labels

class Learner(object):
    def __init__(self, 
                 seed,
                 hidden_dims,
                 input_example,
                 lr):
        
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        
        self.model = BaselineClassifier(hidden_dims)
        self.params = self.model.init(init_rng, input_example)

        self.optimizer = optax.adam(learning_rate=lr)
        self.model_state = train_state.TrainState.create(apply_fn=self.model.apply,
                                                    params=self.params,
                                                    tx=self.optimizer)


    def learner_train(self, batch):
        new_state, loss, acc = train_step(self.model_state, batch)

        self.model_state = new_state
        return loss, acc


    def learner_test(self, batch):
        predicted_label = test_step(self.model_state, batch)
        return predicted_label