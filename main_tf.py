"""Author: Rico Jonschkowski (rico.jonschkowski@tu-berlin.de), ported from Theano to Tensorflow by Marco Morik

This is a simple implementation of the method for state representation learning described our the paper "Learning State
Representations with Robotic Priors" (Jonschkowski & Brock, 2015). This implementation complements the paper to provide
sufficient detail for reproducing our results and for reusing the method in other research while minimizing code
overhead (extensive explanations and descriptions are omitted here and can be found in the paper).

This version uses Tensorflow and Sonnet instead of Theano.

If you are using this implementation in your research, please consider giving credit by citing our paper:

@article{jonschkowski2015learning,
  title={Learning state representations with robotic priors},
  author={Jonschkowski, Rico and Brock, Oliver},
  journal={Autonomous Robots},
  volume={39},
  number={3},
  pages={407--428},
  year={2015},
  publisher={Springer}
}
"""

import numpy as np
import sonnet as snt
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")  # to ignore the following warning:
# anaconda3/lib/python3.5/site-packages/matplotlib/backend_bases.py:2437: MatplotlibDeprecationWarning:
# Using default event loop until function specific to this GUI is implemented
# warnings.warn(str, mplDeprecation)

class SRL4robotics:

    def __init__(self, obs_dim, state_dim, seed=1, learning_rate=0.001, l1_reg = 0.001):

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.batchsize = 256

        # seed random number generator
        self.rand = np.random.RandomState(seed)

        # init values
        self.mean_obs = np.zeros(self.obs_dim)
        self.std_obs = 1


        # Input variables: 
        self.obs_var = tf.placeholder(tf.float32, shape=[None,obs_dim], name="obs_var")
        self.next_obs_var = tf.placeholder(tf.float32, shape=[None,obs_dim], name="next_obs_var")
        #action_var = tf.placeholder(tf.float32, name="action_var")        
        self.is_training = tf.placeholder(tf.bool, shape=[],  name="train_cond")
        self.dissimilar_var = tf.placeholder(tf.int32, shape=[None,2])
        self.same_actions_var = tf.placeholder(tf.int32, shape=[None,2])
        
        # DEFINE OBSERVATION STATE MAPPING SEQ_LAYER -------------------------------------------------------------------------        
                
        regularizers = {"w": tf.contrib.layers.l1_regularizer(scale=l1_reg),
                        "b": tf.contrib.layers.l1_regularizer(scale=l1_reg)}        

        inp_layer = snt.Linear(output_size=self.state_dim, name='inp_to_out', regularizers=regularizers)
        
        # to keep loss terms differentiable, states should never be equal
        noise_layer = lambda x: tf.cond(self.is_training, lambda: x + tf.random_normal(shape=tf.shape(x), stddev=1e-6), lambda: x)
        self.seq_layer = snt.Sequential([inp_layer,noise_layer])
        
        state_var = self.seq_layer(self.obs_var)
        next_state_var = self.seq_layer(self.next_obs_var)          
        self.state = self.seq_layer(self.obs_var) 
        
        
        # DEFINE LOSS FUNCTION -----------------------------------------------------------------------------------------

        state_diff = next_state_var - state_var
        state_diff_norm = tf.norm(state_diff, ord=2, axis=1)
        
        similarity = lambda x, y: tf.exp(- tf.norm((x - y), ord=2, axis=1) ** 2)
        
        temp_coherence_loss = tf.reduce_mean(state_diff_norm ** 2)
        
        causality_loss = tf.reduce_mean(similarity(tf.gather(state_var, self.dissimilar_var[:, 0]),
                                        tf.gather(state_var, self.dissimilar_var[:, 1]))) 
        
        proportionality_loss = tf.reduce_mean(
            (tf.gather(state_diff_norm, self.same_actions_var[:, 0]) - tf.gather(state_diff_norm, self.same_actions_var[:, 1])) ** 2)

        repeatability_loss = tf.reduce_mean(
            similarity(tf.gather(state_var, self.same_actions_var[:, 0]), tf.gather(state_var, self.same_actions_var[:, 1])) *
            tf.norm(tf.gather(state_diff, self.same_actions_var[:, 0]) - tf.gather(state_diff, self.same_actions_var[:, 1]), ord=2, axis=1) ** 2)
        
        graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_regularization_loss = tf.reduce_sum(graph_regularizers)
        
        # compute a weighted sum of the loss terms
        self.loss = 1 * temp_coherence_loss   + 5 * proportionality_loss + 5 * repeatability_loss +  1 * causality_loss + total_regularization_loss

        # TRAINING FUNCTIONS --------------------------------------------------------------------------------------------
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)


    def learn(self, observations, actions, rewards, episode_starts):

        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we normalize the observations, organize the data into minibatches
        # and find pairs for the respective loss terms

        self.mean_obs = np.mean(observations, axis=0, keepdims=True)
        self.std_obs = np.std(observations, ddof=1)
        observations = (observations - self.mean_obs) / self.std_obs

        num_samples = observations.shape[0] - 1  # number of samples

        # indices for all time steps where the episode continues
        indices = np.array([i for i in range(num_samples) if not episode_starts[i + 1]], dtype='int32')
        np.random.shuffle(indices)

        # split indices into minibatches
        minibatchlist = [np.array(sorted(indices[start_idx:start_idx + self.batchsize]))
                         for start_idx in range(0, num_samples - self.batchsize + 1, self.batchsize)]

        find_same_actions = lambda index, minibatch: np.where(np.prod(actions[minibatch] == actions[minibatch[index]], axis=1))[0]
        same_actions = [
            np.array([[i, j] for i in range(self.batchsize) for j in find_same_actions(i, minibatch) if j > i],
                     dtype='int32') for minibatch in minibatchlist]
                     
        # check with samples should be dissimilar because they lead to different rewards aften the same actions
        find_dissimilar = lambda index, minibatch: np.where(np.prod(actions[minibatch] == actions[minibatch[index]], axis=1) *
                            (rewards[minibatch + 1] != rewards[minibatch[index] + 1]))[0]
        dissimilar = [np.array([[i, j] for i in range(self.batchsize) for j in find_dissimilar(i, minibatch) if j > i],
                               dtype='int32') for minibatch in minibatchlist]


        # TRAINING -----------------------------------------------------------------------------------------------------
        init = tf.global_variables_initializer()      
        num_epochs = 150
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)            
            for epoch in range(num_epochs):
                # In each epoch, we do a full pass over the training data:
                epoch_loss = 0
                epoch_batches = 0
                enumerated_minibatches = list(enumerate(minibatchlist))
                np.random.shuffle(enumerated_minibatches)
                
                for i, batch in enumerated_minibatches:
                    diss = dissimilar[i][np.random.permutation(dissimilar[i].shape[0])]  # [:10*self.batchsize]]
                    
                    same = same_actions[i][np.random.permutation(same_actions[i].shape[0])]  # [:10*self.batchsize]]

                    _ , tmp_loss = sess.run([self.train_op,self.loss], feed_dict = {
                                                                        self.obs_var: observations[batch],
                                                                        self.next_obs_var: observations[batch+1],
                                                                        self.dissimilar_var: diss,
                                                                        self.same_actions_var: same,
                                                                        self.is_training: True})
                    epoch_loss += tmp_loss
                    epoch_batches += 1
                # Then we print the results for this epoch:
                if (epoch+1) % 5 == 0:
                    print("Epoch {:3}/{}, loss:{:.4f}".format(epoch+1, num_epochs, epoch_loss / epoch_batches))
    
                # Optionally plot the current state space
                plot_representation(sess.run(self.state, feed_dict = {self.obs_var: observations, self.is_training: False}), rewards, add_colorbar=epoch==0,
                                         name="Learned State Representation (Training Data)")
            #Saves the calculated weights
            saver.save(sess, "/tmp/model.ckpt")
            predicted_state =  sess.run(self.state, feed_dict = {self.obs_var: observations, self.is_training: False})   
        plt.close("Learned State Representation (Training Data)")

        # return predicted states for training observations
        return predicted_state

    def phi(self, observations):
        observations = (observations - self.mean_obs) / self.std_obs
        saver = tf.train.Saver()
        with tf.Session() as sess:
            #Loads the model and calculate the new states
            saver.restore(sess, "/tmp/model.ckpt")
            states = sess.run(self.state, feed_dict = {self.obs_var: observations, self.is_training: False})
        
        return states


def plot_representation(states, rewards, name="Learned State Representation", add_colorbar=True):
    plt.ion()
    plt.figure(name)
    plt.hold(False)
    plt.scatter(states[:, 0], states[:, 1], s=7, c=np.clip(rewards, -1, 1), cmap='bwr', linewidths=0.1)
    plt.xlim([-2, 2])
    plt.xlabel('State dimension 1')
    plt.ylim([-2, 2])
    plt.ylabel('State dimension 2')
    if add_colorbar:
        plt.colorbar(label='Reward')
    plt.pause(0.0001)

def plot_observations(observations, name = 'Observation Samples'):
    plt.ion()
    plt.figure(name)
    m, n = 8, 10
    for i in range(m*n):
        plt.subplot(m, n, i+1)
        plt.imshow(observations[i].reshape(16,16,3), interpolation='nearest')
        plt.gca().invert_yaxis()
        plt.xticks([])
        plt.yticks([])
    plt.pause(0.0001)



if __name__ == '__main__':

    print('\nSIMPLE NAVIGATION TASK\n')

    print('Loading and displaying training data ... ')
    training_data = np.load('simple_navigation_task_train.npz')
    plot_observations(training_data['observations'], name="Observation Samples (Subset of Training Data) -- Simple Navigation Task")

    print('Learning a state representation ... ')
    srl = SRL4robotics(16 * 16 * 3, 2, learning_rate=0.0001, l1_reg = 0.01)
    training_states = srl.learn(**training_data)
    plot_representation(training_states, training_data['rewards'],
                            name='Observation-State-Mapping Applied to Training Data -- Simple Navigation Task',
                            add_colorbar=True)

    print('Loading and displaying test data ... ')
    test_data = np.load('simple_navigation_task_test.npz')
    plot_observations(test_data['observations'], name="Observation Samples (Subset of Test Data) -- Simple Navigation Task")

    print('Testing the learned representation on this new data ... ')
    test_states = srl.phi(test_data['observations'])
    plot_representation(test_states, test_data['rewards'], name='Observation-State-Mapping Applied to Test Data -- Simple Navigation Task', add_colorbar=True)

    ####################################################################################################################
    
    print('\nSLOT CAR TASK\n')

    print('Loading and displaying training data ... ')
    training_data = np.load('slot_car_task_train.npz')
    plot_observations(training_data['observations'], name="Observation Samples (Subset of Training Data) -- Slot Car Task")

    print('Learning a state representation ... ')
    srl = SRL4robotics(16 * 16 * 3, 2, learning_rate=0.001, l1_reg = 0.001)
    training_states = srl.learn(**training_data)
    plot_representation(training_states, training_data['rewards'],
                            name='Observation-State-Mapping Applied to Training Data -- Slot Car Task',
                            add_colorbar=True)

    print('Loading and displaying test data ... ')
    test_data = np.load('slot_car_task_test.npz')
    plot_observations(test_data['observations'], name="Observation Samples (Subset of Test Data) -- Slot Car Task")

    print('Testing the learned representation on this new data ... ')
    test_states = srl.phi(test_data['observations'])
    plot_representation(test_states, test_data['rewards'], name='Observation-State-Mapping Applied to Test Data -- Slot Car Task',
                            add_colorbar=True)

    input('\nPress any key to exit.')
