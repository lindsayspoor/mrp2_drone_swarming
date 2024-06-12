#### based on tutorial from philtabor on GitHub ####

# from tensorflow import keras
import tensorflow as tf
# import tensorflow.keras as keras
# from tf.keras.layers import Dense

# @tf.keras.utils.register_keras_serializable()
class ActorNetwork(tf.keras.Model):
    def __init__(self, n_actions, fc1_dims=64, fc2_dims=64, **kwargs):
        super(ActorNetwork, self).__init__(**kwargs)

        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.fc3 = tf.keras.layers.Dense(n_actions, activation='softmax')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
    
    def get_config(self):
        config = {'n_actions': self.n_actions, 'fc1_dims': self.fc1_dims, 'fc2_dims': self.fc2_dims}
        base_config = super(ActorNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    


# @tf.keras.utils.register_keras_serializable()
class CriticNetwork(tf.keras.Model):
    def __init__(self, fc1_dims=64, fc2_dims=64, **kwargs):
        super(CriticNetwork, self).__init__(**kwargs)
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = tf.keras.layers.Dense(fc1_dims, activation='relu')
        self.fc2 = tf.keras.layers.Dense(fc2_dims, activation='relu')
        self.q = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        q = self.q(x)

        return q
    
    def get_config(self):
        config = {'fc1_dims': self.fc1_dims, 'fc2_dims': self.fc2_dims}
        base_config = super(CriticNetwork, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))