import json
import time
import utils
import wrappers

import tensorflow as tf
import numpy as np

from agent import r2d2


flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('task', -1, "Task id. Use -1 for local training")
flags.DEFINE_enum('job_name', 
                  'learner',
                  ['learner', 'actor'],
                  'Job name. Ignore when task is set to -1')

def main(_):
    data = json.load(open('config.json'))
    utils.check_properties(data)
            
    learner = r2d2.Agent(
        trajectory=data['trajectory'],
        input_shape=data['model_input'],
        num_action=data['model_output'],
        lstm_hidden_size=data['lstm_size'],
        discount_factor=data['discount_factor'],
        start_learning_rate=data['start_learning_rate'],
        end_learning_rate=data['end_learning_rate'],
        learning_frame=data['learning_frame'],
        baseline_loss_coef=data['baseline_loss_coef'],
        entropy_coef=data['entropy_coef'],
        gradient_clip_norm=data['gradient_clip_norm'],
        reward_clipping=data['reward_clipping'],
        model_name='learner',
        learner_name='learner')

    sess = tf.Session()
    learner.set_session(sess)

    learner.test()

if __name__ == '__main__':
    tf.app.run()