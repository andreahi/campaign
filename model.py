import tensorflow as tf
from tensorflow.python.keras.layers import LeakyReLU

from optimization import create_optimizer

product_dim = 150
new_product_dim = 34


def get_model():
    training = tf.placeholder(tf.bool)  # Can be any computed boolean expression.

    birth_date = tf.placeholder(tf.float32, [None], name="birth_date")
    history = tf.placeholder(tf.float32, [None, product_dim], name="history")
    consumer_product = tf.placeholder(tf.float32, [None], name="current_products")
    business_product = tf.placeholder(tf.float32, [None], name="current_products")
    current_products = tf.placeholder(tf.float32, [None, product_dim], name="current_products")
    new_product = tf.placeholder(tf.float32, [None, new_product_dim], name="new_product")

    from tensorflow.contrib import slim

    hidden = tf.concat(
        [
            slim.fully_connected(tf.expand_dims(birth_date, axis=1), 2, activation_fn=LeakyReLU(),
                                 weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1)),
            slim.fully_connected(history, 20, activation_fn=LeakyReLU(),
                                 weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1)),
            slim.fully_connected(current_products, 20, activation_fn=LeakyReLU(),
                                 weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1)),
            slim.fully_connected(tf.expand_dims(consumer_product, axis=1), 1, activation_fn=LeakyReLU(),
                                 weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1)),
            slim.fully_connected(tf.expand_dims(business_product, axis=1), 1, activation_fn=LeakyReLU(),
                                 weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1))
        ]
        , axis=1)

    hidden = slim.fully_connected(hidden, 20, activation_fn=LeakyReLU(), weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1))
    hidden = slim.fully_connected(hidden, 20, activation_fn=LeakyReLU(), weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1))
    hidden = tf.layers.dropout(hidden, rate=.5, training=training)  # DROP-OUT here

    out = slim.fully_connected(hidden, new_product_dim, activation_fn=LeakyReLU(), weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1))
    out = out
    #out = tf.Print(out, [out, tf.shape(out)], "out: ", summarize=100)
    loss = tf.losses.mean_squared_error(new_product, out)

    #optimizer = tf.train.AdamOptimizer(learning_rate=.00001).minimize(loss)
    optimizer = create_optimizer(loss, .001, 100000, 100, False)
    return birth_date, history, current_products, consumer_product, business_product, new_product, loss, optimizer, out, training