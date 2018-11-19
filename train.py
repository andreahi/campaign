import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.python.keras.layers import LeakyReLU
product_dim = 184
new_product_dim = 89

g = tf.Graph()
with g.as_default():

    birth_date = tf.placeholder(tf.float32, [None], name="birth_date")
    history = tf.placeholder(tf.float32, [None, product_dim], name="history")
    current_products = tf.placeholder(tf.float32, [None, product_dim], name="current_products")
    new_product = tf.placeholder(tf.float32, [None, new_product_dim], name="new_product")

    from tensorflow.contrib import slim

    hidden = tf.concat([
        slim.fully_connected(tf.expand_dims(birth_date, axis=1), 10, activation_fn=LeakyReLU(), weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1)),
        slim.fully_connected(history, 100, activation_fn=LeakyReLU(), weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1)),
        slim.fully_connected(current_products, 100, activation_fn=LeakyReLU(), weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=.1))]
        , axis=1)

    hidden = slim.fully_connected(hidden, 100, activation_fn=LeakyReLU(),weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = slim.fully_connected(hidden, 100, activation_fn=LeakyReLU(),weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = slim.fully_connected(hidden, 100, activation_fn=LeakyReLU(),weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    hidden = slim.fully_connected(hidden, 100, activation_fn=LeakyReLU(),weights_initializer=tf.contrib.layers.variance_scaling_initializer())

    out = slim.fully_connected(hidden, new_product_dim, activation_fn=LeakyReLU(),weights_initializer=tf.contrib.layers.variance_scaling_initializer())
    out = tf.nn.softmax(out)
    #out = tf.Print(out, [out, tf.shape(out)], "out: ", summarize=100)
    loss = tf.losses.softmax_cross_entropy(new_product, out)

    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(loss)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    df = pd.read_json("filename_formated.txt")
    print("Amount of traning data: ", len(df))

    axis_sum = np.sum(np.array(list(np.array(df.newProducts))), axis=1)
    if max(axis_sum) != 1 or min(axis_sum) != 1 :
        print("ERROR: wrong dim sum")
        exit(12)

    birth_dates = np.array(list(np.array(df.birthDate)))
    birth_data_normalizer = Normalizer().fit(birth_dates.reshape(-1, 1))
    birth_dates = np.squeeze(birth_data_normalizer.transform(birth_dates.reshape(-1, 1)))

    saver.restore(sess, "./checkpoint.ckpt")


    for i in range(100000):
        _, train_loss = sess.run([optimizer, loss], feed_dict={
            birth_date: birth_dates,
            history: np.array(list(np.array(df.history))),
            current_products: np.array(list(np.array(df.currentProducts))),
            new_product: np.array(list(np.array(df.newProducts)))})
        if i % 1000 == 0:
            print("train_loss: ", train_loss)
            saver.save(sess, "./checkpoint.ckpt")
