import os

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

from FileUtils import save_obj
from model import get_model
from test_model import test_model
os.environ["CUDA_VISIBLE_DEVICES"] = ""

g = tf.Graph()
with g.as_default():

    birth_date, history, current_products, consumer_product, business_product, new_product, loss, optimizer, out, training = get_model()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    df = pd.read_json("train.txt")
    df_test = pd.read_json("test.txt")
    print("Amount of traning data: ", len(df))

    axis_sum = np.sum(np.array(list(np.array(df.newProducts))), axis=1)
    if max(axis_sum) > 10 or min(axis_sum) != 1 :
        print("ERROR: wrong dim sum")
        exit(12)

    birth_dates = np.array(list(np.array(df.birthDate, dtype=float))).reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    birth_dates = min_max_scaler.fit_transform(birth_dates)
    birth_dates = np.squeeze(birth_dates)
    # birth_data_normalizer = Normalizer().fit(birth_dates)
    # birth_dates = np.squeeze(birth_data_normalizer.transform(birth_dates))

    save_obj(min_max_scaler, "min_max_scaler")
    #saver.restore(sess, "./checkpoint.ckpt")
    print("Training samples: ", len(df.newProducts))

    for i in range(1000000):
        _, train_loss = sess.run([optimizer, loss], feed_dict={
            birth_date: birth_dates,
            history: np.array(list(np.array(df.history))),
            current_products: np.array(list(np.array(df.currentProducts))),
            new_product: np.array(list(np.array(df.newProducts))),
            consumer_product: np.array(list(np.array(df.ConsumerType)), dtype=float),
            business_product: np.array(list(np.array(df.BusinessType)), dtype=float),
            training: True
        })
        if i % 100 == 0:
            print("train_loss: ", train_loss)
            print("train accuracy: ", test_model(sess, df, birth_date, history, current_products, consumer_product, business_product, new_product, loss, optimizer, out, training))
            print("test accuracy: ", test_model(sess, df_test, birth_date, history, current_products, consumer_product, business_product, new_product, loss, optimizer, out, training))
            saver.save(sess, "./checkpoint.ckpt")
