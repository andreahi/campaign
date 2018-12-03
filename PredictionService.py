import json
import os

from FileUtils import load_obj

import tensorflow as tf
import numpy as np

from model import get_model
os.environ["CUDA_VISIBLE_DEVICES"] = ""

birth_date, history, current_products, consumer_product, business_product,  new_product, loss, optimizer, out, training = get_model()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()
saver.restore(sess, "./checkpoint.ckpt")

to_name = {

}
def to_index_bag(product_indexes, products):
    products = list(filter(lambda x: x in product_indexes, products))
    print("after filter: ", products)
    l = list(map(lambda x: product_indexes[x], products))
    vector = np.zeros(len(product_indexes))
    vector[l] = 1
    return vector

def predict(birth_data, product_history, current, consumer_type, business_type):
    min_max_scaler = load_obj("min_max_scaler")
    product_indexes = load_obj("product_indexes")
    target_indexs = load_obj("target_indexs")
    inv_target_indexs = {v: k for k, v in target_indexs.items()}

    birth_data = np.array([np.squeeze(min_max_scaler.transform(birth_data))])
    product_history = to_index_bag(product_indexes, product_history)
    current = to_index_bag(product_indexes, current)



    predicted_new_products = sess.run(out, feed_dict={
        birth_date: birth_data,
        history: [product_history],
        current_products: [current],
        consumer_product: np.array(consumer_type),
        business_product: np.array(business_type),
        training : False
    })[0]

    print(predicted_new_products)
    argsort_ = predicted_new_products.argsort()[:][::-1]
    print([str(predicted_new_products[e]) + ":" + str(inv_target_indexs[e]) for e in argsort_])
    return inv_target_indexs[np.argmax(predicted_new_products)]

with open('productMappings.json') as f:
    to_name = json.load(f)


#print(predict(-130554000000, [], [76902, 679, 51744, 5074, 669, 50875, 76001, 76962, 76054, 951, 689, 76054, 689, 76864]))
private_yng_8gb = [51697, 692, 663, 51744, 76462, 90072, 76962, 50875, 5074, 76001]
private_frihet_6gb = [76006, 50875, 670, 676, 692, 663, 665, 689, 76962, 5074, 602, 50144, 76001] # datagrense 90089
private_yng_2gb = [5074, 76001, 51744, 76462, 90072, 50875, 951, 689, 76962, 51676]

business_fri_60GB = ["not set"]
business_fri_20GB = [17808, 50875, 689, 678, 8486, 7656, 76963, 6518, 5075, 9559, 76055]
business_fri_1GB = ["not set"]
business2 = ["not set"]

for e in private_yng_2gb:
    print(to_name[str(e)])
SWAP_history = [76347, 4444001]
mitt_telenor_data = [17808]

no_history = []
predicted_product = predict(751074800000, [], private_frihet_6gb, [1], [0])
print(predicted_product)
print(to_name[str(predicted_product)])

predicted_product = predict(651074800000, mitt_telenor_data, business_fri_20GB, [0], [1])
print(predicted_product)
print(to_name[str(predicted_product)])

