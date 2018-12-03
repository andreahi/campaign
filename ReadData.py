import pandas as pd
import numpy as np

from FileUtils import save_obj

product_indexes = {}
def to_index_bag(products):
    products = eval(products)
    products = list(filter(lambda x: x in product_indexes, products))
    l = list(map(lambda x: product_indexes[x], products))
    vector = np.zeros(len(product_indexes))
    vector[l] = 1
    return vector

target_indexs = {}
def to_index_bag_target(products):
    products = eval(products)
    products = list(filter(lambda x: x in target_indexs, products))
    l = list(map(lambda x: target_indexs[x], products))
    vector = np.zeros(len(target_indexs))
    vector[l] = 1
    return vector

df = pd.read_csv("filename.txt", converters={'history': eval, 'currentProducts': eval, 'newProducts': eval})

history = df.history

products = []
products_count = {}
for e in history:
    products = products + e
    for e2 in e:
        if e2 in products_count:
            products_count[e2] += 1
        else:
            products_count[e2] = 1
for e in df.currentProducts:
    products = products + e
    for e2 in e:
        if e2 in products_count:
            products_count[e2] += 1
        else:
            products_count[e2] = 1

products = sorted(set(products))
products = list(filter(lambda x:  products_count[x] > 50, products))
for i, e in enumerate(products):
    product_indexes[e] = i


target_products = []
target_products_count = {}
for e in df.newProducts:
    target_products = target_products + e
    for e2 in e:
        if e2 in target_products_count:
            target_products_count[e2] += 1
        else:
            target_products_count[e2] = 1
target_products = sorted(set(target_products))
target_products = list(filter(lambda x:  target_products_count[x] > 50, target_products))
for i, e in enumerate(target_products):
    target_indexs[e] = i
    
print(product_indexes)
print(target_indexs)

save_obj(product_indexes, "product_indexes")
save_obj(target_indexs, "target_indexs")

df = pd.read_csv("filename.txt", converters={'history': to_index_bag, 'currentProducts': to_index_bag, 'newProducts': to_index_bag_target})
df = df[[e.any() for e in df.newProducts]]

print(df)
df.to_json("filename_formated.txt")

indexes = list(range(len(df)))
np.random.shuffle(indexes)
print("len at end: " , len(df))
df.iloc[indexes[:200],:].to_json("test.txt")
df.iloc[indexes[200:],:].to_json("train.txt")

print("history len: ", len(df.history.iloc[0]))
print("birthDate : ", df.birthDate.iloc[0])
print("currentProducts len: ", len(df.currentProducts.iloc[0]))
print("newProducts len: ", len(df.newProducts.iloc[0]))

