import pandas as pd
import numpy as np


product_indexes = {}
def to_index_bag(products):
    products = eval(products)
    l = list(map(lambda x: product_indexes[x], products))
    vector = np.zeros(len(product_indexes))
    vector[l] = 1
    return vector

target_indexs = {}
def to_index_bag_target(products):
    products = eval(products)
    l = list(map(lambda x: target_indexs[x], products))
    vector = np.zeros(len(target_indexs))
    vector[l] = 1
    return vector

df = pd.read_csv("filename.txt", converters={'history': eval, 'currentProducts': eval, 'newProducts': eval})

history = df.history

products = []
for e in history:
    products = products + e
for e in df.currentProducts:
    products = products + e

products = sorted(set(products))
for i, e in enumerate(products):
    product_indexes[e] = i


target_products = []
for e in df.newProducts:
    target_products = target_products + e
target_products = sorted(set(target_products))
for i, e in enumerate(target_products):
    target_indexs[e] = i
    
print(product_indexes)
print(target_indexs)



df = pd.read_csv("filename.txt", converters={'history': to_index_bag, 'currentProducts': to_index_bag, 'newProducts': to_index_bag_target})

print(df)
df.to_json("filename_formated.txt")

print("history len: ", len(df.history[0]))
print("birthDate : ", df.birthDate[0])
print("currentProducts len: ", len(df.currentProducts[0]))
print("newProducts len: ", len(df.newProducts[0]))

