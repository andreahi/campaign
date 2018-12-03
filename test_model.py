import numpy as np

from FileUtils import load_obj
import numpy as np

from FileUtils import load_obj


def test_model(sess, df, birth_date, history, current_products, consumer_product, business_product, new_product, loss, optimizer, out, training):

    axis_sum = np.sum(np.array(list(np.array(df.newProducts))), axis=1)
    if max(axis_sum) > 11 or min(axis_sum) != 1 :
        print(max(axis_sum))
        print(min(axis_sum))
        print("ERROR: wrong dim sum")
        exit(12)


    birth_dates = np.array(list(np.array(df.birthDate, dtype=float))).reshape(-1, 1)
    min_max_scaler = load_obj("min_max_scaler")
    birth_dates = min_max_scaler.transform(birth_dates)
    birth_dates = np.squeeze(birth_dates)

    test_loss, predicted_new_products = sess.run([loss, out], feed_dict={
                birth_date: birth_dates,
                history: np.array(list(np.array(df.history))),
                current_products: np.array(list(np.array(df.currentProducts))),
                new_product: np.array(list(np.array(df.newProducts))),
        consumer_product: np.array(list(np.array(df.ConsumerType)), dtype=float),
        business_product: np.array(list(np.array(df.BusinessType)), dtype=float),
        training: False
    })

    predicted = np.argmax(predicted_new_products, axis=1)
    correct = np.argmax(np.array(list(np.array(df.newProducts))), axis=1)

    #correct_masked = np.array(correct == predicted, dtype=int) * (predicted)
    #ascii_histogram(correct_masked)

    return test_loss, np.sum(predicted == correct) / len(correct)




