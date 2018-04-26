import tensorflow as tf
from core import *
from tensorflow.python.framework import ops

ops.reset_default_graph()

# Create graph
sess = tf.Session()

# Load the data

x_vals_train1 ,y_vals_train1=load_training_dataframe(model=1)
x_vals_test ,y_vals_test = load_testing_dataframe(model=1)

# Random sample


x_vals_train = x_vals_train1[0:100]
y_vals_train = y_vals_train1[0:100]
# Declare k-value and batch size

k_max = 20
batch_size = 6

# Placeholders
x_data_train = tf.placeholder( dtype=tf.float32)
x_data_test = tf.placeholder(dtype=tf.float32)
y_target_train = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target_test = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Declare distance metric
history = [list(),list()]
for k in range(k_max):
    distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test, 1))), axis=2)

    # Predict: Get min distance index (Nearest neighbor)
    top_k_xvals, top_k_indices = tf.nn.top_k(tf.negative(distance), k=k)
    prediction_indices = tf.gather(y_target_train, top_k_indices)
    # Predict the mode category
    count_of_predictions = tf.reduce_sum(prediction_indices, axis=1)
    prediction = tf.argmax(count_of_predictions, axis=1)

    # Calculate how many loops over training data
    num_loops = int(np.ceil(len(x_vals_test) / batch_size))
    test_output = []
    actual_vals = []
    for i in range(num_loops):
        min_index = i * batch_size
        max_index = min((i + 1) * batch_size, len(x_vals_test))
        x_batch = x_vals_test[min_index:max_index]
        y_batch = y_vals_test[min_index:max_index]
        predictions = sess.run(prediction, feed_dict={x_data_train: x_vals_train, x_data_test: x_batch,
                                                      y_target_train: y_vals_train, y_target_test: y_batch})
        test_output.extend(predictions)
        actual_vals.extend(np.argmax(y_batch, axis=1))

    test_size = len(x_vals_test)
    accuracy = sum([1. / test_size for i in range(test_size) if test_output[i] == actual_vals[i]])
    history[0].append(k)
    history[1].append(accuracy)
    print('Accuracy on test set: ' + str(accuracy))

plt.plot(history[0],history[1])
plt.xlabel("K")
plt.ylabel("validation Accuracy")
plt.title("Accuracy ")
plt.show()