#Saving and restoring a model (50)
import numpy as np
import tensorflow as tf
import chapitre_9_01 as ch

ch.reset_graph()

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000                                                                       # not shown in the book
learning_rate = 0.01                                                                  # not shown

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")            # not shown
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")            # not shown
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")                                      # not shown
error = y_pred - y                                                                    # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
training_op = optimizer.minimize(mse)                                                 # not shown

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())                                # not shown
            save_path = saver.save(sess, "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_ch_9_50.ckpt")
        sess.run(training_op)
    
    best_theta = theta.eval()
    save_path = saver.save(sess, "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_ch_9_50_final.ckpt")

print("best_theta")
print(best_theta)

with tf.Session() as sess:
    saver.restore(sess, "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_ch_9_50_final.ckpt")
    best_theta_restored = theta.eval() # not shown in the book

np.allclose(best_theta, best_theta_restored)

#If you want to have a saver that loads and restores theta with a different name, such as "weights":
saver = tf.train.Saver({"weights": theta})


ch.reset_graph()
# notice that we start with an empty graph.

saver = tf.train.import_meta_graph("C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_ch_9_50_final.ckpt.meta")  # this loads the graph structure
theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

with tf.Session() as sess:
    saver.restore(sess, "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_ch_9_50_final.ckpt")  # this restores the graph's state
    best_theta_restored = theta.eval() # not shown in the book

np.allclose(best_theta, best_theta_restored)
