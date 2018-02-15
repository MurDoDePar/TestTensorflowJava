import tensorflow as tf
import numpy as np

n_epochs = 1000                                                                       # not shown in the book
learning_rate = 0.01  

from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#root_logdir = "tf_logs"
root_logdir = "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/tf_logs"
logdir = "{}/DoM-{}/".format(root_logdir, now)

# Batch of input and target output (1x1 matrices)
x = tf.placeholder(tf.float32, shape=[None, 1], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

# Trivial linear model
y_pred = tf.identity(tf.layers.dense(x, 1), name='predictions')

error = tf.reduce_mean(tf.square(y_pred - y), name='error')

mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
training_op = optimizer.minimize(mse)                                                 # not shown

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

saver = tf.train.Saver()

# tf.train.Saver.__init__ adds operations to the graph to save
# and restore variables.
saver_def = tf.train.Saver().as_saver_def()

print('Run this operation to initialize variables     : ', init.name)
print('Run this operation for a train step            : ', y_pred.name)

with tf.Session() as sess:
    # Write the graph out to a file.
    tf.train.write_graph(sess.graph_def,'C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/graph/','graph_DoM.pb',False)
    sess.run(init)

    print("n_epochs ", n_epochs)
    
    for epoch in range(n_epochs):
        #print("epoch ", epoch)
        #x_train = sess.run(x_batch)
        #y_train = sess.run(y_batch)
        x_train = np.random.rand(2,1)
        y_train = x_train * 2 + 3
        #print("x ", x_train, " y ", y_train)
        sess.run(training_op, feed_dict={x: x_train, y: y_train})
        #print("sess.run training_op")
        if epoch % 100 == 0 and epoch != 0:
            print("Epoch", epoch, "MSE =", mse.eval(feed_dict={x: x_train, y: y_train}))# not shown  
            save_path = saver.save(sess, "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_DoM.ckpt")
        
        #print("boucle")
        
    #best_theta = theta.eval()
    save_path = saver.save(sess, "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_DoM_final.ckpt")
    print("error ", error)

with tf.Session() as sess:
    saver.restore(sess, "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_DoM_final.ckpt")


sess.close()


