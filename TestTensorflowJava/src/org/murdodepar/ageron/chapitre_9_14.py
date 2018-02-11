import tensorflow as tf
import chapitre_9_01 as ch1

ch1.reset_graph()

x1 = tf.Variable(1)
print(x1.graph is tf.get_default_graph())

graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

print(x2.graph is graph)

print(x2.graph is tf.get_default_graph())

w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())  # 10
    print(z.eval())  # 15

with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)  # 10
    print(z_val)  # 15