import tensorflow as tf
import numpy as np
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


#Pinning operations across devices and servers
reset_graph()

with tf.device("/job:ps"):
    a = tf.Variable(1.0, name="a")

with tf.device("/job:worker"):
    b = a + 2

with tf.device("/job:worker/task:1"):
    c = a + b

with tf.Session("grpc://127.0.0.1:2221") as sess:
    sess.run(a.initializer)
    print(c.eval())


reset_graph()

with tf.device(tf.train.replica_device_setter(
        ps_tasks=2,
        ps_device="/job:ps",
        worker_device="/job:worker")):
    v1 = tf.Variable(1.0, name="v1")  # pinned to /job:ps/task:0 (defaults to /cpu:0)
    v2 = tf.Variable(2.0, name="v2")  # pinned to /job:ps/task:1 (defaults to /cpu:0)
    v3 = tf.Variable(3.0, name="v3")  # pinned to /job:ps/task:0 (defaults to /cpu:0)
    s = v1 + v2            # pinned to /job:worker (defaults to task:0/cpu:0)
    with tf.device("/task:1"):
        p1 = 2 * s         # pinned to /job:worker/task:1 (defaults to /cpu:0)
        with tf.device("/cpu:0"):
            p2 = 3 * s     # pinned to /job:worker/task:1/cpu:0

config = tf.ConfigProto()
config.log_device_placement = True

with tf.Session("grpc://127.0.0.1:2221", config=config) as sess:
    v1.initializer.run()

