import tensorflow as tf
#Cluster
cluster_spec = tf.train.ClusterSpec({
    "ps": [
        "127.0.0.1:2221",  # /job:ps/task:0
        "127.0.0.1:2222",  # /job:ps/task:1
    ],
    "worker": [
        "127.0.0.1:2223",  # /job:worker/task:0
        "127.0.0.1:2224",  # /job:worker/task:1
        "127.0.0.1:2225",  # /job:worker/task:2
    ]})


task_ps0 = tf.train.Server(cluster_spec, job_name="ps", task_index=0)
task_ps1 = tf.train.Server(cluster_spec, job_name="ps", task_index=1)
task_worker0 = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
task_worker1 = tf.train.Server(cluster_spec, job_name="worker", task_index=1)
task_worker2 = tf.train.Server(cluster_spec, job_name="worker", task_index=2)

