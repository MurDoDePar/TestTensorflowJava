Exemple du livre Oreilly 
Deep Learning avec tensorFlow de Aurelien geron
https://github.com/ageron/handson-ml

pip3 install --upgrade tensorflow
pip3 install --upgrade tensorflow-gpu

pip install numpy
pip install matplotlib
pip install pandas 
pip install sklearn
pip install scipy
pip install IPython

import os
os.getcwd()
os.chdir("C:\\Users\\domin\\Google Drive\\Code\git\\TestTensorflowJava\\TestTensorflowJava\\src\org\\murdodepar\\ageron")
os.listdir(".")
exec(open("./chapitre_2_01.py").read())


from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
#root_logdir = "tf_logs"
root_logdir = "C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

source env/bin/activate
#tensorboard --logdir tf_logs/
tensorboard --logdir "C:\Users\domin\Google Drive\Code\git\TestTensorflowJava\TestTensorflowJava\tf_logs"
#tensorboard --logdir "C:\Users\domin\Google Drive\Code\git\TestTensorflowJava\TestTensorflowJava\src\org\murdodepar\ageron\tf_logs"
Starting TensorBoard on port 6006



C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_ch_9_50.ckpt


C:/Users/domin/Google Drive/Code/git/TestTensorflowJava/TestTensorflowJava/model/model_ch_9_50_final.ckpt