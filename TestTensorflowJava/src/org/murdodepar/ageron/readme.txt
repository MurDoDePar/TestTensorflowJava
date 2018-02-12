Exemple du livre Oreilly 
Deep Learning avec tensorFlow de Aurelien geron
https://github.com/ageron/handson-ml

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

source env/bin/activate
tensorboard --logdir tf_logs/
tensorboard --logdir "C:\Users\domin\Google Drive\Code\git\TestTensorflowJava\TestTensorflowJava\src\org\murdodepar\ageron\tf_logs"
Starting TensorBoard on port 6006