#%matplotlib inline
import chapitre_2_04 as ch
import matplotlib.pyplot as plt

ch.housing.hist(bins=50, figsize=(20,15))
#save_fig("attribute_histogram_plots")
plt.savefig("./img/attribute_histogram_plots")
plt.show()
