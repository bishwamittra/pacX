from pac_explanation import utils
from wrappers.xplainer_wrap import Xplainer_wrap
import time

filename = "xplainer/temp/compas_data/compas_data_nbestim_50_maxdepth_3_testsplit_0.2.mod.pkl"
xw = Xplainer_wrap(filename)
dataObj = xw.dataObj
df = dataObj.df


start_time = time.time()
generator = utils.random_generator(dataObj.X_train, dataObj.attribute_type)
for i in generator:
    pass
print(time.time() - start_time)