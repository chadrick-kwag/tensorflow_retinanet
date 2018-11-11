# retinanet in tensorflow

original codebase from the `tensorflow/model` repo. Extracted only the files that were needed to run the estimator in retinanet.
However, it has been modfied from this basis. The data preparation has been modified greatly due to different number of groundtruth boxes in a single image. Parsing this into anchor matrix will be done manually with numpy.