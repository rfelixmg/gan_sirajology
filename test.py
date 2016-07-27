from theano.sandbox.cuda.dnn import dnn_available as da;

print(da() or da.msg)