import numpy as np

def softmax(logits):
    shifted = logits-np.max(logits,axis=1,keepdims=True)
    exps=np.exp(shifted)
    return exps/exps.sum(axis=1,keepdims=True)

def cross_entropy(probs,labels):
    picked=probs[np.arange(len(labels)),labels]
    return  (-np.log(picked)).mean()

def relu(pre_relu):
    return np.maximum(0,pre_relu)

def init_params(input_size,hidden_size,class_count):
    weight1=np.random.randn(input_size,hidden_size)*0.01
    bias1=np.zeros(hidden_size)
    weight2=np.random.randn(hidden_size, class_count) * 0.01
    bias2 = np.zeros(class_count)
    return weight1, bias1, weight2, bias2

def forward(images,weight1,bias1,weight2,bias2):
    pre_relu=images@weight1+bias1
    hidden=relu(pre_relu)
    logits=hidden@weight2+bias2
    probs=softmax(logits)
    return pre_relu,hidden,probs

