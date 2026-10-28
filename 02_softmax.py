import numpy as np
print(np.exp(np.array([0.0,1.0,2.0])))
print(np.log(np.exp(np.array([0.0,1.0,2.0]))))
def softmax(logits):
    shifted = logits-np.max(logits,axis=1,keepdims=True)
    exps=np.exp(shifted)
    return exps/exps.sum(axis=1,keepdims=True)

logits=np.array([[1.0,2.0,3.0],[1.0,1.0,1.0]])
probs=softmax(logits)
assert probs.shape == (2, 3), probs.shape
assert np.allclose(probs.sum(axis=1), 1), probs.sum(axis=1)

big=np.array([[1000.0,1001.0,1002.0]])
print(softmax(big))
assert np.allclose(softmax(big),probs[:1]),softmax(big)

labels=np.array([2,0])
print(probs[np.arange(2),labels])

def cross_entropy(probs,labels):
    picked=probs[np.arange(len(labels)),labels]
    return  (-np.log(picked)).mean()
print(cross_entropy(probs,labels))
print(probs.argmax(axis=1))
print((probs.argmax(axis=1)==labels).mean())