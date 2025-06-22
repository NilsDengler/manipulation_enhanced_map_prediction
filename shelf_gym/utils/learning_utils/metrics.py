def map_accuracy(model_out,gt):
    pred = model_out.argmax(axis = 1)
    gt = (gt>0.51).int()
    return (pred==gt).sum()/gt.flatten().shape[0]

def semantic_accuracy(model_out,gt):
    pred = model_out.argmax(axis = 1)
    return (pred==gt).sum()/gt.flatten().shape[0]
