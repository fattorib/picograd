from MiniNN.Tensor import *
import numpy as np


# Evaluate model on test data


def eval_acc(model, dataloader):
    accuracy = 0
    for inputs, labels in dataloader:
        logps = model.forward(inputs)
        # Calculate accuracy
        ps = np.exp(logps.value)
        # top_p, top_class = ps.topk(1, dim=1)
        top_class = np.argmax(ps, axis=1)
        raw_labels = dataloader.raw_labels
        equals = np.equal(top_class, raw_labels)
        accuracy += np.mean(equals)

    print('Accuracy:', 100*accuracy/len(dataloader), '%')
