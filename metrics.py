import numpy as np

def prep_im(im, index):
    return im == index

def classIoU(ground:np.array, pred:np.array, index):
    """
    Generates the intersect-over-union between two images.

    Args:
        ground (numpy.array): describes the ground truth for each pixel's class
        pred (numpy.array): describes the predicted classes
        index (int): class indentifier

    Returns:
        int: intersect over union for the class defined by index
    """
    pred = prep_im(pred, index)
    ground = prep_im(ground, index)
    
    intersect = np.sum(ground&pred)
    union = np.sum(ground|pred)
    IoU=intersect/union if union else 1.
    return IoU

def classAccuracy(ground, pred, index):
    """
    Generates the accuracy between two images.

    Args:
        ground (numpy.array): describes the ground truth for each pixel's class
        pred (numpy.array): describes the predicted classes
        index (int): class indentifier

    Returns:
        int: accuracy for the class defined by index
    """
    pred = prep_im(pred, index)
    ground = prep_im(ground, index)

    true_vals = np.logical_not(np.logical_xor(pred, ground)).sum()
    return true_vals/ground.size

def accuracy(pred, ground, num_classes):
    acc = np.array([])
    for i in range(num_classes):
      acc = np.append(acc, [classAccuracy(ground, pred, i)])
    return acc
def meanIoU(ground, pred, num_classes):
    iou = []
    for i in range(num_classes):
        iou.append(classIoU(ground, pred, i))
    return np.mean(iou)

def meanAccuracy(ground, pred, num_classes):
    acc = []
    for i in range(num_classes):
        acc.append(classAccuracy(ground, pred, i))
    return np.mean(acc)