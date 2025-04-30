import numpy as np

def prep_im(im, index):
    im = im.copy()
    im[im!=index]=-1
    im[im==index]=0
    im+=1
    return im.astype(np.uint8)

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

    true_vals = np.sum(ground==pred)
    return true_vals/(ground.shape[0]*ground.shape[1])

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