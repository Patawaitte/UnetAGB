import numpy as np

def rotate_fliplr(data, lb):
    data_lr = np.fliplr(data)
    lb_lr = np.fliplr(lb)
    return data_lr, lb_lr

def rotate_flipud(data, lb):
    data_ud = np.flipud(data)
    lb_ud = np.flipud(lb)
    return data_ud, lb_ud

def rotate_random(data, lb):
    angle = np.random.choice([1, 2, 3])
    data_rd = np.rot90(data, angle, axes=(1, 2))
    lb_rd = np.rot90(lb, angle, axes=(0, 1))

    return data_rd, lb_rd


    return transformed, labelt
