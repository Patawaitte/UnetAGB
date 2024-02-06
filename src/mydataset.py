#dataset à partir de Databyfolder
#Les dalles train, valid... doivent être enregistrer dans les fichiers différents, et le dataloader devra être appellé en fonction des fichiers



import torch.utils.data
import os
import rasterio
import torch
import numpy as np
import matplotlib.pyplot as plt
import provider as provider

def normalize(array, min, max):
    return (array - min) / (max - min)


def readlabel(file):
    with rasterio.open(file) as src:
        array = src.read(1)
        label = array
        label =torch.from_numpy(label)
        return label

def readimg(file):
    with rasterio.open(file) as src:

        b1 = src.read(1)/10000
        b2 = src.read(2)/10000
        b3 = src.read(3)/10000
        b4 = src.read(4)/10000
        b5 = src.read(5)/10000
        b6 = src.read(6)/10000

        srcnorm = np.dstack((b1, b2, b3, b4, b5,b6))
        srcnorm = np.transpose(srcnorm,(2,0,1))
        srcnorm =torch.from_numpy(srcnorm)
        return srcnorm

LABEL_FILENAME="Label.tif"

class Dataset_reg(torch.utils.data.Dataset):
    def __init__(self, tileids, augment):
        self.tileids = tileids

        self.samples = list()
        self.classes = [ 0,  1,  2,  3,  4,  5]
        self.augment = augment

        with open(self.tileids , 'r') as f:
            files = [el.replace("\n", "") for el in f.readlines()]

        for f in files:
            self.samples.append(f)

    def __len__(self):
        return len(self.samples)

    def _augment_batch_data(self, x, y):

        rotated_data, rotated_label = provider.rotate_flipud(x, y)
        rotated_data, rotated_label = provider.rotate_flipud(rotated_data, rotated_label)
        rotated_data, rotated_label = provider.rotate_random(rotated_data, rotated_label)

        return rotated_data, rotated_label


    def __getitem__(self, idx):
            path = os.path.join(self.tileids, self.samples[idx])
            label = readlabel(os.path.join(path,LABEL_FILENAME))

            Summer = readimg(os.path.join(path, "Landsat.tif"))
            Winter = readimg(os.path.join(path, "Winter.tif"))

            Summer= np.array(Summer)
            Winter= np.array(Winter)
            #x=Summer   #to test the model for only one saison

            x = np.concatenate((Summer, Winter), axis=0)


            if self.augment:
                x, label = self._augment_batch_data(x, label)
                label = torch.from_numpy(label.copy())
                x =torch.from_numpy(x.copy())

                return x.float(), label.float()
            else:

                label = label
                x =torch.from_numpy(x)

                return x.float(), label.float()


#Test
if __name__=="__main__":

    dataset = Dataset_reg('/TUILES/V3/tilefiles/TRAIN.txt', augment=False)
    a, b = dataset[89]
    array = np.array(a[0])
    fig, (ax1,ax2)=plt.subplots(1,2,figsize=(5,5))
    ax1.imshow(a[4])
    ax2.imshow(b)

    plt.show()
