from src.InferenceDataset import  MyDataset as MyDataset
from torch.utils.data.dataset import Subset
import torch
import numpy as np

import rasterio
from tqdm import tqdm
from rasterio.crs import CRS
import os


my_wkt="""PROJCRS["NAD_1983_Canada_Lambert",BASEGEOGCRS["NAD83",DATUM["North American Datum 1983", ELLIPSOID["GRS 1980",6378137,298.257222101004, LENGTHUNIT["metre",1]],ID["EPSG",6269]],PRIMEM["Greenwich",0,ANGLEUNIT["Degree",0.0174532925199433]]],CONVERSION["unnamed", METHOD["Lambert Conic Conformal (2SP)", ID["EPSG",9802]],PARAMETER["Latitude of false origin",0,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8821]], PARAMETER["Longitude of false origin",-95,ANGLEUNIT["Degree",0.0174532925199433], ID["EPSG",8822]], PARAMETER["Latitude of 1st standard parallel",49, ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8823]],PARAMETER["Latitude of 2nd standard parallel",77,ANGLEUNIT["Degree",0.0174532925199433],ID["EPSG",8824]],PARAMETER["Easting at false origin",0,LENGTHUNIT["metre",1],ID["EPSG",8826]],PARAMETER["Northing at false origin",0,LENGTHUNIT["metre",1], ID["EPSG",8827]]],CS[Cartesian,2],AXIS["(E)",east,ORDER[1], LENGTHUNIT["metre",1,ID["EPSG",9001]]],AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]"""

dst_crs = CRS.from_wkt(my_wkt)

outputfile='/Outputfile/'
inputfiles='/LandsatSummer/'
inputfilew='/LandsatWinter/'

if not os.path.exists(outputfile):
        os.makedirs(outputfile)

# Read the image
def readimg(file):
    with rasterio.open(file) as src:

        b1 = src.read(1)/1000
        b2 = src.read(2)/1000
        b3 = src.read(3)/1000
        b4 = src.read(4)/1000
        b5 = src.read(5)/1000
        b6 = src.read(6)/1000

        srcnorm = np.dstack((b1, b2, b3, b4, b5,b6))
        srcnorm = np.transpose(srcnorm,(2,0,1))
        return srcnorm

# Inference function to predict the AGB with overlapping sliding windows
def inference(pytorch_network, loader, importancemap, windows):

    pytorch_network.eval()
    with torch.no_grad():

        #template
        result = np.zeros((data.shape[1],  data.shape[2], ), dtype='float32')
        overlap = np.zeros((data.shape[1],  data.shape[2], ), dtype='float32')

        for batch, pt in tqdm(loader):
            x= batch
            starting_points = pt

            # Transfer batch on GPU for prediction
            x = x.to(device)
            y_pred = pytorch_network(x)
            pred=y_pred[0].squeeze()

            #Delete the border of the tiles to avoid artifacts
            pred[:,[0, 1], :] = pred[:,[-2, -1], :] = pred[:,:, [0, 1]] = pred[:,:, [-2, -1]] = 0
            importancemap[[0, 1], :] = importancemap[[-2, -1], :] = importancemap[:, [0, 1]] = importancemap[:, [-2, -1]] = 0

            pred = pred.cpu().numpy()

            # Fill the template with the predictions
            for i in range(len(pred)):
                xs = starting_points[0][i]
                ys = starting_points[1][i]

                result[xs:xs + windows[0], ys:ys + windows[0]] += pred[i]
                overlap[ xs:xs + windows[0], ys:ys + windows[0]] += importancemap

                meanresult=result / overlap

    return meanresult

#Tranform numpy to tiff function
def toTIFF(dfn, name):
    dfn.to_csv(name+".xyz", index = False, header = None, sep = ",")


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

PATH1='/save/S_W_V15/UNet64v3_inf_modelnum_0.pt'


ListName = [PATH1, ]


for y in tqdm(range(2015, 2021)):
    filename=os.path.join(outputfile+'AGBmean_Unet_'+str(y)+'.tif')

    if os.path.exists(filename):
        print(f"{filename} exists. Skipping...")
    else:

        files=inputfiles+'clip_Landsat_summer_'+str(y)+'.tif'
        filew=inputfilew+'clip_Landsat_winter_'+str(y)+'.tif'


        WINDOW_SIZE    = (64, 64)   #Size of the window
        STRIDE         = (32, 32)   #Overlap of the window
        IMPORTANCE_MAP = np.ones((*WINDOW_SIZE,), dtype='float32')

        dataset = MyDataset(filew, files, WINDOW_SIZE, STRIDE )
        data =readimg(filew)
        patch_means, patch_variances = [], []
        for path in ListName:
            model = torch.load(path)
            model.eval()
            cuda_device = 0
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Training hyperparameters
            batch_size = 512
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

            resultat = inference(model, test_loader, IMPORTANCE_MAP, WINDOW_SIZE)
            torch.cuda.synchronize()


        # Export raster
        # Write to TIFF
            raster = rasterio.open(filew)
            red = raster.read(4)
            kwargs = raster.meta
            kwargs.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw')

            with rasterio.open(filename, 'w', **kwargs) as dst:
                dst.write_band(1, resultat.astype(rasterio.float32))






