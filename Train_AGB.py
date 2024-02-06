from torch.utils.data.dataset import Subset
import random
import os
import torch
from src.Unet_regression import UNet
from src.mydataset import Dataset_reg as Dataset
import numpy as np
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import pandas as pd
from tqdm import tqdm
import resource
from pympler.tracker import SummaryTracker
import yaml
import csv

memory_tracker = SummaryTracker()


def using(point=""):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           ''' % (point, usage[0], usage[1],
                usage[2]/1024.0)


CONFIG_PATH = '/Code/NewModels/config/'


# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


nll_eval = lambda mu, sigma, y: np.log(sigma)/2 + ((y-mu)**2)/(2*sigma)
mse_eval = lambda mu, y: np.sqrt(np.mean(np.power(mu-y, 2)))


config = load_config("config.yaml")
outputfile = str(config["output"])+'/'+str(config["name"])+'/'+str(config["size_img"])+'/'+str(config["num_test"])+'/'

if not os.path.exists(outputfile):
    os.makedirs(outputfile)

# Open the CSV file for writing
with open(outputfile+'config.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['key', 'value'])

    # Write the data rows
    for key, value in config.items():
        writer.writerow([key, value])

with open(outputfile+'epoch_results.csv', 'w', newline='') as csvfile:
    # Create a CSV writer object
    writer = csv.writer(csvfile)

    # Write the header row
    writer.writerow(['epoch', 'loss', 'val_loss', 'train_acc', 'val_acc'])


m = 'Unet'
name = str(config["name"])+str(config["size_img"])+'_z+'+str(config["num_epochs"])+'ep_s_'+str(config["batch_size"])+'_WeightTest1_Albu_wd_10-5_channel_'+str(config["channel"])


cuda_device = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_dataset = Dataset('/TUILES/SIZE_'+str(config["size_img"])+'/tilefiles/TRAIN.txt', augment=config['augment'])
test_dataset = Dataset('/TUILES/SIZE_'+str(config["size_img"])+'/tilefiles/TEST.txt', augment=False)
valid_dataset = Dataset('/TUILES/SIZE_'+str(config["size_img"])+'/tilefiles/VALID.txt', augment=False)


train_loader = torch.utils.data.DataLoader(train_dataset,  num_workers=4,  batch_size=config["batch_size"], shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, num_workers=1,   batch_size=config["batch_size"], shuffle=False, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=1,   batch_size=config["batch_size"], shuffle=False, drop_last=True)
print("train", len(train_dataset))
print("test", len(test_dataset))
print("valid", len(valid_dataset))


dataloader = train_loader, test_loader


def Regression_accuracy(y_pred, y_true):
    mse = mean_squared_error(y_pred.cpu().detach().numpy().flatten(), y_true.cpu().detach().numpy().flatten())
    return mse


def pytorch_train_one_epoch(pytorch_network, optimizer, loss_function, scheduler):
    """
    Trains the neural network for one epoch on the train DataLoader.

    Args:
        pytorch_network (torch.nn.Module): The neural network to train.
        optimizer (torch.optim.Optimizer): The optimizer of the neural network
        loss_function: The loss function
        scheduler: Learning rate scheduler.

    Returns:
        A tuple (loss, accuracy) corresponding to an average of the losses and
        an average of the accuracy, respectively, on the train DataLoader.
    """

    pytorch_network.train(True)

    if scheduler:
        scheduler.step()

    with torch.enable_grad():
        loss_sum = 0
        acc_sum = 0
        example_count = 0

        for (x, y) in train_loader:
            # Transfer batch on GPU if needed.
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_pred = pytorch_network(x)

            loss = loss_function(y_pred.squeeze(), y)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(x)
            acc_sum += float(Regression_accuracy(y_pred, y))
            example_count += len(x)

    avg_loss = loss_sum / example_count
    avg_acc = acc_sum / example_count
    return avg_loss, avg_acc

def pytorch_test(pytorch_network, loader, loss_function):
    """
    Tests the neural network on a DataLoader.

    Args:
        pytorch_network (torch.nn.Module): The neural network to test.
        loader (torch.utils.data.DataLoader): The DataLoader to test on.
        loss_function: The loss function.

    Returns:
        A tuple (loss, accuracy) corresponding to an average of the losses and
        an average of the accuracy, respectively, on the DataLoader.
    """

    pred = []
    true = []
    sigma = []

    pytorch_network.eval()
    with torch.no_grad():
        loss_sum = 0.
        acc_sum = 0.
        example_count = 0
        for (x, y) in loader:
            # Transfer batch on GPU if needed.
            x = x.to(device)
            y = y.to(device)
            y_pred = pytorch_network(x)

            # GPU to CPU numpy conversion
            preds = y_pred.flatten().cpu().detach().numpy()
            preds = preds.astype(float)
            target = y.flatten().cpu().detach().numpy()
            target = target.astype(float)


            for i in range(len(preds)):
                pred.append(preds[i])
                true.append(target[i])

            loss = loss_function(y_pred.squeeze(), y)


            loss_sum += float(loss) * len(x)
            acc_sum += float(Regression_accuracy(y_pred, y))

            example_count += len(x)
    avg_loss = loss_sum / example_count
    avg_acc = acc_sum / example_count
    return avg_loss, avg_acc, pred, true,




def pytorch_train(pytorch_network, optimizer, scheduler):
    """
    This function transfers the neural network to the right device,
    trains it for a certain number of epochs, tests at each epoch on
    the validation set and outputs the results on the test set at the
    end of training.

    Args:
        pytorch_network (torch.nn.Module): The neural network to train.
        scheduler: Learning rate scheduler.

    Example:
        This function displays something like this:

        .. code-block:: python

            Epoch 1/5: loss: 0.5026924496193726, acc: 84.26666259765625, val_loss: 0.17258917854229608, val_acc: 94.75
            Epoch 2/5: loss: 0.13690324830015502, acc: 95.73332977294922, val_loss: 0.14024296019474666, val_acc: 95.68333435058594
            Epoch 3/5: loss: 0.08836929737279813, acc: 97.29582977294922, val_loss: 0.10380942322810491, val_acc: 96.66666412353516
            Epoch 4/5: loss: 0.06714504160980383, acc: 97.91874694824219, val_loss: 0.09626663728555043, val_acc: 97.18333435058594
            Epoch 5/5: loss: 0.05063822727650404, acc: 98.42708587646484, val_loss: 0.10017542181412378, val_acc: 96.95833587646484
            Test:
                Loss: 0.09501855444908142
                Accuracy: 97.12999725341797
    """



    # Transfer weights on GPU if needed.
    pytorch_network.to(device)

    loss_function = nn.MSELoss()   #for regression

    for epoch in tqdm(range(1, config["num_epochs"] + 1)):
        # Print Learning Rate
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        # Training the neural network via backpropagation
        train_loss, train_acc = pytorch_train_one_epoch(pytorch_network, optimizer, loss_function, scheduler)
        scheduler.step()

        # Validation at the end of the epoch
        valid_loss, valid_acc, _, _, _ = pytorch_test(pytorch_network, valid_loader, loss_function)

        print("Epoch {}/{}: loss: {}, val_loss: {}, train_acc: {}, val_acc: {}".format(
            epoch, config["num_epochs"], train_loss,  valid_loss, train_acc, valid_acc
        ))
        # Write the loss results to a CSV file
        with open(outputfile+'epoch_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, valid_loss, train_acc, valid_acc])

    # Test at the end of the training
    test_loss, test_acc, pred, true = pytorch_test(pytorch_network, test_loader, loss_function)
    print('Test:\n\tLoss: {}\n\tAccuracy: {}'.format(test_loss, test_acc))
    return test_loss, test_acc, pred, true


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    mus = []
    trues = []


    model = UNet(n_classes=1, padding=True, up_mode='upconv', in_channels=config["channel"], depth=config["depth"]).to(device)
    op = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(op, 0.95)

    test_loss, test_acc, pred, true = pytorch_train(model, op, scheduler)
    dftestresult = pd.DataFrame({'Pred': pred, 'True': true})
    dftestresult.to_parquet(outputfile+'/TestResult.parquet', index=False)

    mus.append(pred)
    trues.append(true)

    PATH = outputfile+name+'modelnum.pt'
    PATHout2 = outputfile+name+'inf_modelnum.pt'

    torch.save(model.state_dict(), PATH)
    torch.save(model, PATHout2)
