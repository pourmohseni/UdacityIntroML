
import argparse
import torch
from torchvision import datasets, transforms, models
import time
from torch import nn, optim
from collections import OrderedDict
import json
import os

TRAIN = 'train'
VALIDATION = 'valid'
TEST = 'test'


def parse_args():
    """
    method to parse the command line arguments
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory containing the training and validation data")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save the trained model")
    parser.add_argument("--arch", type=str, default="vgg16", choices=["vgg11", "vgg16", "vgg19", "densenet121", "densenet161"], help="Pre-trained model from the given options from torchvision.models to be used for transfer learning")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate of the model (float)")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the model's classifier (integer)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (integer)")
    parser.add_argument("--gpu", action='store_true', help="Use GPU (set true if GPU is available)")

    args = parser.parse_args()

    return args

def load_data(train_dir, valid_dir, test_dir):
    """
    reads the input datasets (train, validation, tes) from the given directories and applies the necessary transforms
    :param train_dir: directory containing the training data
    :param valid_dir: directory containing the validation data
    :param test_dir: directory containing the test data
    :return:
    """
    # define data transforms
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    image_datasets = {TRAIN: datasets.ImageFolder(train_dir, transform=train_transforms),
                      VALIDATION: datasets.ImageFolder(valid_dir, transform=valid_transforms),
                      TEST: datasets.ImageFolder(test_dir, transform=test_transforms)}

    # defining the dataloaders
    dataloaders = {TRAIN: torch.utils.data.DataLoader(image_datasets['train'], batch_size=40, shuffle=True),
                   VALIDATION: torch.utils.data.DataLoader(image_datasets['valid'], batch_size=40),
                   TEST: torch.utils.data.DataLoader(image_datasets['test'], batch_size=40)}

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    return dataloaders, cat_to_name, image_datasets[TRAIN].class_to_idx

def validate(model, testloader, criterion, device):
    """
    performs model validation using the data provided by the testloader and the given criterion for error evaluation
    :param model:
    :param testloader:
    :param criterion:
    :param device:
    :return:
    """
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        # Move input/label tensors to the device
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # get the output probabilities
        probabilities = torch.exp(output)

        # Find the predicted class (the class with highest probability)
        top_p, top_class = probabilities.topk(1, dim=1)

        # Compare the predicted class with true label
        equality = top_class == labels.view(*top_class.shape)

        # Calculate prediction accuracy
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def train(model, trainloader, testloader, criterion, optimizer, epochs=2, print_every=10, device='cpu'):
    """
    peforms model training
    :param model:
    :param trainloader:
    :param testloader:
    :param criterion:
    :param optimizer:
    :param epochs:
    :param print_every:
    :param device:
    :return: the trained model
    """
    steps = 0
    running_loss = 0
    start_time = time.time()
    
    model.to(device)
    for e in range(epochs):

        # enable dropout by switching to training mode
        model.train()

        for images, labels in trainloader:
            steps += 1
            # Move input/label tensors to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # disable dropout by switching to evaluation (inference) mode
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    valid_loss, accuracy = validate(model, testloader, criterion, device)

                print(f"Epoch {e + 1}/{epochs}, "
                      f"Train loss: {running_loss / print_every:.4f}, "
                      f"Validation loss: {valid_loss / len(testloader):.4f}, "
                      f"Validation accuracy: {accuracy / len(testloader):.4f}")

                running_loss = 0

                # enable dopouts again by switching back to training mode
                model.train()

    end_time = time.time()
    elapesd_time = end_time - start_time
    print('Training completed.\n', 'Training time: ', elapesd_time)
    return model

def get_pretrained_model(model_arch="vgg16"):
    """
    get the pre-trained model
    :param model_arch:
    :return:
    """
    if model_arch == "vgg11":
        model = models.vgg11(pretrained=True)
        in_features = 25088
    elif model_arch == "vgg16":
        model = models.vgg16(pretrained=True)
        in_features = 25088
    elif model_arch == "vgg19":
        model = models.vgg19(pretrained=True)
        in_features = 25088
    elif model_arch == "densenet121":
        model = models.densenet121(pretrained=True)
        in_features = 1024
    elif model_arch == "densenet161":
        model = models.densenet161(pretrained=True)
        in_features = 1024
    else:
        print("pretrained architecture " , model_arch , " is not part of the options")
    return model, in_features

def build_model(model, in_features, hidden_units=512, num_categories=102, learning_rate=0.001):
    """

    :param model: pre-trained network architecture
    :param in_features: in_features parameter of the pre-trained model's classifier
    :param hidden_units: number of hidden units in the model's classifier
    :param num_categories: number of output categories
    :param learning_rate: learning rate
    :return: the built model, the criterion, and the optimizer fir the classifier part
    """
    # Freeze model parameters (prevent backpropagation through them)
    for param in model.parameters():
        param.requires_grad = False

    # set the model's classifier
    model.classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dpo1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, num_categories)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer

def get_device(gpu=False):
    """
    sets the correct device to be used. GPU is used if the gpu flag is set and cuda is available, otherwise, CPU is used.
    :param gpu:
    :return: current device's name
    """
    if not gpu:
        current_device = 'cpu'
    else:
        current_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if current_device == 'cpu':
            print("GPU is not available...")
    return current_device

def test(model, testloader, criterion, device):
    """
    tests the trained model against the test data
    :param model:
    :param testloader:
    :param criterion:
    :param device:
    :return:
    """
    # disable dropout by switching to evaluation (inference) mode
    model.eval()

    # Turn off gradients for validation
    with torch.no_grad():
        test_loss, accuracy = validate(model, testloader, criterion, device)

    # enable dopouts again by switching back to training mode
    model.train()

    print("Test Accuracy: {:.5f}".format(accuracy / len(testloader)))


def save_checkpoint(model, input_size, output_size, pre_trained_model, train_epochs, learning_rate, optimizer, save_dir):
    checkpoint = {'input_size': input_size,
                  'output_size': output_size,
                  'pretrained_model': pre_trained_model,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'train_epochs': train_epochs,
                  'learning_rate': learning_rate,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}

    torch.save(checkpoint, os.path.join(save_dir , 'model_checkpoint.pth'))

def checked_dir(dir, make_dir=False):
    if not os.path.isdir(dir):
        if make_dir:
            os.mkdir(dir)
    if not os.path.isdir(dir):
        error_message = "directory '" + str(dir) + "' not found."
        raise Exception(error_message)
    return dir

def main():
    # check commandline arguments
    print('>> Parsing the command line arguments...')
    args = parse_args()
    data_dir = checked_dir(args.data_dir)
    train_dir = checked_dir(os.path.join(args.data_dir, 'train'))
    valid_dir = checked_dir(os.path.join(args.data_dir, 'valid'))
    test_dir = checked_dir(os.path.join(args.data_dir, 'test'))
    save_dir = checked_dir(args.save_dir, make_dir=True)
    pretrained_arch = args.arch
    learning_rate = args.learning_rate
    num_hidden_units = args.hidden_units
    epochs = args.epochs
    current_device = get_device(args.gpu)

    print("\t" + "train_dir: " + train_dir)
    print("\t" + "valid_dir: " + valid_dir)
    print("\t" + "test_dir: " + test_dir)
    print("\t" + "save_dir: " + save_dir)
    print("\t" + "pretrained_arch: " + pretrained_arch)
    print("\t" + "learning_rate: " + str(learning_rate))
    print("\t" + "hidden_units: " + str(num_hidden_units))
    print("\t" + "epochs: " + str(epochs))
    print("\t" + "device: " + current_device)

    # load data
    print('Loading the data sets...')
    dataloaders, cat_to_name, class_to_idx = load_data(train_dir=train_dir, valid_dir=valid_dir, test_dir=test_dir)
    num_categories = len(cat_to_name)

    # build the model
    print('Building the model...')
    pretrained_model, in_features = get_pretrained_model(pretrained_arch)
    model, criterion, optimizer = build_model(model=pretrained_model, in_features=in_features, hidden_units=num_hidden_units,
                num_categories=num_categories, learning_rate=learning_rate)

    # train the model
    print('Training the model...')
    model = train(model=model, trainloader=dataloaders[TRAIN], testloader=dataloaders[VALIDATION], criterion=criterion, optimizer=optimizer, epochs=epochs, print_every=20, device=current_device)

    # test the model
    print('Testing the model...')
    test(model=model, testloader=dataloaders[TEST], criterion=criterion, device=current_device)

    # save the model's checkpoint
    print("Saving the model's checkpoint")
    model.class_to_idx = class_to_idx
    save_checkpoint(model=model, input_size=in_features, output_size=num_categories, pre_trained_model=pretrained_model, train_epochs=epochs, learning_rate=learning_rate, optimizer=optimizer, save_dir=save_dir)

    print("Finished")

if __name__ == '__main__':
    main()