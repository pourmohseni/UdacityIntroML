import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import json


def parse_args():
    """
    method to parse the command line arguments
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("checkpoint", type=str, default=".", help="Trained model's checkpoint")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Number of most-likely categories to be outputed (integer)")
    parser.add_argument("--category_names", type=str, help="Json file containing category names")
    parser.add_argument("--gpu", action='store_true', help="Use GPU (set true if GPU is available)")

    args = parser.parse_args()

    return args


def load_checkpoint(filepath):
    """
    loads a mode from the given path
    :param filepath: 
    :return: 
    """
    checkpoint = torch.load(filepath)
    model = checkpoint['pretrained_model']

    # Freeze model parameters (prevent backpropagation through them)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.epochs = checkpoint['train_epochs']
    model.learning_rate = checkpoint['learning_rate']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']

    return model


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    :param image: 
    :return: 
    """
    im = Image.open(image)

    # necessary transformations to the image
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

    return transform(im)


def predict(image_path, model, topk=5, device='cpu'):
    """
    makes predictions
    :param image_path: path to image
    :param model: trained model
    :param topk: number of top categories to be outputed
    :param device: device type
    :return: list of top probabilities,list of top categories
    """

    model.to(device)

    # disable dropout by switching to evaluation (inference) mode
    model.eval()

    # get the image and move it to the current device
    image = process_image(image_path)
    image = image.to(device)
    image = image.unsqueeze(0)  # required for vgg

    # make prediction with gradients turned off
    with torch.no_grad():
        # forward pass
        output = model(image)

        # get the output probabilities
        probabilities = torch.exp(output)

        # Find the predicted class (the class with highest probability)
        top_p, top_classes = probabilities.topk(topk, dim=1)

        # inverting the class_to_index dictionary
        index_to_class = {value: key for key, value in model.class_to_idx.items()}

        # get list of classes
        top_classes = [np.int(index_to_class[c]) for c in np.array(top_classes.detach())[0]]

        # get the list of probabilities
        top_p = np.array(top_p.detach())[0]

    return top_p, top_classes


def get_device(gpu=False):
    """
    sets the correct device to be used. GPU is used if the gpu flag is set and cuda is available, otherwise, 
    CPU is used. :param gpu: :return: current device's name 
    """
    
    if not gpu:
        current_device = 'cpu'
    else:
        current_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if current_device == 'cpu':
            print("GPU is not available...")
    return current_device


def check_file(file):
    if not os.path.isfile(file):
        error_message = "file '" + str(file) + "' not found."
        raise Exception(error_message)
    return file


def main():
    # check commandline arguments
    print('Parsing the command line arguments...')
    args = parse_args()
    image_path = check_file(args.image_path)
    checkpoint = check_file(args.checkpoint)
    top_k = args.top_k
    category_names_file = None
    if args.category_names is not None:
        category_names_file = check_file(args.category_names)
    current_device = get_device(args.gpu)
    
    print('Loading the model...')
    # load the model
    model = load_checkpoint(checkpoint)
        
    print('Processing the image...')
    # load and process the image
    image = process_image(image_path)

    print('Making predictions...')
    # make a prediction 
    top, classes = predict(image_path=image_path, model=model, topk=top_k, device=current_device)

    # get image categories if the categories file is provided
    if category_names_file is None:
        image_cats = classes
    else:
        with open(category_names_file, 'r') as f:
            cat_to_name = json.load(f)
        image_cats = [cat_to_name[str(img_class)] for img_class in classes]

    print("top categories: ", image_cats)
    print("top probabilities: ", top)


if __name__ == '__main__':
    main()
