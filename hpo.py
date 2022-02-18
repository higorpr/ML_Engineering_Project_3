#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import torch
import argparse
import os

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch import nn, optim
from PIL import ImageFile

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
    '''

    print('Testing Model on Whole Testing Dataset')

    model.eval()  # Setting model to evaluation mode
    running_loss = 0  # Sum of losses for average loss calculation
    running_corrects = 0  # Sum of correct predictions for accuracy calculation

    for input_tensor, labels in test_loader:  # Iterating over test loader
        input_tensor = input_tensor.to(device)
        labels = labels.to(device)

        # Getting predictions for each tensor in test loader:
        output = model(input_tensor)
        loss = criterion(output, labels)

        percentage, preds = torch.max(output, 1)
        running_loss += loss.item() * input_tensor.size(0)
        running_corrects += torch.sum(preds == labels)

    # End of iteration over test loader data

    # Calculation of average loss:
    avg_loss = running_loss / len(test_loader.dataset)
    # Calculation of accuracy:
    acc = running_corrects / len(test_loader.dataset)

    print ('Printing Log')
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            avg_loss, running_corrects, len(test_loader.dataset), 100.0 * acc
        )
    )

def train(model, epochs, train_loader, validation_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
    '''

    loader = {'train': train_loader, 'eval': validation_loader}
    epochs = epochs
    best_loss = 1e6
    loss_counter = 0

    for epoch in range(epochs):  # Iteration over all epochs
        for phase in ['train', 'eval']:  # Iteration for training and evaluation under epoch
            print(f'Epoch: {epoch}, Phase: {phase}')
            # Sets model to train or evaluation modes
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Initiates cumulative variables IN EACH EPOCH:
            accum_loss = 0  # Sum of all the loss values (for each prediction)
            accum_corrects = 0  # Number of correct predictions made by the model
            samples_ran = 0  # Number of samples ran by the model

            for input_tensor, labels in loader[phase]:  # Iteration over data in train loader

                # Sending tensors to CPU or GPU
                input_tensor = input_tensor.to(device)
                labels = labels.to(device)

                # Getting output tensor
                output = model(input_tensor)
                # Getting loss value
                loss = criterion(output, labels)

                # Part of epoch that differentiates over train and eval phases:
                if phase == 'train':  # Steps that are exclusive to training phase
                    optimizer.zero_grad()  # Resets gradients in optimizer
                    loss.backward()  # Executes backward pass
                    optimizer.step()  # Updates parameters of Neural Network

                percentages, preds = torch.max(output, dim=1)  # Get class predictions
                accum_loss += loss.item() * input_tensor.size(0)
                accum_corrects += torch.sum(preds == labels).item()
                samples_ran += len(input_tensor)

                # Printing log after every 2000 samples ran:
                if samples_ran % 2000 == 0:
                    accuracy = accum_corrects / samples_ran
                    print(f'Log Entry: Epoch {epoch}, Phase {phase}.')
                    print(
                        'Images [{}/{} ({:.0f}%)] / Loss: {:.2f} / Accumulated Loss: {:.0f} / '
                        'Accuracy: {}/{} ({:.2f}%)'.format(
                            samples_ran,
                            len(loader[phase].dataset),
                            100 * (samples_ran / len(loader[phase].dataset)),
                            loss.item(),
                            accum_loss,
                            accum_corrects,
                            samples_ran,
                            100 * accuracy
                        ))

                # Section that trains and evaluates on a portion of the dataset
                # Comment if you want to train on the whole dataset:
                if samples_ran >= (0.25 * len(loader[phase].dataset)):
                    break

            if phase == 'eval':
                avg_epoch_loss = accum_loss / samples_ran
                if avg_epoch_loss < best_loss:  # If the avg_loss increased in the epoch, signalize it.
                    best_loss = avg_epoch_loss  # Update of maximum loss so far
                else:
                    loss_counter += 1  # Counter to stop epochs

        if loss_counter > 0:  # If the avg_loss in evaluation phase increased, the model started to diverge.
            break
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(),
        nn.Linear(128, 133)
    )
    
    return model

def create_data_loaders(train_dir, test_dir, eval_dir, train_batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''

    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_data = ImageFolder(root=train_dir, transform=training_transform)
    test_data = ImageFolder(root=test_dir, transform=testing_transform)
    validation_data = ImageFolder(root=eval_dir, transform=testing_transform)

    train_loader = DataLoader(train_data, train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, test_batch_size, shuffle=False)
    validation_loader = DataLoader(validation_data, test_batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader


def main(args):

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    '''
    TODO: Initialize a model by calling the net function
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    model = net()
    model = model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    train_loader, validation_loader, test_loader = create_data_loaders(args.train_dir, args.test_dir, args.eval_dir,
                                                                       args.batch_size, args.test_batch_size)

    train(model, args.epochs, train_loader, validation_loader, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    path = './model_hpo_optimization'
    torch.save(model, path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        metavar='N',
        help='batch size for training (default: 10)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        metavar='N',
        help='learning rate (default: 0.1)'
    )

    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=10,
        metavar='N',
        help='batch size for training (default: 10)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        metavar='N',
        help='number of epochs for training (default: 5)'
    )

    # Container environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--eval-dir", type=str, default=os.environ["SM_CHANNEL_VAL"])

    args = parser.parse_args()

    main(args)
