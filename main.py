import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split, Subset
from torchviz import make_dot
from torchsummary import summary
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import random
from torchvision import transforms
from torchmetrics.classification import MulticlassF1Score

BATCH_SIZE = 256
CLASSES_COUNT = 10
LEARNING_RATE = 0.001
EPOCHS_COUNT = 20 # changed to 25 to oveserve overfit in section 2 (I think that accroding to forum conditions need to be the same)
TRAIN_SPLIT_RATIO = 0.001 # 0.8 - regular split. ~0.0005 for receiving an overfit

DEBUG = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used for learning: {DEVICE}")

class MyNet(nn.Module):
    def __init__(self, classes_count):
        super(MyNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)  # in: 28X28, out: 28X28X16
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                                padding=1)  # in: 28X28X16, out: 28X28X32
        self.mp_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # in: 28X28X32, out: 14X14X32
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                                padding=1)  # in: 14X14X32, out: 14X14X64
        self.linear_1 = nn.Linear(3136, 300)  # in: 7X7X32, out: 14X14X64
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(300, classes_count)
        self.sm_1 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_1(self.conv_2(out))
        out = self.mp_1(out)
        out = self.relu_1(self.conv_3(out))
        out = self.mp_1(out)
        out = out.reshape(out.size(0), -1)
        out = self.relu_1(self.linear_1(out))
        out = self.linear_2(out)
        return out

class MyNet2(nn.Module):
    def __init__(self,classes_count):
        super(MyNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, classes_count)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x) # softmax is included in the loss function
        return x

def visualize_model_as_graph(model:MyNet):
    dummy_input = torch.randn(1, 1, 28, 28) #single MNIST image size
    dummy_result = model(dummy_input)
    dot = make_dot(dummy_result, params=dict(model.named_parameters()))
    print(type(dot))
    print(dot)
    dot.render("model_graph", format='png')

def load_train_set(origin_train_data, split_ratio, batch_size,force_balanced_categories=True):
    train_size = int(len(origin_train_data) * split_ratio)
    validation_size = len(origin_train_data) - train_size
    if force_balanced_categories:
        targets = origin_train_data.targets
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=validation_size / (train_size + validation_size))
        for train_idx, val_idx in splitter.split(X=range(len(targets)), y=targets):
            train_set = Subset(origin_train_data, train_idx)
            validation_set = Subset(origin_train_data, val_idx)
    else:
        train_set, validation_set = random_split(origin_train_data, [train_size, validation_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=False)

    if DEBUG:
        # Verify categories are balanced:
        sets_dict = {"Train": train_set, "Validation": validation_set}
        visualize_data_splits(sets_dict)
    return train_loader, validation_loader

def add_augmentation_using_torchvision(train_set, method, percentage):
    # Get the original transform if it exists
    original_transform = train_set.transform if hasattr(train_set, 'transform') else None

    # Define some augmentation options
    augmentation_methods = {
        'flip': transforms.RandomHorizontalFlip(p=1.0),
        'rotate': transforms.RandomRotation(degrees=15),
        'color_jitter': transforms.ColorJitter(brightness=0.5, contrast=0.5),
        'crop': transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
    }

    if method not in augmentation_methods:
        raise ValueError(f"Unsupported augmentation method: {method}")

    # Decide whether to apply augmentation to each image
    def apply_with_probability(img):
        if random.random() < percentage:
            return augmentation_methods[method](img)
        return img

    # Compose the new transform
    train_set.transform = transforms.Compose([
        transforms.Lambda(apply_with_probability),
        original_transform if original_transform else transforms.ToTensor()
    ])

    return train_set

def plot_categories_histograms(set,header):
    # Extract class labels from the original dataset (Subset stores indices)
    # Supports subset or full dataset via try except mechanism on input
    try:
        labels = [set.dataset.targets[i] for i in set.indices]
    except:
        labels = [set.targets[i] for i in range(len(set.targets))]
    # Creating histogram dicts:
    counting_dict={}
    for k in range(CLASSES_COUNT):
        counting_dict[k]=0
    # Count occurrences of each class
    for i,value in enumerate(np.array(labels)):
        counting_dict[value] +=1

    # Extract class indices and corresponding counts
    classes = list(counting_dict.keys())
    counts = [counting_dict[k] for k in classes]

    total = sum(counts)
    percentages = [count / total * 100 for count in counts]

    # Plotting training set histogram
    plt.figure(figsize=(8, 4))
    bars = plt.bar(classes, counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'{header} Set Class Distribution')
    plt.xticks(classes)
    plt.tight_layout()

    # Add percentage labels on top of bars
    for bar, percent in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height , f'{percent:.1f}%',
                 ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_train_progress(train_losses, validation_losses, secondary_axes=True):
    assert(len(train_losses) == len(validation_losses))
    x1 = np.arange(len(train_losses))
    validation_x_values = x1[~np.isnan(validation_losses)]
    validation_y_values = np.array(validation_losses)[~np.isnan(validation_losses)]


    filtered = gaussian_filter1d(train_losses, sigma=2)

    fix, ax = plt.subplots()
    ax.plot(x1, filtered, label='Train Loss', linestyle="solid", color="blue")
    ax.plot(validation_x_values, validation_y_values, linestyle='solid', color="red", zorder=4, label='Validation Loss')
    ax.set_xlabel("Mini Batch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Train vs. Validation Loss Progress")

    if secondary_axes:
        epoch_numbers = np.arange(0, len(validation_x_values) + 1)
        validation_x_values_with_0 = np.insert(validation_x_values, 0, 0)
        ax2 = ax.twiny() # twin x-axis
        ax2.set_xlim(ax.get_xlim())  # align the secondary x-axis with the primary
        ax2.set_xticks(validation_x_values_with_0)
        ax2.set_xticklabels(epoch_numbers)
        ax2.set_xlabel("Epoch")

    plt.savefig(f"image_progress_{datetime.datetime.now()}.png".replace('-', '_').
                replace(':', '_').replace(' ', "_"))

    plt.show()


def plot_conf_matrix(all_labels, all_predicted):
    sklrn_conf_mat = confusion_matrix(np.array(all_labels), np.array(all_predicted))
    normalized_conf_mat = sklrn_conf_mat.astype('float') / sklrn_conf_mat.sum(axis=1)[:, np.newaxis]

    sns.heatmap(normalized_conf_mat, annot=True, fmt=".2f", cmap="Greens", xticklabels=range(CLASSES_COUNT),
                yticklabels=range(CLASSES_COUNT))
    plt.xlabel("Predictaded")
    plt.ylabel("True Labels")
    plt.title("Test Results: Confusion Matrix")
    plt.savefig(
        f"confusion_matrix_{datetime.datetime.now()}.png".replace('-', '_').
        replace(':', '_').replace(' ', "_"))
    plt.show()


def plot_f1_scores(all_predicted, all_labels):
    f1 = MulticlassF1Score(num_classes=CLASSES_COUNT, average=None)
    f1_score = f1(torch.tensor(all_predicted), torch.tensor(all_labels))
    class_names = [str(i) for i in range(10)]
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, f1_score, alpha=0.7, edgecolor='black')
    plt.xlabel("Classes")
    plt.ylabel("F1 Score")
    plt.title(f"F1 Score per Class. Macro Value:{np.average(f1_score):.4f}")
    for i, v in enumerate(f1_score):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
    print(f1_score)

def load_test_set(test_data, batch_size):
    return DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

def train(model:MyNet, train_data, split_ratio, batch_size, classes_count, learning_rate):
    print("loading the set into train and validation")
    train_loader, validation_loader = load_train_set(train_data, split_ratio, batch_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps_count = len(train_loader)
    train_losses = []
    validation_losses = []
    for epoch in range(EPOCHS_COUNT):
        #train the epoch
        print(f"starting epoch {epoch+1}")
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            validation_losses.append(np.nan)
            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS_COUNT}, step {i+1}/{steps_count}, current loss:{loss.item()}")
        print(f"Epoch {epoch+1} / {EPOCHS_COUNT} is done. Loss:{loss.item()}")
        #validate the epoch
        with torch.no_grad():
            correct_count = 0
            total_images = 0
            acc_loss = 0
            acc_items = 0
            for i, (images, labels) in enumerate(validation_loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                val, predicted = torch.max(outputs.data, dim=1)
                total_images += len(images)
                correct = (predicted == labels).sum().item()
                correct_count += correct
                loss = loss_function(outputs, labels)
                batch_size = images.size(0)
                acc_loss += loss.item() * batch_size
                acc_items += batch_size
                del images, labels, outputs
            validation_losses[-1] = (acc_loss / acc_items)
            print(f"validation accuracy ratio: {correct_count / total_images}")
            if correct_count / total_images > 0.99:
                print("Stopping the training")
                break
    plot_train_progress(train_losses, validation_losses)


def test(model: MyNet, test_data: object, batch_size: object):
    with torch.no_grad():
        correct_cout = 0
        total_count = 0
        all_predicted = []
        all_labels = []
        test_set = load_test_set(test_data, batch_size=batch_size)
        for images, labels in test_set:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            vals, predicted = torch.max(outputs.data, 1)
            all_predicted += predicted.tolist()
            all_labels += labels.tolist()
            total_count += len(labels)
            correct_cout += (predicted == labels).sum().item()
            del images, labels, outputs
        print(f"Test results: accuracy of {correct_cout / total_count}")
        plot_conf_matrix(all_labels, all_predicted)
        plot_f1_scores(all_predicted, all_labels)

def load_mnist():
    train_data = datasets.MNIST(root='./mnist/', download=True, train=True, transform=ToTensor())
    test_data = datasets.MNIST(root='./mnist/', download=True, train=False, transform=ToTensor())
    return train_data, test_data

def inspect_minist_data(train_data, test_data):
    print(type(train_data))
    print(type(test_data))

    image, label = train_data[0]
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(test_data.classes[label])
    plt.show()

def visualize_data_splits(sets_dict):
    for key, dataset in sets_dict.items():
        plot_categories_histograms(dataset, key)
    return

if __name__ == '__main__':
    print("loading data set")
    train_data, test_data = load_mnist()
    print("creating the model")
    model = MyNet(CLASSES_COUNT).to(DEVICE)
    summary(model, (1, 28, 28))
    #visualize_model_as_graph(model)
    #inspect_minist_data(train_data, test_data)

    # Only for debug Purposes for seif 2
    if DEBUG:
        visualize_data_splits({"Test":test_data})

    print("training...")
    train(model, train_data, TRAIN_SPLIT_RATIO, BATCH_SIZE, CLASSES_COUNT, LEARNING_RATE)
    print("testing...")
    test(model,test_data, BATCH_SIZE)
    print("done")

