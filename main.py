import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from torchviz import make_dot
from torchsummary import summary
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
import datetime
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter1d

BATCH_SIZE = 256
CLASSES_COUNT = 10
LEARNING_RATE = 0.0005
EPOCHS_COUNT = 10
TRAIN_SPLIT_RATIO = 0.8
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
        self.mp_2 = nn.MaxPool2d(kernel_size=2, stride=2)  # in: 14X14X64, out: 7X7X64
        self.linear_1 = nn.Linear(3136, 300)  # in: 7X7X32, out: 14X14X64
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(300, classes_count)
        self.sm_1 = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.mp_1(out)
        out = self.conv_3(out)
        out = self.mp_2(out)
        out = out.reshape(out.size(0), -1)
        out = self.linear_1(out)
        out = self.relu_1(out)
        out = self.linear_2(out)
        out = self.sm_1(out)
        return out


def visualize_model_as_graph(model:MyNet):
    dummy_input = torch.randn(1, 1, 28, 28) #single MNIST image size
    dummy_result = model(dummy_input)
    dot = make_dot(dummy_result, params=dict(model.named_parameters()))
    print(type(dot))
    print(dot)
    dot.render("model_graph", format='png')


def load_train_set(origin_train_data, split_ratio, batch_size):
    train_size = int(len(origin_train_data) * split_ratio)
    validation_size = len(origin_train_data) - train_size
    train_set, validation_set = random_split(origin_train_data, [train_size, validation_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader


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
    assert(len(train_losses) == len(validation_losses))
    x1 = np.arange(len(train_losses))
    validation_x_values = x1[~np.isnan(validation_losses)]
    validation_y_values = np.array(validation_losses)[~np.isnan(validation_losses)]
    filtered = gaussian_filter1d(train_losses, sigma=2)
    plt.plot(x1, filtered, label='Train Loss', linestyle="solid", color="blue")
    #plt.scatter(x1, validation_losses, color="red", marker="s", s=80, zorder=3)
    plt.plot(validation_x_values, validation_y_values, linestyle='solid', color="red", zorder=4, label='Validation Loss')
    plt.xlabel("mini batches")
    plt.ylabel("loss")
    plt.legend()
    #ax2 = plt.twiny()
    #ax2.set_xticks(validation_x_values)
    plt.title("Train vs. Validation Loss Progress")
    plt.savefig(f"image_progress_{datetime.datetime.now()}.png".replace('-', '_').replace(':', '_').replace(' ', "_"))
    plt.show()


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

        sklrn_conf_mat = confusion_matrix(np.array(all_labels), np.array(all_predicted))
        normalized_conf_mat = sklrn_conf_mat.astype('float') / sklrn_conf_mat.sum(axis=1)[:, np.newaxis]

        sns.heatmap(normalized_conf_mat, annot=True, fmt=".2f", cmap="Greens", xticklabels=range(CLASSES_COUNT), yticklabels=range(CLASSES_COUNT))
        plt.xlabel("Predictaded")
        plt.ylabel("True Labels")
        plt.title("Test Results: Confusion Matrix")
        plt.savefig(
            f"confusion_matrix_{datetime.datetime.now()}.png".replace('-', '_').replace(':', '_').replace(' ', "_"))
        plt.show()
        """
        TP = 0
        TN = 1
        FP = 2
        FN = 3
        results_hist_by_class = np.zeros((CLASSES_COUNT, 4))
        for class_idx in range(CLASSES_COUNT):
            for i, label  in enumerate(all_labels):
                predicted = all_predicted[i]
                if label == class_idx:
                    if label == predicted:
                        results_hist_by_class[class_idx, TP] += 1
                    else:
                        results_hist_by_class[class_idx, TP] += 1
        """



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


if __name__ == '__main__':
    print("loading data set")
    train_data, test_data = load_mnist()
    print("creating the model")
    model = MyNet(CLASSES_COUNT).to(DEVICE)
    # summary(model, (1, 28, 28))
    #visualize_model_as_graph(model)
    #inspect_minist_data(train_data, test_data)
    print("training...")
    train(model, train_data, TRAIN_SPLIT_RATIO, BATCH_SIZE, CLASSES_COUNT, LEARNING_RATE)
    print("testing...")
    test(model,test_data, BATCH_SIZE)
    print("done")

