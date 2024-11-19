from code_for_google_colab.main_imgnet import train_imagenet_with_RBF
from consts import ModelType
from networks.resnet34_linear import train_resnet_model_with_linear, check_resnet_model_with_linear
from networks.resnet34_rbf import train_resnet_model_with_rbf, check_resnet_model_with_rbf
from networks.test import train_test, check_test


def train_chosen_model(model_type, epochs, learning_rate):
    if model_type.name == 'RESNET_RBF':
        train_resnet_model_with_rbf(learning_rate, epochs)
    elif model_type.name == 'RESNET_LINEAR':
        train_resnet_model_with_linear(learning_rate, epochs)
    elif model_type.name == 'IMAGENET':
        train_imagenet_with_RBF(learning_rate, epochs)
    elif model_type.name == 'IMAGENET_RBF':
        train_test(learning_rate, epochs)


def check_chosen_model(model_type, epochs):
    if model_type.name == 'RESNET_RBF':
        check_resnet_model_with_rbf(epochs)
    elif model_type.name == 'RESNET_LINEAR':
        check_resnet_model_with_linear(epochs)
    elif model_type.name == 'IMAGENET':
        pass
        # check_test(epochs)
    elif model_type.name == 'IMAGENET_RBF':
        pass
        # check_test(epochs)


if __name__ == "__main__":
    # users_input
    model_type = ModelType.IMAGENET_RBF
    epochs = 20
    learning_rate = 10e-4

    train_chosen_model(model_type, epochs, learning_rate)
    check_chosen_model(model_type, epochs)
