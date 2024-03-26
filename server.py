from typing import Dict, Optional, Tuple
import flwr as fl

import tensorflow as tf
import os
import csv
import cv2
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing import image

import edgeimpulse as ei

#server_address = "192.168.1.102:5052"
server_address = "192.168.85.226:5052"

# this variable determines if model profiling and deployment with Edge Impulse will be done
profile_and_deploy_model_with_EI = False

# change this to your Edge Impulse API key
ei.API_KEY = "ei_af1a6b94544b625aa65cd03079585ba019e999af510c8533657a3f9d23c8c905"

# defining a .eim file that will be automatically downloaded on the server computer
deploy_eim = "modelfile.eim"

# list with the classes for the image classification
classes = ["without-mask", "mask"]
class_labels = {classes: i for i, classes in enumerate(classes)}
number_of_classes = len(classes)

# defining image size, 
IMAGE_SIZE = (160, 160)
CLIENT_NUMBER = 2

header_effectivness = ['clients number', 'round', 'training_loss', 'training_accuracy']
with open('epoch_25.csv', 'w') as fileEffec:
    writer = csv.writer(fileEffec)
    writer.writerow(header_effectivness)

federatedLearningcounts = 10
local_client_epochs = 20
local_client_batch_size = 8

start_training = datetime.now()

def main() -> None:
    # load and compile model for : server-side parameter initialization, server-side parameter evaluation
    """
    # EfficientNetB0 model
    model = tf.keras.applications.EfficientNetB0(
        input_shape=(160, 160, 3), weights=None, classes=2
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling=None,
        classes=2,
        classifier_activation="softmax"
    )
    # freeze the layers in the base model so they don't get updated
    base_model.trainable = False

    # define classification without-mask
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    # create the final model
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    # compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=CLIENT_NUMBER,
        min_evaluate_clients=CLIENT_NUMBER,
        min_available_clients=CLIENT_NUMBER,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(
            model.get_weights()),
    )

    # start Flower server (SSL-enabled) for X rounds of federated learning
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=federatedLearningcounts),
        strategy=strategy
    )

def load_dataset():
    # defining the directory with the server's test images. We only use the test images!
    directory = "datasets/dataset_server"
    sub_directories = ["test", "train"]

    loaded_dataset = []
    for sub_directory in sub_directories:
        path = os.path.join(directory, sub_directory)
        images = []
        labels = []

        print("Server dataset loading {}".format(sub_directory))

        for folder in os.listdir(path):
            label = class_labels[folder]

            # iterate through each image in the folder
            for file in os.listdir(os.path.join(path,folder)):
                # get path name of the image
                img_path = os.path.join(os.path.join(path, folder), file)

                # open and resize the image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE)

                # append the image and its corresponding label to loaded_dataset
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype= 'float32')
        labels = np.array(labels, dtype= 'int32')

        loaded_dataset.append((images, labels))
    
    return loaded_dataset

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # load data and model here to avoid the overhead of doing it in `evaluate` itself
    (training_images, training_labels), (test_images, test_labels) = load_dataset()
    print("[Server] test_images shape:", test_images.shape)
    print("[Server] test_labels shape:", test_labels.shape)

    # the `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        print("======= server round %s/%s evaluate() ===== " %(server_round, federatedLearningcounts))
        # update model with the latest parameters
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        print("======= server round %s/%s accuracy : %s =======" %(server_round, federatedLearningcounts,accuracy))

        with open('epoch_25.csv', 'a') as fileEffec:
            writer = csv.writer(fileEffec)
            writer.writerow([CLIENT_NUMBER, server_round, loss, accuracy])

        if (server_round == federatedLearningcounts):
            end_training = datetime.now()
            training_time_minutes = end_training.minute - start_training.minute
            training_time_second = end_training.second - start_training.second

            if(training_time_second == 0):
                print(f"Training of the model for {CLIENT_NUMBER} took {training_time_minutes} minutes.")
            elif(training_time_second > 0):
                print(f"Training of the model for {CLIENT_NUMBER} took {training_time_minutes} minutes and {abs(training_time_second)} second.") 
            elif(training_time_minutes < 0):
                print(f"Training of the model for {CLIENT_NUMBER} took {(training_time_minutes-1)} minutes and {(training_time_second+60)} second.") 

            # save the decentralized ML model locally on the server computer
            print("Saving updated model locally..")
            model.save('saved_models/mobilenetv2')      # save model in SavedModel format

            # test the updated model
            test_updated_model(model)

            # ***** Using Edge Impulse Python SDK to profile the updated model and deploy it for various MCUs/MPUs *****
            if (profile_and_deploy_model_with_EI):
                ei_profile_and_deploy_model(model)
            else:
                print("Skipping profiling and deploying with Edge Impulse BYOM!")
             
        return loss, {"accuracy": accuracy}
    return evaluate

def fit_config(server_round: int):
    # return training configuration dict for each round
    config = {
        "batch_size": local_client_batch_size,
        "local_epochs": local_client_epochs,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round."""
    val_steps = 4
    return {"val_steps": val_steps}

def test_updated_model(model):
    # test the model by giving it an image and get its prediction
    test_image_withoutmask_path = "datasets/dataset_test/without-mask_5.jpg"
    test_image_withoutmask = cv2.imread(test_image_withoutmask_path)
    test_image_withoutmask = cv2.cvtColor(test_image_withoutmask, cv2.COLOR_BGR2RGB)
    test_image_withoutmask = cv2.resize(test_image_withoutmask, IMAGE_SIZE)

    test_image_mask_path = "datasets/dataset_test/mask_4.jpg"
    test_image_mask = cv2.imread(test_image_mask_path)
    test_image_mask = cv2.cvtColor(test_image_mask, cv2.COLOR_BGR2RGB)
    test_image_mask = cv2.resize(test_image_mask, IMAGE_SIZE)

    print("Testing the final model on an image.....")
    # chose either test_image_withoutmask or test_image_mask for the prediction
    image_test_result = model.predict(np.expand_dims(test_image_mask, axis=0))
    # print the prediction scores/confidence for each class
    # index 0 = without-mask, index 1 = mask
    print(image_test_result[0])

    # an easy trick to see the model's prediction scores ("confidence") for each class
    # we can get the highest score/confidence among all classes
    # map the highest score's index to its class
    highest_prediction_score = max(image_test_result[0])
    highest_prediction_score_index = 0
    for i in range(len(image_test_result[0])):
        if image_test_result[0][i] == highest_prediction_score:
            highest_prediction_score_index = i

    most_confident_class = classes[highest_prediction_score_index]
    print("The model mostly predicted %s with a score/confidence of %s" %(most_confident_class, highest_prediction_score))

def ei_profile_and_deploy_model(model):
    # list the available profile target devices
    ei.model.list_profile_devices()

    # estimate the RAM, ROM, and inference time for our model on the target hardware family
    try:
        profile = ei.model.profile(model=model, device='raspberry-pi-4')
        print(profile.summary())
    except Exception as e:
        print(f"Could not profile: {e}")

    # list the available profile target devices
    ei.model.list_deployment_targets()

    # set model information, such as your list of labels
    model_output_type = ei.model.output_type.Classification(labels=classes)

    # create eim executable with trained model
    deploy_bytes = None
    try:
        deploy_bytes = ei.model.deploy(model=model,
                                    model_output_type=model_output_type,
                                    deploy_target='runner-linux-aarch64',
                                    output_directory='ei_deployed_model')
    except Exception as e:
        print(f"Could not deploy: {e}")

    # write the downloaded raw bytes to a file
    if deploy_bytes:
        with open(deploy_eim, 'wb') as f:
            f.write(deploy_bytes)

if __name__ == "__main__":
    main()
