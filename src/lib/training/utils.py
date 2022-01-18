import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import inception_v3, xception, vgg16, resnet50
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint#, EarlyStopping

def get_base_model(base_model_name):
    models = {
        "inception-v3": inception_v3.InceptionV3,
        "xception": xception.Xception,
        "vgg16": vgg16.VGG16,
        "resnet50": resnet50.ResNet50,
    }
    return models[base_model_name](weights="imagenet", include_top=False)


def get_preprocessing_fn(base_model_name):
    functions = {
        "inception-v3": inception_v3.preprocess_input,
        "xception": xception.preprocess_input,
        "vgg16": vgg16.preprocess_input,
        "resnet50": resnet50.preprocess_input,
    }
    return functions[base_model_name]


def get_input_size(base_model_name):
    functions = {
        "inception-v3": 299,
        "xception": 299,
        "vgg16": 224,
        "resnet50": 224,
    }
    return functions[base_model_name]


def create_model(base_model_name, n_outputs, index_layer_trainable=None):
    base_model = get_base_model(base_model_name)
    if n_outputs == 2:
        n_outputs = 1
    loss = "categorical_crossentropy" if n_outputs > 1 else "binary_crossentropy"
    activation = "softmax" if n_outputs > 1 else "sigmoid"

    output = base_model.output

    if base_model_name == "inception-v3":
        output = GlobalAveragePooling2D()(output)
        output = Dense(1024, activation="relu")(output)
    elif base_model_name == "xception":
        output = GlobalAveragePooling2D()(output)
        output = Dense(512, activation="relu")(output)
    elif base_model_name == "vgg16":
        output = GlobalAveragePooling2D()(output)
        output = Dense(4096, activation="relu")(output)
        output = Dense(4096, activation="relu")(output)
    elif base_model_name == "resnet50":
        output = GlobalAveragePooling2D()(output)
        output = Flatten()(output)
        output = Dense(256, activation="relu")(output)
        output = Dropout(0.7)(output)

    predictions = Dense(n_outputs, activation=activation)(output)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Si index_layer_trainable es None, entonces no hay fine tuning
    # y se entrena usando el optimizador Adam
    if index_layer_trainable is None:
        index_layer_trainable = len(base_model.layers)
        optimizer = "adam"
    # En caso contrario, se usa SGD con una tasa de aprendizaje muy baja
    else:
        optimizer = SGD(lr=0.0001, momentum=0.9)

    # Congelamos las primeras capas
    for layer in base_model.layers[:index_layer_trainable]:
        layer.trainable = False

    # El resto de las capas es entrenable
    for layer in base_model.layers[index_layer_trainable:]:
        layer.trainable = True

    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
    return model

def create_model_finetuning(base_model_name, n_outputs, checkpoint):
    base_model = get_base_model(base_model_name)
    if n_outputs == 2:
        n_outputs = 1
    loss = "categorical_crossentropy" if n_outputs > 1 else "binary_crossentropy"
    activation = "softmax" if n_outputs > 1 else "sigmoid"

    # base_model.trainable = False

    output = base_model.output

    if base_model_name == "inception-v3":
        output = GlobalAveragePooling2D()(output)
        output = Dense(1024, activation="relu")(output)
    elif base_model_name == "xception":
        output = GlobalAveragePooling2D()(output)
        output = Dense(512, activation="relu")(output)
    elif base_model_name == "vgg16":
        output = GlobalAveragePooling2D()(output)
        output = Dense(4096, activation="relu")(output)
        output = Dense(4096, activation="relu")(output)
    elif base_model_name == "resnet50":
        output = GlobalAveragePooling2D()(output)
        output = Flatten()(output)
        output = Dense(256, activation="relu")(output)
        output = Dropout(0.7)(output)

    predictions = Dense(n_outputs, activation=activation)(output)

    model = Model(inputs=base_model.input, outputs=predictions)

    # optimizer = "adam"
    optimizer = SGD(lr=0.0001, momentum=0.9)

    model.load_weights(checkpoint)
    print ("Checkpoint " + checkpoint + " cargado.")
    # Congelamos las primeras capas
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
    return model


def get_balanced_weights(y_classes):
    class_weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_classes), y=y_classes
    )
    print("Pesos balanceados", class_weights)
    return dict(enumerate(class_weights))


def check_data_augmentation(aug_args):
    data_aug = (
        aug_args["rotation_range"] != 0
        or aug_args["width_shift_range"] != 0.0
        or aug_args["height_shift_range"] != 0.0
        or aug_args["zoom_range"] != 0.0
        or aug_args["horizontal_flip"]
        or aug_args["vertical_flip"]
    )
    print("Data augmentation?:", data_aug)
    return data_aug


def get_generator(
    data,
    images_dir,
    x_col,
    y_col,
    batch_size,
    img_size,
    preprocess_fn,
    classes=None,
    aug_args=None,
    random_seed=None,
    shuffle=True,
):

    if aug_args is None:
        aug_args = defaultdict(int)
    n_classes = len(data[y_col].unique()) if classes is None else len(classes)
    class_mode = "binary" if n_classes <= 2 else "categorical"

    datagen = ImageDataGenerator(
        rotation_range=aug_args["rotation_range"],
        width_shift_range=aug_args["width_shift_range"],
        height_shift_range=aug_args["height_shift_range"],
        zoom_range=aug_args["zoom_range"],
        horizontal_flip=aug_args["horizontal_flip"],
        vertical_flip=aug_args["vertical_flip"],
        fill_mode="constant",
        cval=0,
        preprocessing_function=preprocess_fn,
    )

    generator = datagen.flow_from_dataframe(
        dataframe=data,
        x_col=x_col,
        y_col=y_col,
        directory=images_dir,
        classes=classes,
        class_mode=class_mode,
        shuffle=shuffle,
        batch_size=batch_size,
        target_size=(img_size, img_size),
        seed=random_seed,
    )

    print(generator.class_indices)
    return generator


def get_generator_prediction(
    data, images_dir, x_col, batch_size, img_size, preprocess_fn, classes
):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)
    generator = datagen.flow_from_dataframe(
        dataframe=data,
        x_col=x_col,
        classes=classes,
        directory=images_dir,
        class_mode=None,
        shuffle=False,
        batch_size=batch_size,
        target_size=(img_size, img_size),
    )
    return generator


def save_model(model, model_path):
    folder, _ = os.path.split(model_path)
    os.makedirs(folder, exist_ok=True)
    model.save(model_path)


def save_history(history, history_path):
    folder, _ = os.path.split(history_path)
    os.makedirs(folder, exist_ok=True)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_path)


def train_model(
    base_model_name,
    images_dir,
    train_df,
    val_df,
    classes,
    x_col,
    y_col,
    batch_size,
    max_epochs,
    index_layer_trainable,
    balanced_weights,
    aug_args,
    random_seed,
    checkpoint_filepath
):
    model = create_model(base_model_name, len(classes), index_layer_trainable)
    preprocess_fn = get_preprocessing_fn(base_model_name)
    img_size = get_input_size(base_model_name)
    train_generator = get_generator(
        data=train_df,
        images_dir=images_dir,
        x_col=x_col,
        y_col=y_col,
        batch_size=batch_size,
        img_size=img_size,
        preprocess_fn=preprocess_fn,
        classes=classes,
        aug_args=aug_args,
        random_seed=random_seed,
    )

    val_generator = get_generator(
        data=val_df,
        images_dir=images_dir,
        x_col=x_col,
        y_col=y_col,
        batch_size=batch_size,
        img_size=img_size,
        preprocess_fn=preprocess_fn,
        classes=classes,
        random_seed=random_seed,
    )

    weights = (
        get_balanced_weights(train_generator.classes) if balanced_weights else None
    )

    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                       save_weights_only=False,
                                       monitor="val_loss",
                                       mode="min",
                                       save_best_only=True)

    callbacks_list = [model_checkpoint]

    history = model.fit_generator(
        train_generator,
        class_weight=weights,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=max_epochs,
        workers=4,
        validation_data=val_generator,
        validation_steps=val_generator.n // batch_size,
        callbacks=callbacks_list
    )

    return model, history

def finetune_model(
    base_model_name,
    images_dir,
    train_df,
    val_df,
    classes,
    x_col,
    y_col,
    batch_size,
    max_epochs,
    balanced_weights,
    aug_args,
    random_seed,
    checkpoint_filepath,
    previous_checkpoint
):
    model = create_model_finetuning(base_model_name, len(classes), previous_checkpoint)
    preprocess_fn = get_preprocessing_fn(base_model_name)
    img_size = get_input_size(base_model_name)
    train_generator = get_generator(
        data=train_df,
        images_dir=images_dir,
        x_col=x_col,
        y_col=y_col,
        batch_size=batch_size,
        img_size=img_size,
        preprocess_fn=preprocess_fn,
        classes=classes,
        aug_args=aug_args,
        random_seed=random_seed,
    )

    val_generator = get_generator(
        data=val_df,
        images_dir=images_dir,
        x_col=x_col,
        y_col=y_col,
        batch_size=batch_size,
        img_size=img_size,
        preprocess_fn=preprocess_fn,
        classes=classes,
        random_seed=random_seed,
    )

    weights = (
        get_balanced_weights(train_generator.classes) if balanced_weights else None
    )

    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                       save_weights_only=False,
                                       monitor="val_loss",
                                       mode="min",
                                       save_best_only=True)

    callbacks_list = [model_checkpoint]

    history = model.fit_generator(
        train_generator,
        class_weight=weights,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=max_epochs,
        workers=4,
        validation_data=val_generator,
        validation_steps=val_generator.n // batch_size,
        callbacks=callbacks_list
    )

    return model, history
