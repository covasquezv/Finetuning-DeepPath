import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from ...lib.training import utils as trainutils

def main(
    patches_df_path,
    images_dir,
    min_threshold,
    base_model_name,
    x_col,
    y_col,
    epochs,
    batch_size,
    history_save_dir,
    model_save_dir,
    random_seed,
    rotation_range,
    width_shift_range,
    height_shift_range,
    zoom_range,
    horizontal_flip,
    vertical_flip,
    balanced_weights,
    previous_checkpoint,
    df_patches_val=""):

    aug_args = {
        "rotation_range": rotation_range,
        "width_shift_range": width_shift_range,
        "height_shift_range": height_shift_range,
        "zoom_range": zoom_range,
        "horizontal_flip": horizontal_flip,
        "vertical_flip": vertical_flip,
    }

    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    model_path = os.path.join(
        model_save_dir,
        f"{base_model_name}_finetuned.h5",
    )
    folder, _ = os.path.split(model_path)
    os.makedirs(folder, exist_ok=True)

    patches_df = pd.read_csv(patches_df_path)
    patches_df = patches_df[~patches_df[y_col].isna()]
    patches_useful = patches_df[patches_df["Tissue"] > min_threshold]
    patches_useful = patches_df[patches_df["Artifact"] < 0.2]

    classes = sorted(patches_df[y_col].unique())
    print("Training with classes:", ", ".join(classes))

    train_df = patches_useful
    test_df = pd.read_csv(df_patches_val)

    model, history = trainutils.finetune_model(
        base_model_name,
        images_dir,
        train_df,
        test_df,
        classes,
        x_col,
        y_col,
        batch_size,
        epochs,
        balanced_weights,
        aug_args,
        random_seed,
        model_path,
        previous_checkpoint
    )

    history_path = os.path.join(
        history_save_dir,
        f"{base_model_name}_finetuned.csv",
    )

    model_path = os.path.join(
        model_save_dir,
        f"{base_model_name}_finetuned_end.h5",
    )

    trainutils.save_history(history, history_path)
    trainutils.save_model(model, model_path)
    K.clear_session()

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--patches_df_path",
        type=str,
        help="""\
        Ruta al archivo csv con los datos de los parches, indicando la cantidad
        de tejido, fondo y artefacto calculada para cada imagen.""",
        required=True,
    )
    PARSER.add_argument(
        "--images_dir",
        type=str,
        help="""\
        Directorio base donde se encuentran las imágenes.\
        """,
        required=True,
    )
    PARSER.add_argument(
        "--min_threshold",
        type=float,
        help="""\
        Umbral mínimo para el filtro de calidad.\
        """,
        required=True,
    )
    PARSER.add_argument(
        "--base_model_name",
        type=str,
        help="""\
        Nombre del modelo a usar en el entrenamiento. \
        """,
        choices=["inception-v3", "xception", "vgg16"],
        required=True,
    )
    PARSER.add_argument(
        "--x_col",
        type=str,
        help="""\
        Nombre de la columna del dataframe usada como input.\
        """,
        default="Filepath",
    )
    PARSER.add_argument(
        "--y_col",
        type=str,
        help="""\
        Nombre de la columna del dataframe usada como output.\
        """,
        default="Label",
    )
    PARSER.add_argument(
        "--epochs",
        type=int,
        help="""\
        Número de épocas de entrenamiento.\
        """,
        required=True,
    )
    PARSER.add_argument(
        "--batch_size",
        type=int,
        help="""\
        Tamaño de cada batch de entrenamiento.\
        """,
        default=32,
    )
    PARSER.add_argument(
        "--history_save_dir",
        type=str,
        help="""\
        Directorio donde se guardarán los historiales de entrenamiento.
        """,
        required=True,
    )
    PARSER.add_argument(
        "--model_save_dir",
        type=str,
        help="""\
        Directorio donde se guardarán los modelos ya entrenados.
        """,
        required=True,
    )
    PARSER.add_argument(
        "--random_seed",
        type=int,
        help="""\
        Semilla para obtener números aleatorios reproducibles.\
        """,
        default=0,
    )
    PARSER.add_argument(
        "--rotation_range",
        type=int,
        help="""\
        Rango de ángulo de rotación para data augmentation.\
        """,
        default=10,
    )
    PARSER.add_argument(
        "--width_shift_range",
        type=float,
        help="""\
        Rango de desplazamiento horizontal, entre 0 y 1, para data augmentation.\
        """,
        default=0.1,
    )
    PARSER.add_argument(
        "--height_shift_range",
        type=float,
        help="""\
        Rango de desplazamiento vertical, entre 0 y 1, para data augmentation.\
        """,
        default=0.1,
    )
    PARSER.add_argument(
        "--zoom_range",
        type=float,
        help="""\
        Rango de zoom para data augmentation.\
        """,
        default=0.1,
    )
    PARSER.add_argument(
        "--horizontal_flip",
        type=bool,
        help="""\
        True para voltear imágenes horizontalmente (data augmentation).\
        """,
        default=True,
    )
    PARSER.add_argument(
        "--vertical_flip",
        type=bool,
        help="""\
        True para voltear imágenes verticalmente (data augmentation).\
        """,
        default=True,
    )
    PARSER.add_argument(
        "--balanced_weights",
        type=bool,
        help="""\
        True para que se asigne pesos a las clases de tal forma de balancear el
        entrenamiento.\
        """,
        default=True,
    )
    PARSER.add_argument(
        "--previous_checkpoint",
        type=str,
        help="""\
        Path con pesos del modelo a ajustar.\
        """,
        default=True,
    )
    PARSER.add_argument(
        "--df_patches_val",
        type=str,
        help="""\
        DataFrame con los patches a utilizar como validación para cada modelo.\
        """,
        default=True,
    )
    FLAGS = PARSER.parse_args()

    main(
        FLAGS.patches_df_path,
        FLAGS.images_dir,
        FLAGS.min_threshold,
        FLAGS.base_model_name,
        FLAGS.x_col,
        FLAGS.y_col,
        FLAGS.epochs,
        FLAGS.batch_size,
        FLAGS.history_save_dir,
        FLAGS.model_save_dir,
        FLAGS.random_seed,
        FLAGS.rotation_range,
        FLAGS.width_shift_range,
        FLAGS.height_shift_range,
        FLAGS.zoom_range,
        FLAGS.horizontal_flip,
        FLAGS.vertical_flip,
        FLAGS.balanced_weights,
        FLAGS.previous_checkpoint,
        FLAGS.df_patches_val
    )
