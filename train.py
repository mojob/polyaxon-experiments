import argparse
import json
from argparse import ArgumentTypeError


from keras import backend as kb
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.datasets import mnist
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical

from polyaxon_client.tracking import Experiment
from polyaxon_client.tracking.contrib.keras import PolyaxonKeras

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_arguments():
    """
        Parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--log_learning_rate',
        type=int,
        default=-2,
    )
    parser.add_argument(
        '--loss_metric',
        type=str_arg,
        default='mae',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
    )
    parser.add_argument(
        '--layers',
        type=comma_separated_arg,
        default='dense:512,dense:252,dense:120',
    )
    parser.add_argument(
        '--kernel_size',
        type=int,
        default=3,
    )
    parser.add_argument(
        '--pool_size',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--first_layer_output',
        type=int,
        default=32,
    )
    args = parser.parse_args()
    return args


OPTIMIZERS = {
    'adam': optimizers.Adam,
    'rmsprop': optimizers.RMSprop,
    'sgd': optimizers.SGD,
}


def str_arg(string):
    """
    Converts input to string and checks for None and empty values
    """
    value = str(string)
    if not string or string == '':
        raise ArgumentTypeError('Value is required')
    return value


def comma_separated_arg(string):
    """
    Splits the comma-separated string into an array of strings
    """
    value = str(string)
    if not string or string == '':
        raise ArgumentTypeError('Value is required')
    return value.split(',')


def r2_keras(y_true, y_pred):
    """
    R2 is a way of getting 'accuracy' as a probability distribution between 0 and 1
    1 - residual sum of square / total sum of squares
    A high R-square of above 60%(0.60) is required for studies in the 'pure science'
    field because the behaviour of molecules and/or particles can be reasonably predicted
    to some degree of accuracy in science research; while an R-square as low as 10% is generally
    accepted for studies in the field of arts, humanities and social sciences because human behaviour
    cannot be accurately predicted, therefore, a low R-square is often not a problem in studies in the arts,
    humanities and social science field
    """
    ss_res = kb.sum(kb.square(y_true - y_pred))
    ss_tot = kb.sum(kb.square(y_true - kb.mean(y_true)))
    return 1 - ss_res / (ss_tot + kb.epsilon())


def create_data_set(img_rows, img_cols):
    """"
    Creating train and test mnist data sets
    """
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if kb.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)  # 10 = 0-9
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test, input_shape


def create_sequential_model(
        layers,
        kernel_size,
        pool_size,
        input_shape,
        first_layer_output,
):
    """
    Creates Keras sequential layers from a string
    specified in train_hyperparams.yaml.

    , is used to separate layers
    : is used to separate layer attributes.

    The first attribute (before :) is the type of layer
    (Conv2D, MaxPooling2D, Dropout, Flatten, Dense)
    The second attribute (behind :) is the size of the output array
    """
    input_layer = Conv2D(
        filters=first_layer_output,
        kernel_size=(kernel_size, kernel_size),
        activation='relu',
        input_shape=input_shape,
    )
    sequential_layers = [input_layer]
    # Add network layers
    for idx, layer in enumerate(layers):
        layer_split = layer.split(':')
        layer_type = layer_split[0]
        layer_unit_dim = None
        layer_name = None
        if len(layer_split) > 1:
            layer_unit_dim = layer_split[1]
            layer_name = str(idx) + '-' + layer_type + '-' + layer_unit_dim
        if layer_type == 'dense':
            layer_name += '-relu'
            dense_layer = Dense(
                units=int(layer_unit_dim),
                activation='relu',
                name=layer_name,
            )
            sequential_layers.append(dense_layer)
        if layer_type == 'conv2d':
            layer_name += '-relu'
            dense_layer = Conv2D(
                filters=int(layer_unit_dim),
                kernel_size=(kernel_size, kernel_size),
                activation='relu',
                name=layer_name,
            )
            sequential_layers.append(dense_layer)
        if layer_type == 'maxpooling2d':
            max_pooling_layer = MaxPooling2D(
                pool_size=(pool_size, pool_size),
            )
            sequential_layers.append(max_pooling_layer)
        if layer_type == 'flatten':
            sequential_layers.append(Flatten())
        if layer_type == 'dropout':
            dropout_layer = Dropout(
                rate=float(layer_unit_dim),
                name=layer_name,
            )
            sequential_layers.append(dropout_layer)
    output_layer = Dense(
        10,  # 0-9
        activation='softmax',
    )
    sequential_layers.append(output_layer)
    return sequential_layers


def create_result_report(model, x_test, y_test):
    """
    Create a dictionary with results from test data set
    predictions.

    Reports mae, mse, r2
    """
    prediction_on_test = model.predict(x_test)
    mae = mean_absolute_error(
        y_test,
        prediction_on_test,
    )
    mse = mean_squared_error(
        y_test,
        prediction_on_test,
    )
    r2 = r2_score(
        y_test,
        prediction_on_test,
    )
    print(
        'First 5 predictions: \n{}'.format(
            (prediction_on_test.tolist()[:5], y_test.tolist()[:5]),
        ),
    )
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'R2': float(r2),
    }


def main():
    args = parse_arguments()
    # Polyaxon setup
    # Automatically connects when ran by 'polyaxon run'
    experiment = Experiment()
    # Log all arguments
    experiment.log_params(**vars(args))
    # Data set
    # input image dimensions
    img_rows, img_cols = 28, 28
    x_train, y_train, x_test, y_test, input_shape = create_data_set(
        img_rows=img_rows,
        img_cols=img_cols,
    )
    # Polyaxon log data
    experiment.log_data_ref(
        data=x_train,
        data_name='x_train',
    )
    experiment.log_data_ref(
        data=y_train,
        data_name='y_train',
    )
    experiment.log_data_ref(
        data=x_test,
        data_name='x_test',
    )
    experiment.log_data_ref(
        data=y_test,
        data_name='y_test',
    )
    # Create layers from args
    layers = create_sequential_model(
        kernel_size=args.kernel_size,
        pool_size=args.pool_size,
        layers=args.layers,
        input_shape=input_shape,
        first_layer_output=args.first_layer_output,
    )
    # Initiate model from layers
    model = Sequential(
        layers=layers,
    )
    # Use optimizer from args
    optimizer_cls = OPTIMIZERS[args.optimizer]
    # Use loss metric from args
    model.compile(
        optimizer=optimizer_cls(lr=(10**args.log_learning_rate)),
        loss=args.loss_metric,
        metrics=[r2_keras, 'mse', 'mae', 'accuracy'],
    )
    # Fit model
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=128,
        epochs=args.epochs,
        validation_data=(x_test, y_test),
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=12,
                mode='min',
                verbose=1,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=6,
                min_lr=0,
                verbose=1,
            ),
            # Polyaxon provides a Keras callback,
            # you can use this callback with your experiment
            # to report metrics automatically
            PolyaxonKeras(
                experiment=experiment,
            ),
        ],
        verbose=2,
    )
    # Evaluate and log results
    score = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Logging the metrics by unwrapping all values
    # Not necessary when polyaxon does this in its own
    # keras callback. But nice if you are doing something
    # like a custom metric.
    report = create_result_report(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )
    experiment.log_metrics(
        **report,
    )
    print(
        model.summary(),
    )
    print(
        json.dumps(
            report,
            indent=4,
        ),
    )


if __name__ == '__main__':
    main()
