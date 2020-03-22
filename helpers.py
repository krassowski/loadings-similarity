from collections import defaultdict

from pandas import Categorical, DataFrame
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow, cm


def loadings_frame(loadings, feature_names):
    loadings = (
        DataFrame(loadings, columns=feature_names)
        .stack().reset_index()
        .rename(columns={'level_0': 'component', 'level_1': 'feature', 0: 'value'})
    )
    loadings.feature = Categorical(loadings.feature, ordered=True, categories=feature_names)
    return loadings


def collinear_columns_generator(data, noise_ratio: float, normalise=False, repeats=1):
    df = DataFrame(data)
    for column_name in df.columns:
        data_with_noise = df.copy()
        for i in range(repeats):
            column = data_with_noise[column_name]

            if normalise:
                column = (column - column.mean()) / column.std()

            # reduce the variation (information contribution) by subtracting the deviation
            deviation = column + column.mean()
            noisy_colinear = column - deviation * noise_ratio

            data_with_noise[f'{column_name}_colinear_{i}_noise_{noise_ratio}'] = noisy_colinear
            yield data_with_noise


def plot_digit(digit_data):
    return imshow(
        digit_data.reshape(28, 28),
        cmap=cm.gray,
        interpolation='nearest',
        clim=[0, 255],
    )


def plot_observation(data, observation_id=0):
    return plot_digit(data[observation_id])


def plot_digits_row(data, encoding, average_of=1):
    ids_by_digit = defaultdict(list)
    for digit in range(10):
        for i, target in enumerate(encoding):
            if digit == int(target) and len(ids_by_digit[digit]) < average_of:
                ids_by_digit[digit].append(i)
    plt.figure(figsize=(20, 4))

    for digit, ids in ids_by_digit.items():
        plt.subplot(1, 10, digit + 1);
        plt.axis('off');
        plot_digit(sum(data[ids]) / len(ids))
    plt
