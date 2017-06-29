import numpy as np

SEQ_SIZE = 512
NUM_CLASSES = 3

def inputs(data_dir, input_norm=True):
    """Load the data and labels.
    Args:
        input_norm: boolean, if data should be normalised

    Returns:
        X_train: train data
        X_val: validation data
        X_test: test data
        y_train: train labels
        y_val: validation labels
        y_test: test labels
    """
    afib_data = np.load(data_dir+'afib_data.npy')
    aflu_data = np.load(data_dir+'aflu_data.npy')
    norm_data = np.load(data_dir+'norm_data.npy')

    # Min number of samples in the sets
    min_samples = np.min([afib_data.shape[0], aflu_data.shape[0], norm_data.shape[0]])

    # Prepare the train and test data
    afib_ind = np.random.choice(afib_data.shape[0], min_samples, replace=False)
    aflu_ind = np.random.choice(aflu_data.shape[0], min_samples, replace=False)
    norm_ind = np.random.choice(norm_data.shape[0], min_samples, replace=False)

    afib_data = np.concatenate((1 * np.ones((min_samples, 1)), afib_data[afib_ind, :]), axis=1)
    aflu_data = np.concatenate((2 * np.ones((min_samples, 1)), aflu_data[aflu_ind, :]), axis=1)
    norm_data = np.concatenate((3 * np.ones((min_samples, 1)), norm_data[norm_ind, :]), axis=1)

    # 20% of the data for validation and testing
    afib_test_ind = np.random.choice(afib_data.shape[0], afib_data.shape[0] // 5, replace=False)
    aflu_test_ind = np.random.choice(aflu_data.shape[0], aflu_data.shape[0] // 5, replace=False)
    norm_test_ind = np.random.choice(norm_data.shape[0], norm_data.shape[0] // 5, replace=False)

    afib_test = afib_data[afib_test_ind, :]
    aflu_test = aflu_data[aflu_test_ind, :]
    norm_test = norm_data[norm_test_ind, :]

    afib_train = np.delete(afib_data, afib_test_ind, axis=0)
    aflu_train = np.delete(aflu_data, aflu_test_ind, axis=0)
    norm_train = np.delete(norm_data, norm_test_ind, axis=0)

    X_train = np.concatenate((afib_train[:, 1:], aflu_train[:, 1:], norm_train[:, 1:]), axis=0)
    y_train = np.concatenate((afib_train[:, 0], aflu_train[:, 0], norm_train[:, 0]), axis=0)

    all_test = np.concatenate((afib_test, aflu_test, norm_test), axis=0)
    test, val = np.split(all_test, [all_test.shape[0] // 2], axis=0)

    X_test = test[:, 1:]
    y_test = test[:, 0]
    X_val = val[:, 1:]
    y_val = val[:, 0]

    # The 1e-9 avoids dividing by zero
    if input_norm:
        X_train -= np.mean(X_train, axis=0)
        X_train /= np.std(X_train, axis=0) + 1e-9
        X_test -= np.mean(X_test, axis=0)
        X_test /= np.std(X_test, axis=0) + 1e-9
        X_val -= np.mean(X_val, axis=0)
        X_val /= np.std(X_val, axis=0) + 1e-9

    return X_train, X_val, X_test, y_train, y_val, y_test
