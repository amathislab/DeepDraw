import os
import argparse
import pickle
import numpy as np

PATH_TO_DATA = './svm_data/'
PATH_TO_RESULTS = './ovo_results/'

os.system('sudo pip install h5py')
import h5py

from sklearn.svm import LinearSVC

def make_trainval(data, labels):
    num_examples, num_inputs, num_time = data.shape
    num_train = int(0.9*num_examples)
    train_X = data[:num_train].reshape(-1, num_inputs*num_time)
    train_y = labels[:num_train]
    val_X = data[num_train:].reshape(-1, num_inputs*num_time)
    val_y = labels[num_train:]
    return train_X, train_y, val_X, val_y

def grid_search(train_X, train_y, val_X, val_y):
    c_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    myclassifiers = []
    myscores = []
    for c in c_grid:
        mylinearsvc = LinearSVC(loss='hinge', C=c)
        mylinearsvc.fit(train_X, train_y)
        myclassifiers.append(mylinearsvc)
        myscores.append(mylinearsvc.score(val_X, val_y))

    return myclassifiers, myscores, c_grid

def main(args):
    pair = [args.class1, args.class2]

    if args.suppressed:
        dataset_list = ['svm_horizontal.hdf5', 'svm_vertical.hdf5',
                        'svm_size1.hdf5', 'svm_size2.hdf5', 'svm_size3.hdf5',
                        'svm_speed1.hdf5', 'svm_speed2.hdf5', 'svm_speed3.hdf5', 'svm_speed4.hdf5',
                        'svm_sp1.hdf5', 'svm_sp2.hdf5', 'svm_sp3.hdf5', 'svm_sp4.hdf5',
                        'svm_sp5.hdf5', 'svm_sp6.hdf5']
        result_keys = ['horizontal', 'vertical', 'size1', 'size2', 'size3', 'speed1', 'speed2',
                       'speed3', 'speed4', 'sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6']
    else:
        dataset_list = ['svm_train.hdf5']
        result_keys = [args.input_type]

    results = {}
    for idx in range(len(dataset_list)):
        with h5py.File(os.path.join(PATH_TO_DATA, dataset_list[idx]), 'r') as mydataset:
            data = mydataset[args.input_type][()]
            labels = mydataset['label'][()]

        valid_labels = np.array([label in pair for label in labels])
        mydata = data[valid_labels]
        mylabels = labels[valid_labels]

        train_X, train_y, val_X, val_y = make_trainval(mydata, mylabels)

        temp_results = grid_search(train_X, train_y, val_X, val_y)
        results[result_keys[idx]] = temp_results

    prefix = '_suppressed_' if args.suppressed else '_full_variability_'
    name = args.input_type + prefix + '_'.join(str(i) for i in pair) + '.p'
    pickle.dump(results, open(os.path.join(PATH_TO_RESULTS, name), 'wb'))
    print('Finished saving everything.')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVM Binary OVO Classification')
    parser.add_argument('--class1', type=int, help='Class1 label')
    parser.add_argument('--class2', type=int, help='Class2 label')
    parser.add_argument('--input_type', type=str, help='Type of input representation')
    parser.add_argument('--suppressed', dest='suppressed', action='store_true')
    parser.add_argument('--full_variability', dest='suppressed', action='store_false')
    parser.set_defaults(suppressed=True)

    main(parser.parse_args())
