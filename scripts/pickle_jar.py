import pickle
import numpy as np

files = ['fold_0.pickle', 'fold_1-3.pickle', 'tupelo_0.pickle', 'tupelo_1.pickle']

var = {}
for file_name in files:
    with open(file_name, 'rb') as file:
        var[file_name] = pickle.load(file)

lambda_list = np.logspace(start=1, stop=3, num=11)
hidden_units_list = np.array((1, 2, 4, 6, 8, 10, 12, 15))

K_1 = 10

outer_fold_errors = {'lin': np.ndarray((K_1, len(lambda_list))),
                     'ann': np.ndarray((K_1, len(hidden_units_list)))}

best_lambdas = np.ndarray((K_1,))
best_hid_units = np.ndarray((K_1,))

test_miss_rates = {key: np.ndarray((K_1,)) for key in ('base', 'lin', 'ann')}


for part in var.values():
    folds = part['used_folds']
    outer_fold_errors['lin'][folds] =\
        part['outer_fold_errors']['lin'][folds]

    outer_fold_errors['ann'][folds] =\
        part['outer_fold_errors']['ann'][folds]

    best_lambdas[folds] =\
        part['best_lambdas'][folds]

    best_hid_units[folds] =\
        part['best_hid_units'][folds]

    test_miss_rates['base'][folds] =\
        part['test_miss_rates']['base'][folds]

    test_miss_rates['lin'][folds] =\
        part['test_miss_rates']['lin'][folds]

    test_miss_rates['ann'][folds] =\
        part['test_miss_rates']['ann'][folds]

    for key in part['predictions']:
        try:
            part['predictions'][key] = np.concatenate(part['predictions'][key])

        except RuntimeError:
            part['predictions'][key] =\
                np.reshape(np.concatenate([p.detach().numpy() for p in part['predictions'][key]]), (-1))

predictions = {key: np.concatenate([part['predictions'][key] for part in var.values()]) for key in ('true',
                                                                                                    'base',
                                                                                                    'lin',
                                                                                                    'ann')}

with open(f'whole.pickle', 'wb') as file:
    pickle.dump({'outer_fold_errors': outer_fold_errors,
                 'best_lambdas': best_lambdas,
                 'best_hid_units': best_hid_units,
                 'test_miss_rates': test_miss_rates,
                 'predictions': predictions}, file)
