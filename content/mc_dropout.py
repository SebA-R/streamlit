import numpy as np
import torch
import deepchem as dc
from deepchem.models import GATModel
import os

functionals_dictionary = {
    "SPW92": 0,
    "B97D": 1,
    "MPW91": 2,
    "PBE": 3,
    "BLYP": 4,
    "N12": 5,
    "B97MV": 6,
    "mBEEF": 7,
    "M06L": 8,
    "revM06L": 9,
    "MN15L": 10,
    "revTPSS": 11,
    "TPSS": 12,
    "wB97XD": 13,
    "CAMB3LYP": 14,
    "wB97XV": 15,
    "LRCwPBE": 16,
    "LRCwPBEh": 17,
    "MPW1K": 18,
    "PBE0": 19,
    "HSEHJS": 20,
    "rcamB3LYP": 21,
    "BHHLYP": 22,
    "PBE50": 23,
    "BMK": 24,
    "M06SX": 25,
    "M062X": 26,
    "wB97MV": 27,
    "wM05D": 28,
    "MN15": 29,
    "PW6B95": 30,
    "SCAN0": 31,
    "M11": 32,
    "revTPSSh": 33,
    "TPSSh": 34,
    "B3LYP": 35,
    "HFLYP": 36,
    "SOGGA11X": 37,
}


def load_model(model_name):
    recommender_state = torch.load(
        model_name, map_location=torch.device('cpu'))
    activation = torch.nn.Sigmoid()
    recommender = GATModel(
        mode='regression',
        n_tasks=38,
        batch_size=128,
        n_layers=2,
        graph_attention_layers=[128, 128],
        learning_rate=0.001,
        activation=activation,
        dropout=0.1,
        predictor_dropout=0.2
    )
    return recommender

def monte_carlo_predict(model, dataset, num_samples, functional):
    mc_predictions = []
    model.fit(dataset, nb_epoch=1)
    for _ in range(num_samples):
        model.fit(dataset, nb_epoch=1)
        with torch.no_grad():
            output = model.predict(dataset)
            output = np.array(output*100)
            output = output[:, functionals_dictionary[functional]]
            mc_predictions.append(output)

    mc_predictions = np.array(mc_predictions)
    mc_prediction_mean = np.mean(mc_predictions, axis=0)
    mc_prediction_variance = np.var(mc_predictions, axis=0)

    return mc_prediction_mean, mc_prediction_variance

def mc_dropout_main(dataset):
    model_path = os.path.join(os.path.dirname(__file__), os.pardir, 'models', 'functional_recommender.pt')
    model = load_model(model_path)
    

    return [monte_carlo_predict(model, dataset, 50, functional) for functional in functionals_dictionary.keys()]

if __name__ == "__main__":
    X_test = np.load('X_test_graph.npy', allow_pickle=True)
    y_test = np.load('final_test_gat.npy', allow_pickle=True)
    test_dataset = dc.data.NumpyDataset(X=X_test, y=y_test)
    recommender = load_model('functional_recommender.pt')

    for functional in functionals_dictionary:
        mean, variance = monte_carlo_predict(
            recommender, test_dataset, 500, functional)
        np.save(functional+'_mean.npy', mean)
        np.save(functional+'_variance.npy', variance)
        print(functional)
