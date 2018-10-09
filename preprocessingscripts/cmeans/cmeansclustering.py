import numpy as np
import logging


from fuzzycmeans.fuzzy_clustering import FCM
from fuzzycmeans.visualization import draw_model_2d


def example():
    X = np.array([[1, 1], [1, 2], [2, 2], [9, 10], [10, 10], [10, 9], [9, 9], [20,20]])
    fcm = FCM(n_clusters=4)
    fcm.set_logger(tostdout=True, level=logging.DEBUG)
    fcm.fit(X, [0, 0, 0, 1, 1, 1, 1, 2])
    # fcm.fit(X)
    testing_data = np.array([[0, 1.9], [0.5, 2], [-1, 2], [1.9, 0.8], [9.5, 6.5], [8, 5], [15,13], [16,12], [14,14], [10,7]])
    predicted_membership = fcm.predict(testing_data)
    print("\n\ntesting data")
    print(testing_data)
    print("predicted membership")
    print(predicted_membership)
    print("\n\n")
    draw_model_2d(fcm, data=testing_data, membership=predicted_membership)

example()
