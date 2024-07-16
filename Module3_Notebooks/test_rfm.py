import numpy as np
import rfm


def test_rfm():
    x_train = np.array([[1., 0], [2, 1], [3, 0], [4, 1], [5, 0], [6, 1]])
    y_train = x_train[:, 0]
    y_train = y_train.reshape(-1, 1)

    model = rfm.RFM()
    model = model.fit(x_train, y_train, num_iters=2, reg=1e-4,
                      centering=True, verbose=False, diag_only=False,
                      verify_gradients=True)

    M = model.get_M()
    expectedc_M = np.array([
        [0.07748122, 0.01431913],
        [0.01431913, 0.0038696]]
    )
    assert np.allclose(M, expectedc_M)
