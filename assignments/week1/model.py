import numpy as np

# Citation: https://medium.com/analytics-vidhya/multiple-linear-regression-from-scratch-using-only-numpy-98fc010a1926
class LinearRegression:
    """
    A linear regression model that do not use gradient descent to fit the model.

    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.array([0])
        self.b = np.random.rand(1) * 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        train the model with the data and label

        Arguments:
            X (np.ndarray): The data.
            y (np.ndarray): The label.

        Returns:
            None

        """
        sample_size = X.shape[0]
        X = np.append(X, np.ones((sample_size, 1)), axis=1)
        y = np.array(y).reshape(len(y), 1)

        self.w = np.dot((np.linalg.inv(np.dot(X.T, X))), np.dot(X.T, y))
        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        return np.dot(X, self.w)


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def grad(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient.

        Arguments:
            X (np.ndarray): The data.
            y (np.ndarray): The label.
            w (np.ndarray): The weights.

        Returns:
            np.ndarray: gradient

        """
        n = len(X)

        return 1 / n * np.sum(np.dot((np.dot(X, w) - y.T), X), axis=1).reshape(-1, 1)

    def MSE(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the mean squared error

        Arguments:
            y (np.ndarray): The label.
            y_pred (np.ndarray): The predicted label.

        Returns:
            float: The mean squared error.

        """
        return np.mean((y - y_pred) ** 2)

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Train the model

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The label.
            lr (float, optional): The learning rate. Defaults to 0.01.
            epochs (int, optional): The number of epochs. Defaults to 1000.

        Returns:
            None

        """
        self.w = np.zeros((X.shape[1], 1))
        n = len(X)

        cost_history = [self.MSE(y, X @ self.w)]

        for epoch in range(epochs):
            if epoch % 100 == 0:
                print("Epoch: {} - MSE: {}".format(epoch, cost_history[-1]))
            y_pred = np.dot(X, self.w) + self.b
            error = y - y_pred
            cost_history.append(self.MSE(y, y_pred))

            grad_w = -(1 / n) * np.dot(X.T, error)
            grad_b = -(1 / n) * np.sum(error)

            self.w = self.w - lr * grad_w
            self.b = self.b - lr * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        return np.dot(X, self.w) + self.b
