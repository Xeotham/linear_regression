from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from matplotlib.pyplot import show, pause, subplots, scatter
from numpy import ndarray, mean, std, hstack, ones, sum
from numpy.random import randn


errors = {
    "EOF": "Error: Invalid value due to a EOF.",
    "ERR_NO_DIGIT": "Error, the provided mileage isn't composed of only digits."
}


def cost_function(m: int, X: ndarray, theta: ndarray, Y):
    return (1 / 2 * m) * sum((model(X, theta) - Y) ** 2)


def model(X: ndarray, theta: ndarray):
    return X.dot(theta)


def gradient(theta: ndarray, m: int, X: ndarray, Y: ndarray):
    return (1 / m) * X.T.dot((model(X, theta) - Y))


def gradient_descent(m: int, X: ndarray, Y: ndarray, learning_rate: float, max_iter: int, tol: float):
    theta: ndarray = randn(2, 1)
    previous_cost = float("inf")

    for i in range(0, max_iter):
        theta = theta - (learning_rate * gradient(theta, m, X, Y))
        current_cost = cost_function(m, X, theta, Y)
        if previous_cost - current_cost <= tol:
            break
        previous_cost = current_cost
    return theta



def regression(x: ndarray, y: ndarray, learning_rate: float = 0.01, max_iter: int = 100, tol: float = 1e-4):
    X = hstack((x, ones((x.shape[0], 1))))
    m = len(x)

    theta: ndarray = gradient_descent(m, X, y, learning_rate, max_iter, tol)

    return theta


def predict_price(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)


def unnormalize(value, base):
    return (value * std(base)) + mean(base)


def normalize(value, base):
    return (value - mean(base)) / std(base)


def predict(theta0, theta1, x, y):
    print(theta0, theta1)
    try:
        while True:
            input_mileage = input("Input your mileage: ")
            if len(input_mileage) == 0:
                return
            assert input_mileage.isdigit(), "ERR_NO_DIGIT"
            calculated_price = predict_price(normalize(int(input_mileage), x), theta0, theta1)
            print(f"The calculated price is: {unnormalize(calculated_price, y)}")
    except EOFError:
        print(errors["EOF"], "TEST")
        return
    except AssertionError as code:
        err_code = str(code)
        print(errors[err_code], "TEST")
        predict(theta0, theta1, x, y)


def main():
    try:
        dataset: DataFrame = read_csv("./data.csv")
        x: ndarray = dataset.iloc[:, 0].values
        y: ndarray = dataset.iloc[:, 1].values

        x = x.reshape((x.shape[0], 1))
        y = y.reshape((y.shape[0], 1))

        norm_x = (x - mean(x)) / std(x)
        norm_y = (y - mean(y)) / std(y)

        theta = regression(norm_x, norm_y, max_iter=10000, learning_rate=0.01)
        flatten = theta.flat
        predict(flatten[0], flatten[1], x, y)

        # Initialize plot
        fig, ax = subplots()
        ax.scatter(x, y, label="Data")
        line, = ax.plot(x, [ unnormalize(z, y) for z in  model(hstack((x, ones((x.shape[0], 1)))), theta)], color='red', label="Fitted Line")
        ax.set_xlabel("Mileage")
        ax.set_ylabel("Price")
        ax.set_title("Linear Regression: Mileage vs. Price (Real-Time)")
        ax.legend()
        show()


    except AssertionError as msg:
        print(str(msg))
        return
    except FileNotFoundError:
        print("FileNotFoundError: provided file not found.")
    except PermissionError:
        print("PermissionError: permission denied on provided file.")
    except EmptyDataError:
        print("EmptyDataError: Provided dataset is empty.")


if __name__ == "__main__":
    main()