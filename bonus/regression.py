from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from matplotlib.pyplot import show, pause, subplots, scatter
from numpy import ndarray, mean, std, hstack, ones, sum, array
from numpy.random import randn
from sys import argv
from getopt import getopt


errors = {
    "EOF": "Error: Invalid value due to a EOF.",
    "ERR_NO_DIGIT": "Error, the provided mileage isn't composed of only digits."
}

input_file: str = "./data.csv"
learning_rate: float = 0.01
max_iter: int = 100
tol: float = 1e-4
query: bool = False
plot_value: int = 2
regression_value: int = 1
precision: bool = False


def cost_function(m: int, X: ndarray, theta: ndarray, Y: ndarray):
    return (1 / (2 * m)) * sum((model(X, theta) - Y) ** 2)


def model(X: ndarray, theta: ndarray):
    return X.dot(theta)


def gradient(theta: ndarray, m: int, X: ndarray, Y: ndarray):
    return (1 / m) * X.T.dot((model(X, theta) - Y))


def gradient_descent(m: int, X: ndarray, Y: ndarray, learning_rate: float, max_iter: int, tol: float):
    theta: ndarray = randn(2, 1)
    previous_cost = float("inf")

    for i in range(0, max_iter):
        theta -= (learning_rate * gradient(theta, m, X, Y))
        current_cost = cost_function(m, X, theta, Y)
        if previous_cost - current_cost <= tol:
            break
        previous_cost = current_cost
    return theta



def regression(x: ndarray, y: ndarray, learning_rate: float, max_iter: int, tol: float):
    X = hstack((x, ones((x.shape[0], 1))))
    m = len(x)

    theta: ndarray = gradient_descent(m, X, y, learning_rate, max_iter, tol)

    return theta


def unnormalize(value, base):
    return (value * std(base)) + mean(base)


def normalize(value, base):
    return (value - mean(base)) / std(base)


def predict(theta, x, y):
    try:
        while True:
            input_mileage = input("Input your mileage: ")
            if len(input_mileage) == 0:
                return
            assert input_mileage.isdigit(), "ERR_NO_DIGIT"

            normalized_value = normalize(int(input_mileage), x)
            parsed_input = array([[normalized_value, 1]])
            calculated_price = model(parsed_input, theta)
            print(f"The calculated price is: {unnormalize(calculated_price[0][0], y)}")
    except EOFError:
        print(errors["EOF"])
        return
    except AssertionError as code:
        err_code = str(code)
        print(errors[err_code])
        predict(theta, x, y)


def get_flags():
    """
    Handle and pars flags.

    -f --file=          : Input the csv file.
    -l --learning_rate= : Define the learning rate.
    -i --iter=          : Define the number of iteration.
    -t --tolerance=     : Define the tolerance of the regression.
    -q --query          : Launch the prediction input query.
    -p --plot=          : Show the plot. 0 = Just the data Scatter,
                                         1 = regression line,
                                         2 = animated regression line (default).
    -r --regression=    : Precise the level of regression. 1 = Linear Regression (default),
                                                           2 >= Polynomial of n degrees.
    --precision:        : Calculate the precision of the algorithm.
    :return:
    """

    args = argv[1:]

    short_flags: str = "f:l:i:t:p:r:q"
    long_flags: list[str] = ["file=", "learning_rate=", "iter=", "tolerance=", "query", "plot=", "regression=", "precision"]

    try:
        opts, args = getopt(args, short_flags, long_flags)

        for opt, arg in opts:
            if opt in ('-f', '--file'):
                input_file = arg
            elif opt in ('-l', '--learning_rate'):
                learning_rate = float(arg)
            elif opt in ("-i", "--iter"):
                max_iter = int(arg)
            elif opt in ("-t", "--tolerance"):
                tol = float(arg)
            elif opt in ("-q", "--query"):
                query = True
            elif opt in ("-p", "--plot"):
                assert int(arg) <= 2, "Error: The plot parameter must be between 0 and 2."
                assert int(arg) >= 0, "Error: The plot parameter mustn't be negative."
                plot_value = int(arg)
            elif opt in ("-r", "--regression"):
                assert int(arg) >= 1, "Error: The regression parameter must be greater than 1."
                regression_value = int(arg)
            elif opt in ("--precision"):
                precision = True



    except:
        return


def main():
    try:
        print({"test", "test", "test"})
        dataset: DataFrame = read_csv(input_file)
        x: ndarray = dataset.iloc[:, 0].values
        y: ndarray = dataset.iloc[:, 1].values

        x = x.reshape((x.shape[0], 1))
        y = y.reshape((y.shape[0], 1))

        norm_x = (x - mean(x)) / std(x)
        norm_y = (y - mean(y)) / std(y)

        theta = regression(norm_x, norm_y, learning_rate, max_iter, tol)

        predict(theta, x, y)

        # X = hstack((norm_x, ones((x.shape[0], 1))))

        # After training, plot raw data and fitted line
        # fig, ax = subplots()
        # ax.scatter(x, y, label="Data")
        # # Unnormalize predictions for plotting
        # predicted_norm_y = model(X, theta)
        # predicted_y = unnormalize(predicted_norm_y, y)
        # ax.plot(x, predicted_y, color='red', label="Fitted Line")
        # ax.set_xlabel("Mileage")
        # ax.set_ylabel("Price")
        # ax.set_title("Linear Regression: Mileage vs. Price")
        # ax.legend()
        # show()

    except AssertionError as msg:
        print(str(msg))
        return
    except FileNotFoundError:
        print("FileNotFoundError: provided file not found.")
    except PermissionError:
        print("PermissionError: permission denied on provided file.")
    except EmptyDataError:
        print("EmptyDataError: Provided dataset is empty.")
    except KeyboardInterrupt:
        print("KeyboardInterrupt")


if __name__ == "__main__":
    main()