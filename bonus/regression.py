from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from matplotlib.pyplot import show, pause, subplots
from numpy import ndarray, mean, std, hstack, ones, sum, array, zeros
from numpy.random import randn
from sys import argv
from getopt import getopt, GetoptError


errors = {
    "EOF": "Error: Invalid value due to a EOF.",
    "ERR_NO_DIGIT": "Error, the provided mileage isn't composed of only digits."
}

class Arguments:
    input_file: str = "./data.csv"
    learning_rate: float = 0.01
    max_iter: int = 1000
    tol: float = 1e-4
    query: bool = False
    plot_value: int = -1
    plot_func: callable = None
    regression_value: int = 1
    precision: bool = False

input_args = Arguments

def cost_function(m: int, X: ndarray, theta: ndarray, Y: ndarray):
    return (1 / (2 * m)) * sum((model(X, theta) - Y) ** 2)


def model(X: ndarray, theta: ndarray):
    return X.dot(theta)


def gradient(theta: ndarray, m: int, X: ndarray, Y: ndarray):
    return (1 / m) * X.T.dot((model(X, theta) - Y))


def gradient_descent(m: int, X: ndarray, Y: ndarray, learning_rate: float, max_iter: int, tol: float):
    theta: ndarray = zeros((len(X[0]), 1))
    previous_cost = float("inf")

    if input_args.plot_value == 2:
        input_args.plot_func = input_args.plot_func(theta)


    for i in range(0, max_iter):
        theta -= (learning_rate * gradient(theta, m, X, Y))
        current_cost = cost_function(m, X, theta, Y)
        if previous_cost - current_cost <= tol:
            break
        previous_cost = current_cost

        if input_args.plot_value == 2:
            input_args.plot_func(theta)
    return theta


def create_X(x):
    X = x ** 0

    for i in range(input_args.regression_value - 1, -1, -1):
        X = hstack((x ** (input_args.regression_value - i), X))
    return (X)


def regression(x: ndarray, y: ndarray, learning_rate: float, max_iter: int, tol: float):
    X = create_X(x)
    m = len(x)

    theta: ndarray = gradient_descent(m, X, y, learning_rate, max_iter, tol)

    if input_args.precision:
        print(f"Precision: {cost_function(m, X, theta, y)}")

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
        print("\n",errors["EOF"], sep="")
        return
    except AssertionError as code:
        err_code = str(code)
        print(errors[err_code])
        predict(theta, x, y)


def plot_handle(X: ndarray, x: ndarray, y: ndarray, theta: ndarray = array([0])):
    fig, ax = subplots()
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")
    ax.set_title("Linear Regression: Mileage vs. Price")

    if input_args.plot_value == 0 or input_args.plot_value == 1:
        ax.scatter(x, y, label="Data")
        ax.legend()
        if input_args.plot_value == 1:
            predicted_norm_y = model(X, theta)
            predicted_y = unnormalize(predicted_norm_y, y)
            ax.plot(x, predicted_y, color='red', label="Fitted Line")
            ax.legend()
    elif input_args.plot_value == 2:
        ax.scatter(x, y, label="Data")
        def show_plot(first_theta):
            predicted_norm_y = model(X, first_theta)
            predicted_y = unnormalize(predicted_norm_y, y)
            line, = ax.plot(x, predicted_y, color='red', label="Fitted Line")
            ax.legend()
            def update_data(updated_theta):
                updated_norm_y = model(X, updated_theta)
                updated_y = unnormalize(updated_norm_y, y)
                line.set_ydata(updated_y)
                pause(0.01)
            return update_data
        return show_plot


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
                                         2 = animated regression line.
    -r --regression=    : Precise the level of regression. 1 = Linear Regression (default),
                                                           2 >= Polynomial of n degrees.
    --precision:        : Calculate the precision of the algorithm.
    :return:
    """

    args = argv[1:]

    short_flags: str = "f:l:i:t:p:r:q"
    long_flags: list[str] = ["file=", "learning_rate=", "iter=", "tolerance=", "query", "plot=", "regression=", "precision", "help"]

    opts, args = getopt(args, short_flags, long_flags)

    for opt, arg in opts:
        if opt in ('-f', '--file'):
            input_args.input_file = arg
        elif opt in ('-l', '--learning_rate'):
            input_args.learning_rate = float(arg)
        elif opt in ("-i", "--iter"):
            input_args.max_iter = int(arg)
        elif opt in ("-t", "--tolerance"):
            input_args.tol = float(arg)
        elif opt in ("-q", "--query"):
            input_args.query = True
        elif opt in ("-p", "--plot"):
            assert int(arg) <= 2, "Error: The plot parameter must be between 0 and 2."
            assert int(arg) >= 0, "Error: The plot parameter mustn't be negative."
            input_args.plot_value = int(arg)
        elif opt in ("-r", "--regression"):
            assert int(arg) >= 1, "Error: The regression parameter must be greater than 1."
            input_args.regression_value = int(arg)
        elif opt in ("--precision"):
            input_args.precision = True
        elif opt in ("--help"):
            print("python3 [flags]\n"
                   "    -f --file=          : Input the csv file.\n"
                   "    -l --learning_rate= : Define the learning rate.\n"
                   "    -i --iter=          : Define the number of iteration.\n"
                   "    -t --tolerance=     : Define the tolerance of the regression.\n"
                   "    -q --query          : Launch the prediction input query.\n"
                   "    -p --plot=          : Show the plot. 0 = Just the data Scatter,\n"
                   "                                         1 = regression line,\n"
                   "                                         2 = animated regression line (default).\n"
                   "    -r --regression=    : Precise the level of regression. 1 = Linear Regression (default),\n"
                   "                                                           2 >= Polynomial of n degrees.\n"
                   "    --precision:        : Calculate the precision of the algorithm.")
            return False
    return True


def main():
    try:
        if get_flags() is False:
            return

        dataset: DataFrame = read_csv(input_args.input_file)
        x: ndarray = dataset.iloc[:, 0].values
        y: ndarray = dataset.iloc[:, 1].values

        x = x.reshape((x.shape[0], 1))
        y = y.reshape((y.shape[0], 1))

        norm_x = (x - mean(x)) / std(x)
        norm_y = (y - mean(y)) / std(y)


        if input_args.plot_value == 2:
            input_args.plot_func = plot_handle(create_X(norm_x), x, y)
        theta = regression(norm_x, norm_y, input_args.learning_rate, input_args.max_iter, input_args.tol)



        if input_args.query is True:
            predict(theta, x, y)
        else:
            file = open("./.theta.txt", "w")
            file.write(f"{theta.flat[0]}\n{theta.flat[1]}")

        if 2 > input_args.plot_value >= 0:
            plot_handle(create_X(norm_x), x, y, theta)
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
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    except GetoptError as err:
        print(f"GetoptError: {err}")


if __name__ == "__main__":
    main()