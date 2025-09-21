from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from matplotlib.pyplot import show, pause, subplots
from numpy import ndarray, mean, std


def estimate_price(theta0: float, theta1: float) -> callable:
    """
    Return the estimated price based on theta0, theta1 and the mileage.
    :param theta0: Theta0.
    :param theta1: Theta1.
    :return: Value of the price based on the linear function calcul.
    """

    def calc_estimate_price(mileage: ndarray) -> ndarray:
        return theta0 + (theta1 * mileage)

    return calc_estimate_price


def cost_function(estimate_price_func: callable, mileage: ndarray, price: ndarray):
    """
    Mean Squared Error.
    :param estimate_price_func:
    :param mileage:
    :param price:
    :return:
    """
    prediction = estimate_price_func(mileage)

    return mean((prediction - price) ** 2) / 2


def gradient_descent(mileage: ndarray, price: ndarray, learning_rate: float = 0.01, max_iter: int = 100, tol: float = 1e-4) -> tuple:
    """
    Function to find theta0 and theta1,
    which are the needed value to correctly
    make the linear regression.
    :param dataset: The actual data which will help to train the model.
    :return: calculated theta0 and theta1.
    """

    tmp_theta: list = [0, 0]
    theta: tuple = (0, 0)
    previous_cost: float = float('inf')

    # Initialize plot
    fig, ax = subplots()
    ax.scatter(mileage, price, label="Data")
    line, = ax.plot(mileage, estimate_price(tmp_theta[0], tmp_theta[1])(mileage), color='red', label="Fitted Line")
    ax.set_xlabel("Mileage")
    ax.set_ylabel("Price")
    ax.set_title("Linear Regression: Mileage vs. Price (Real-Time)")
    ax.legend()
    pause(0.01)

    for i in range(0, max_iter, 1):
        act_price_function: callable = estimate_price(tmp_theta[0], tmp_theta[1])
        prediction = act_price_function(mileage)
        error = prediction - price
        tmp_theta[0] -= learning_rate * mean(error)
        tmp_theta[1] -= learning_rate * mean(error * mileage)
        theta = (tmp_theta[0], tmp_theta[1])
        current_cost = cost_function(estimate_price(theta[0], theta[1]), mileage, price)
        if previous_cost - current_cost <= tol:
            break
        previous_cost = current_cost

        # Update the line data
        line.set_ydata(estimate_price(tmp_theta[0], tmp_theta[1])(mileage))
        ax.relim()
        pause(0.01)
    return theta


def predict_price(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)


def unnormalize(value, base):
    return (value * std(base)) + mean(base)


def normalize(value, base):
    return (value - mean(base)) / std(base)


def main():
    try:
        dataset: DataFrame = read_csv("./data.csv", header=0)
        mileage: ndarray = dataset["km"].values
        price: ndarray = dataset["price"].values

        norm_mileage = (mileage - mean(mileage)) / std(mileage)
        norm_price = (price - mean(price)) / std(price)

        theta0, theta1 = gradient_descent(norm_mileage, norm_price, max_iter=10000, learning_rate=0.01)

        file = open("./.theta.txt", "w")
        file.write(f"{theta0}\n{theta1}")

        # scatter(mileage, price, label="Data")
        # plot(mileage,
        #      [unnormalize(predict_price(theta0, theta1, normalize(x, mileage)), price) for x
        #       in mileage],
        #      color='red', label="Fitted Line")
        # xlabel("Mileage")
        # ylabel("Price")
        # title("Linear Regression: Mileage vs. Price")
        # legend()
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