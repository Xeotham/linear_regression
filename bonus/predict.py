from pandas import DataFrame, read_csv
from numpy import ndarray, mean, std, array
from regression import unnormalize, normalize


errors = {
    "EOF": "\nError: Invalid value due to a EOF.",
    "ERR_NO_DIGIT": "Error: the provided mileage isn't composed of only digits.",
    "ERR_T0": "Error: theta0 not found.",
    "ERR_T1": "Error: theta1 not found.",
    "ERR_INT": "\nError: Input canceled."
}


def predict(theta, x, y):
    """
    Function to predict prices based on the given thetas.
    :param theta: Theta0 and Theta1.
    :param x: List of x.
    :param y: List of y.
    """
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


def model(X: ndarray, theta: ndarray):
    """
    Model to calculate estimation.
    :param X: Matrix of values.
    :param theta: list of thetas.
    :return:
    """
    return X.dot(theta)

def get_thetas() -> tuple[float, float]:
    """
    Function that check if the file theta.txt exist and get its value.
    :return: Value stored in theta.txt or 0 if the file does not exist.
    """
    try:
        file = open("./.theta.txt", "r")
        theta0_str: str = file.readline()
        theta1_str: str = file.readline()
        assert theta0_str, "ERR_T0"
        assert theta1_str, "ERR_T1"
        return float(theta0_str), float(theta1_str)
    except FileNotFoundError:
        return 0.0, 0.0
    except:
        raise


def main():
    try:
        theta = get_thetas()
        dataset: DataFrame = read_csv("./data.csv", header=0)
        mileage: ndarray = dataset["km"].values
        price: ndarray = dataset["price"].values

        predict(theta, mileage, price)
    except EOFError:
        print(errors["EOF"])
        return
    except AssertionError as code:
        err_code = str(code)
        print(errors[err_code])
        if err_code == "ERR_NO_DIGIT":
            return main()
    except KeyboardInterrupt:
        print(errors["ERR_INT"])
    except FileNotFoundError as err:
        print(str(err))


if __name__ == "__main__":
    main()