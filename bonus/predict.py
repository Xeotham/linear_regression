from pandas import DataFrame, read_csv
from numpy import ndarray, mean, std, array
from train import unnormalize, normalize


errors = {
    "EOF": "Error: Invalid value due to a EOF.",
    "ERR_NO_DIGIT": "Error, the provided mileage isn't composed of only digits."
}


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


def model(X: ndarray, theta: ndarray):
    return X.dot(theta)

def get_thetas() -> ndarray:
    """
    Function that check if the file theta.txt exist and get its value.
    :return: Value stored in theta.txt or 0 if the file does not exist.
    """
    try:
        file = open("./.theta.txt", "r")
        theta0_str: str = file.readline()
        theta1_str: str = file.readline()
        # TODO: Secure the theta0 and theta1
        return array([float(theta0_str), float(theta1_str)])
    except FileNotFoundError:
        return array([0.0, 0.0])


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
        main()


if __name__ == "__main__":
    main()