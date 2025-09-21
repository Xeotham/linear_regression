from pandas import DataFrame, read_csv
from numpy import ndarray, mean, std
from train import unnormalize, normalize


errors = {
    "EOF": "Error: Invalid value due to a EOF.",
    "ERR_NO_DIGIT": "Error, the provided mileage isn't composed of only digits."
}

def estimate_price(mileage: int, theta0: float, theta1: float) -> float:
    """
    Return the estimated price based on theta0, theta1 and the mileage.
    :param mileage: Inputed mileage.
    :param theta0: Theta0.
    :param theta1: Theta1.
    :return: Value of the price based on the linear function calcul.
    """
    return theta0 + (theta1 * mileage)

def get_thetas() -> tuple[float, float]:
    """
    Function that check if the file theta.txt exist and get its value.
    :return: Value stored in theta.txt or 0 if the file does not exist.
    """
    try:
        file = open("./.theta.txt", "r")
        theta0_str: str = file.readline()
        theta1_str: str = file.readline()
        # TODO: Secure the theta0 and theta1
        return float(theta0_str), float(theta1_str)
    except FileNotFoundError:
        return 0.0, 0.0


def main():
    try:
        theta = get_thetas()
        dataset: DataFrame = read_csv("./data.csv", header=0)
        mileage: ndarray = dataset["km"].values
        price: ndarray = dataset["price"].values

        while True:
            input_mileage = input("Input your mileage: ")
            if len(input_mileage) == 0:
                return
            assert input_mileage.isdigit(), "ERR_NO_DIGIT"
            calculated_price = estimate_price(normalize(int(input_mileage), mileage), theta[0], theta[1])
            print(f"The calculated price is: {unnormalize(calculated_price, price)}")
    except EOFError:
        print(errors["EOF"])
        return
    except AssertionError as code:
        err_code = str(code)
        print(errors[err_code])
        main()


if __name__ == "__main__":
    main()