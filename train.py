from pandas import read_csv, DataFrame
from pandas.errors import EmptyDataError
from matplotlib.pyplot import show, scatter, plot


def gradient_descent():
    """"""



def main():
    try:
        dataset: DataFrame = read_csv("./data.csv", header=0)
        # dataset = dataset.T
        km_df: DataFrame = dataset["km"]
        price_df: DataFrame = dataset["price"]
        print(f"{dataset},\n {km_df},\n {price_df} ")
        # print(f"{dataset}")
        scatter(km_df, price_df)
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