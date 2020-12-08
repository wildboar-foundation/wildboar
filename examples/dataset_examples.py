from wildboar.datasets import load_dataset

x, y = load_dataset("Wafer", repository="wildboar/ucr")
print(list_datasets(repository="wildboar/ucr"))
