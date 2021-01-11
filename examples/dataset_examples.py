from wildboar.datasets import load_dataset, list_datasets

x, y = load_dataset("Wafer", bundle="wildboar/ucr")
print(list_datasets(bundle="wildboar/ucr-tiny"))
