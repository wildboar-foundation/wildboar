from wildboar.datasets import (
    load_dataset,
    list_datasets,
    list_repositories,
    list_bundles,
)

x, y = load_dataset("Wafer", repository="wildboar/ucr")
print(list_datasets(repository="wildboar/ucr-tiny"))
print(list_repositories())
print(list_bundles("wildboar"))
