from wildboar.datasets import (
    list_bundles,
    list_collections,
    list_datasets,
    list_repositories,
    load_dataset,
    load_datasets,
)

print(list_datasets(repository="wildboar/ucr:no-missing"))
print(list_datasets(repository="wildboar/outlier:1.0:hard"))
print(list_repositories())
print(list_bundles("wildboar"))

for dataset, (x, y) in load_datasets(
    repository="wildboar/ucr-tiny",
    filter=["n_timestep>10", "n_samples<=200", "n_labels<=3"],
):
    print(dataset)

for dataset in list_datasets(repository="wildboar/ucr", collection="bake-off"):
    print(dataset)

print(list_collections("wildboar/ucr"))

load_dataset("GunPoint", repository="wildboar/ucr-tiny")
