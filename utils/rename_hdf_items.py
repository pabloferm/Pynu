import h5py


def rename_hdf5_dataset(file_path, dataset_name, new_dataset_name):
    with h5py.File(file_path, "r+") as file:
        if dataset_name in file:
            dataset = file[dataset_name]
            del file[dataset_name]
            file.create_dataset(new_dataset_name, data=dataset[...])
            print(
                f"Dataset '{dataset_name}' renamed to '{new_dataset_name}' successfully."
            )
        else:
            print(f"Dataset '{dataset_name}' not found in the file.")


# Example usage
file_path = "/home/pablofer/Pheno/Fitter/ICtest/analysis.hdf5"
dataset_name = [
    "Physics Parameters/Atmospheric/FluxTilt",
    "Nuisance Parameters/Atmospheric/NuNuBarRatio",
    "Nuisance Parameters/Atmospheric/FluxNormalization_Above1GeV",
    "Nuisance Parameters/Atmospheric/FlavorRatio",
]
new_dataset_name = [
    "Physics Parameters/Atmospheric/tilt",
    "Nuisance Parameters/Atmospheric/nunubar_ratio",
    "Nuisance Parameters/Atmospheric/normalization_above1GeV",
    "Nuisance Parameters/Atmospheric/flavor_ratio",
]

for old, new in zip(dataset_name, new_dataset_name):
    rename_hdf5_dataset(file_path, old, new)
