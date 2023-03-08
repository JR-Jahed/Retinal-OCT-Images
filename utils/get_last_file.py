import os


def get_last_file(model_path):
    f = []
    for (dir_path, dir_names, filenames) in os.walk(model_path):
        f.extend(filenames)
        break

    f.sort()

    last_file = f[-1]
    return last_file.split(".")[0]