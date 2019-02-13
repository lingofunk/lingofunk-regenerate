def get_model_name(class_label):
    return "model_" + class_label


def get_data_file_key_for_model_constructor(data_name):
    return data_name + "_path"


def log(text, prefix=""):
    print(prefix + text)
