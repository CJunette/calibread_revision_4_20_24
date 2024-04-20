import os


def read_hanlp_key():
    file_path = f"{os.getcwd()}/keys/hanlp_key.txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        key_str = f.read()

    return key_str



