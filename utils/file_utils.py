def save_list_to_file(list, file_path):
    with open(file_path, 'w') as fp:
        fp.write('\n'.join(list))


def read_list_from_file(path: str):
    with open(path) as f:
        list = f.readlines()
    return list
