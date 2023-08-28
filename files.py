import json
import os.path
from os import walk


act_path: str = os.path.abspath(os.path.dirname(__file__))


def get_dir(path: str):
    return os.path.join(act_path, "files", path)


def clean_name(file: str) -> str:
    return file[file.index('-') + 1:file.index(".")] \
        if file[0].isdigit() else file[:file.index(".")]


def get_files(dir: str):
    files = list(next(walk(get_dir(dir)), (None, None, []))[2])
    return [clean_name(file) for file in files]


def save_in_record(point: dict) -> None:
    with open('record.json', 'w') as record:
        json.dump(point, record, indent=2)


def read_json():
    with open('record.json', 'r') as record:
        data = json.load(record)
    return data


def get_from_record(reg: str, key: str, value: list | str = None) -> str | None:
    try:
        # Intenta recuperar del json el registro de 'reg' y
        # de ese registro la clave que contiene el valor buscado
        valores = read_json()[reg][key]
        return valores
    except FileNotFoundError:
        # si el archivo no existe lo crea con el formato por defecto
        # y le guarda el valor buscado
        structure = {'sar': {}, 'users': {}, 'config': {}}
        structure[reg] = value
        save_in_record(structure)
    except KeyError:
        # Si el archivo existe pero no tiene registro que se busca, le agrega la clave y valor
        act = read_json()
        act[reg][key] = value
        save_in_record(act)
    return None


def file(folder, name):
    return os.path.join("files", folder,
                        f"{name}.html" if folder == "mapas" else f"{name}.png")


def file_name(dir: str, tipo: str):
    files = get_files(dir)
    nummber = int(files[-1][4:-5])
    name = f"{dir[:-1]}00{nummber + 1}{tipo}"
    return os.path.join("files", dir, name)
