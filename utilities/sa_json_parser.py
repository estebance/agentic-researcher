import json
from ast import parse


def parse_sa_json():
    with open('../sa.json', 'r') as file:
        vertex_sa_dict = json.load(file)
        vertex_sa = json.dumps(vertex_sa_dict)
        vertex_sa = json.dumps(vertex_sa)
        print(vertex_sa)


if __name__ == "__main__":
    parse_sa_json()