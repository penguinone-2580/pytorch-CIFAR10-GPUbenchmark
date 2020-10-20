import csv
from tqdm import tqdm
import subprocess
import json
"""
csvファイルから各種パラメータを一括読み込みする関数
Argument
csv_path: csvファイルのパス
"""
def read_parameters(csv_path, index):
    with open(csv_path, encoding='utf-8-sig') as f: #utf-8-sigでエンコードしないと1列目のキーがおかしくなる
        reader = csv.DictReader(f)
        l = [row for row in tqdm(reader)]
        parameters_dict = l[index]

    return parameters_dict

#文字列のTrueをbool値のTrueに変換しそれ以外をFalseに変換する関数
def str_to_bool(str):
    return str.lower() == "true"

#nvidia-smi --query-gpuに突っ込むオプションのデフォルト値
DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)
#GPUの情報を取得する関数
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    dict = [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

    return dict[0]
