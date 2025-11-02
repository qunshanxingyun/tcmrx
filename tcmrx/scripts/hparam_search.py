"""简单的超参数搜索工具。

该脚本针对过滤/重加权相关的关键参数构建网格，并顺序调用
``run_train.py`` 执行多组实验。默认网格覆盖靶点裁剪上限、质量阈值
以及频率重加权的幂指数，可通过 ``--grid`` 传入 YAML 文件进行自定义。
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Any

import yaml


DEFAULT_GRID = {
    "filtering.disease_target_trimming.max_items": [400, 600],
    "filtering.disease_target_trimming.mass_threshold": [0.85, 0.9],
    "filtering.frequency_reweighting.disease.power": [1.2, 1.5],
    "filtering.frequency_reweighting.formula.power": [1.0, 1.2],
    "filtering.formula_target_trimming.max_items": [1200, 1500],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCM-RX 超参数搜索")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="基础配置文件路径",
    )
    parser.add_argument(
        "--paths",
        type=str,
        default="config/paths.yaml",
        help="数据路径配置文件",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        help="自定义参数网格 YAML，格式: {dotted.key: [v1, v2]}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="hparam_runs",
        help="临时配置与日志输出目录",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印组合而不执行训练",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="额外传递给 run_train.py 的参数",
    )
    return parser.parse_args()


def load_grid(path: str | None) -> Dict[str, List[Any]]:
    if path is None:
        return DEFAULT_GRID

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("网格配置必须是字典，形如 {dotted.key: [values]}")

    for key, values in data.items():
        if not isinstance(values, list):
            raise ValueError(f"网格参数 {key} 的值必须是列表")

    return data


def iter_combinations(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


def set_nested_value(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def main():
    args = parse_args()
    base_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    grid = load_grid(args.grid)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_args = args.extra_args or []

    for idx, override in enumerate(iter_combinations(grid)):
        config_copy = deepcopy(base_config)
        for dotted_key, value in override.items():
            set_nested_value(config_copy, dotted_key, value)

        label = "_".join(f"{k.split('.')[-1]}-{v}" for k, v in override.items())
        experiment_name = f"grid_{idx:02d}_{label}"

        tmp_config = tempfile.NamedTemporaryFile(
            suffix=".yaml",
            prefix=f"hparam_{idx:02d}_",
            dir=output_dir,
            delete=False,
        )
        with tmp_config:
            yaml.safe_dump(config_copy, tmp_config)

        cmd = [
            sys.executable,
            "-m",
            "tcmrx.scripts.run_train",
            "--config",
            str(tmp_config.name),
            "--paths",
            args.paths,
            "--experiment",
            experiment_name,
        ] + extra_args

        print(f"[HP-SEARCH] combo {idx}: {override}")
        if args.dry_run:
            continue

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
