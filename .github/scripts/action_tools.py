# Copyright (c) OpenMMLab. All rights reserved.
import os

import fire


def add_summary(csv_path: str):
    """Add csv file to github step summary.

    Args:
        csv_path (str): Input csv file.
    """
    _, fname = os.path.split(csv_path)
    typ, _ = os.path.splitext(fname)
    summary_file = os.environ['GITHUB_STEP_SUMMARY']
    with open(csv_path, 'r') as fr, open(summary_file, 'a') as fw:
        lines = fr.readlines()
        header = lines[0].strip().split(',')
        n_col = len(header)
        header = '|' + '|'.join(header) + '|'
        aligner = '|' + '|'.join([':-:'] * n_col) + '|'
        fw.write(header + '\n')
        fw.write(aligner + '\n')
        for line in lines[1:]:
            line = '|' + line.strip().replace(',', '|') + '|'
            fw.write(line + '\n')


if __name__ == '__main__':
    fire.Fire()
