"""
将 stereo_e2e_occ 目录下所有 .py 文件打包到一个 txt 文件中。

输出格式:
  1. 统计信息 (文件数、总行数)
  2. 文件结构树
  3. 每个文件的完整代码 (带分隔线和行数标注)

用法:
  python pack_code.py
  python pack_code.py -o custom_output.txt
"""
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', default=None, help='输出文件路径')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_name = os.path.basename(script_dir)

    # 收集 .py 文件 (排除自身和 __pycache__)
    py_files = []
    for root, dirs, files in os.walk(script_dir):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for f in sorted(files):
            if f.endswith('.py') and f != 'pack_code.py':
                full = os.path.join(root, f)
                rel = os.path.relpath(full, script_dir)
                py_files.append((rel.replace('\\', '/'), full))

    # 读取所有文件内容
    file_contents = {}
    total_lines = 0
    for rel, full in py_files:
        with open(full, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()
        file_contents[rel] = lines
        total_lines += len(lines)

    # 输出路径
    out_path = args.output or os.path.join(script_dir, f'{project_name}_all_code.txt')

    with open(out_path, 'w', encoding='utf-8') as out:
        # --- 头部 ---
        out.write('=' * 80 + '\n')
        out.write(f'代码摘要 - {project_name}\n')
        out.write('=' * 80 + '\n\n')
        out.write(f'统计信息:\n')
        out.write(f'  文件数: {len(py_files)}\n')
        out.write(f'  总行数: {total_lines}\n\n')

        # --- 文件树 ---
        out.write('=' * 80 + '\n')
        out.write('文件结构\n')
        out.write('=' * 80 + '\n\n')
        out.write(f'{project_name}/\n')
        for i, (rel, _) in enumerate(py_files):
            is_last = (i == len(py_files) - 1)
            prefix = '\u2514\u2500\u2500 ' if is_last else '\u251c\u2500\u2500 '
            line_count = len(file_contents[rel])
            out.write(f'{prefix}{rel}  ({line_count} \u884c)\n')
        out.write('\n')

        # --- 每个文件的完整代码 ---
        out.write('=' * 80 + '\n')
        out.write('文件内容\n')
        out.write('=' * 80 + '\n\n')

        for rel, _ in py_files:
            lines = file_contents[rel]
            out.write('-' * 80 + '\n')
            out.write(f'文件: {rel} ({len(lines)} 行)\n')
            out.write('-' * 80 + '\n')
            out.write(''.join(lines))
            if lines and not lines[-1].endswith('\n'):
                out.write('\n')
            out.write('\n\n')

    print(f'已打包 {len(py_files)} 个文件, {total_lines} 行代码')
    print(f'输出: {out_path}')


if __name__ == '__main__':
    main()
