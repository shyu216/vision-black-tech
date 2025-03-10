import os

# 指定要扫描的文件夹路径
folder_path = './'
output_md = 'README.md'

# 获取文件夹中的所有 PNG 文件
png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# 生成 Markdown 文件
with open(output_md, 'w') as md_file:
    md_file.write('# Python Demos\n\n')
    for png in png_files:
        md_file.write(f'## {png[:-4]}\n\n')
        md_file.write(f'![{png}]({png})\n\n')

print(f'Markdown file {output_md} has been generated.')