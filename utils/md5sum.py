import shutil
import subprocess
from pathlib import Path

src = '/home/public/images'
dst = ''
paths = list(Path(src).glob('[!.]*'))

files = [i for i in paths]
o = ['md5sum'] + files

result = subprocess.run(o, stdout=subprocess.PIPE)
stout = result.stdout.decode('utf-8').split('\n')[:-1]

lst = []
for out in stout:
    md5, filepath = out.split()[0], ' '.join(out.split()[1:])
    if md5 in lst:
        continue
    else:
        lst.append(md5)
        shutil.copy(filepath, dst)
