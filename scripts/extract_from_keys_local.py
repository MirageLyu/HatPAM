# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:45:41 2021

@author: 11834
"""
from tqdm import tqdm
import binascii
from datetime import datetime

from git import Repo, Commit

repo = Repo("ansible/")
commits = list(repo.iter_commits('devel'))

# g = Github("80fafa7d02c7e6219badcd9bbb8034991a0cbfb5")
# repo = g.get_repo("pandas-dev/pandas")
# print("Getting commits...")
# commits = repo.get_commits()
print("Finish get commits.")

keys=['bug','Bug','fix','Fix','error','Error','ERROR','check','Check',
      'wrong','Wrong','nan','NAN','inf','issue','ISSUE','Issue','fault','Fault',
      'fail','Fail','FAIL','crash','Crash']

api_keys=['API','api','Api','missing check','null point','return','parameter',
          'arg','para','ARG']


f = open('ansible_py.txt', 'w', encoding='utf-8')
f_api = open("ansible_api_py.txt", 'w', encoding='utf-8')


print("All commits: " + str(len(commits)))
for commit in tqdm(commits):
    files=commit.stats.files.keys()
    passed=False
    for fs in files:
        if fs.endswith('.py'):
            passed=True
            break
    if not passed:
        continue

    # print(commit.stats.files)

    mess=commit.message
    if 'typo' in mess:
        continue

    txt=''
    atxt=''


    for k in keys:
        if k in mess:            
            # txt+='**************************************************\n'
            id = binascii.b2a_hex(commit.binsha).decode("utf-8")
            txt+='commit id:'+ id +'\n'
            # txt+='commit date:' + str(datetime.fromtimestamp(commit.committed_date))+'\n'
            txt+='commit url:' + str(" https://github.com/ansible/ansible/commit/" + id)+'\n'
            # txt+='commit files:'+str(commit.stats.files)+'\n'
            txt+='commit message:' + str(commit.message)+'\n'
            txt+='--------------------------------------------------\n\n'
            f.write(txt)
            break

        
    for kj in api_keys:
        if kj in mess:
           id = binascii.b2a_hex(commit.binsha).decode("utf-8")
           atxt+='commit id:'+ id +'\n'
        #    atxt+='commit date:' + str(datetime.fromtimestamp(commit.committed_date))+'\n'
           atxt+='commit url:' + str(" https://github.com/ansible/ansible/commit/" + id)+'\n'
           # atxt+='commit files:'+str(commit.stats.files)+'\n'
           atxt+='commit message:' + str(commit.message)+'\n'
           atxt+='--------------------------------------------------\n\n'
           f_api.write(atxt)
           break
    
    