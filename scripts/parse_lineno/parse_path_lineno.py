from tqdm import tqdm
import binascii
from datetime import datetime

from git import Repo, Commit

repo = Repo("scikit-learn/")

commit_ids = open("sklearn_commit.txt", "r").read().splitlines()

commit_diff_log_file = open("commit_diff.txt", "w", encoding="utf-8")

for commit_id in commit_ids:
    prev_commit = repo.commit(commit_id+"~1")
    print("Previous commit ID: " + binascii.b2a_hex(prev_commit.binsha).decode("utf-8"), file=commit_diff_log_file)
    cur_commit = repo.commit(commit_id)
    print("Current commit ID:" + commit_id, file=commit_diff_log_file)
    
    diffs = cur_commit.diff(prev_commit)
    for diff in diffs:
        if diff.a_blob is None:
            pass
            # print("="*30, file=commit_diff_log_file)
            # print("New File: " + diff.b_path, file=commit_diff_log_file)
            # print(diff.b_blob.data_stream.read().decode("utf-8"), file=commit_diff_log_file)
            # print("="*30, file=commit_diff_log_file)
        elif diff.b_blob is None:
            pass
            # print("="*30, file=commit_diff_log_file)
            # print("Delete File: " + diff.a_path, file=commit_diff_log_file)
            # print("="*30, file=commit_diff_log_file)
        else:
            previous_blob_content = diff.a_blob.data_stream.read().decode("utf-8")
            current_blob_content = diff.b_blob.data_stream.read().decode("utf-8")
            print("="*30, file=commit_diff_log_file)
            ptr_p = 0
            ptr_c = 0
            while ptr_p < len(previous_blob_content) and ptr_c < len(current_blob_content):
                if previous_blob_content[ptr_p] == current_blob_content:
                    ptr_p += 1
                    ptr_c += 1
                    continue
                # TODO: Find the next same line...  too exhausted...
                # TODO automatic locate the API's line number

            print("Previous: " + diff.a_path, file=commit_diff_log_file)
            print(diff.a_path, file=commit_diff_log_file)
            print(diff.a_blob.data_stream.read().decode("utf-8"), file=commit_diff_log_file)
            print("-"*30, file=commit_diff_log_file)
            print("Current: " + diff.b_path, file=commit_diff_log_file)
            print(diff.b_blob.data_stream.read().decode("utf-8"), file=commit_diff_log_file)
            print("="*30, file=commit_diff_log_file)
    print("~~"*20 + "\n"*8, file=commit_diff_log_file)

    # for filepath in file_path_list:
    #     print(filepath)
    #     print("-"*20)
    #     print(file_list[filepath])
    #     print("\n\n")



# commits = list(repo.iter_commits('main'))
#
# # g = Github("80fafa7d02c7e6219badcd9bbb8034991a0cbfb5")
# # repo = g.get_repo("pandas-dev/pandas")
# # print("Getting commits...")
# # commits = repo.get_commits()
# # print("Finish get commits.")
#
# keys=['bug','Bug','fix','Fix','error','Error','ERROR','check','Check',
#       'wrong','Wrong','nan','NAN','inf','issue','ISSUE','Issue','fault','Fault',
#       'fail','Fail','FAIL','crash','Crash']
#
# api_keys=['API','api','Api','missing check','null point','return','parameter',
#           'arg','para','ARG']
#
#
# f = open('ansible_py.txt', 'w', encoding='utf-8')
# f_api = open("ansible_api_py.txt", 'w', encoding='utf-8')
#
#
# print("All commits: " + str(len(commits)))
# for commit in tqdm(commits):
#     files=commit.stats.files.keys()
#     passed=False
#     for fs in files:
#         if fs.endswith('.py'):
#             passed=True
#             break
#     if not passed:
#         continue
#
#     # print(commit.stats.files)
#
#     mess=commit.message
#     if 'typo' in mess:
#         continue
#
#     txt=''
#     atxt=''
#
#
#     for k in keys:
#         if k in mess:
#             # txt+='**************************************************\n'
#             id = binascii.b2a_hex(commit.binsha).decode("utf-8")
#             txt+='commit id:'+ id +'\n'
#             # txt+='commit date:' + str(datetime.fromtimestamp(commit.committed_date))+'\n'
#             txt+='commit url:' + str(" https://github.com/ansible/ansible/commit/" + id)+'\n'
#             # txt+='commit files:'+str(commit.stats.files)+'\n'
#             txt+='commit message:' + str(commit.message)+'\n'
#             txt+='--------------------------------------------------\n\n'
#             f.write(txt)
#             break
#
#
#     for kj in api_keys:
#         if kj in mess:
#            id = binascii.b2a_hex(commit.binsha).decode("utf-8")
#            atxt+='commit id:'+ id +'\n'
#         #    atxt+='commit date:' + str(datetime.fromtimestamp(commit.committed_date))+'\n'
#            atxt+='commit url:' + str(" https://github.com/ansible/ansible/commit/" + id)+'\n'
#            # atxt+='commit files:'+str(commit.stats.files)+'\n'
#            atxt+='commit message:' + str(commit.message)+'\n'
#            atxt+='--------------------------------------------------\n\n'
#            f_api.write(atxt)
#            break
#
    