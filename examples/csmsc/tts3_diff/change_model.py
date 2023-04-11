import paddle
import numpy as np
import sys

# 把 snapshot_iter_65000.pdz 的 fs2 模块的参数传入 snapshot_iter_32.pdz， 为了 训练 diffusion 不用再训练 fs2 模块了
model_fs2_path = "snapshot_iter_76000.pdz"
#model_diff_path = "snapshot_iter_old_12.pdz"
#new_model_path = "./snapshot_iter_12.pdz"
model_diff_path = sys.argv[1]
new_model_path = sys.argv[2]



model_fs2 = paddle.load(model_fs2_path)["main_params"]
model_diff_all = paddle.load(model_diff_path)
new_model = model_diff_all["main_params"]


for item in model_fs2:
    # if "fs2" in item:
    new_item = "fs2." + item
    new_model[new_item] = model_fs2[item]
    print(item, new_item)
paddle.save(model_diff_all, new_model_path)
