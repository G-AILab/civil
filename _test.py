# from civil_handler222 import CivilsHandler
# from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
# if __name__ == '__main__':

#     # =====和torch-model-archiver类似=====
#     context = MockContext(
#         model_pt_file="exp_logs/civil/RoadBank/TS_CoT/20241101_103432/model.pkl",
#         model_dir="model-store",
#         model_file="model.py",
#         gpu_id="0",
#         model_name="model",
#     )
#     handler = CivilsHandler()
#     print(context.manifest)
#     print(context.system_properties)
#     handler.initialize(context)


import torch
import requests
import io
import random
# 定义接口地址
# url = "http://127.0.0.1:28080/predictions/ts2vec-har"
url = "http://192.168.5.246:19080/predictions/ts2vec-har"

# 创建形状为 (10, 9, 128) 的 PyTorch 数据
dataset_root = "/workspace/CA-TCC/data"
data_path = f"{dataset_root}/HAR/"
train_ = torch.load(data_path + "test.pt")
for idx in range(0,len(train_['samples']),5):

    indices = random.sample(range(len(train_["samples"])), 5)
    ratio_data = train_["samples"][indices]

    # dat_dict = dict()
    # dat_dict["samples"] = ratio_data
    # torch.save(dat_dict, "subset_data.pt")
    # data_ = torch.load("subset_data.pt")["samples"]
    file_buffer = io.BytesIO()
    torch.save(ratio_data, file_buffer)
    file_buffer.seek(0)
    file_size = len(file_buffer.getvalue())

    # print(data_.shape)

    # 将数据保存为二进制流
    buffer = io.BytesIO()
    torch.save(ratio_data, buffer)
    buffer.seek(0)  # 将指针移动到流的开头

    # 定义请求头（用于二进制流）
    headers = {"Content-Type": "application/octet-stream"}

    # 发送 POST 请求
    try:
        response = requests.post(url, headers=headers, data=buffer.getvalue())
        # 检查响应状态
        if response.status_code == 200:
            print("请求成功，响应内容：")
            print(response.json())
        else:
            print(f"请求失败，状态码：{response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"请求发生错误：{e}")

    break
