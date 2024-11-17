import torch
from ts.torch_handler.base_handler import BaseHandler

import os
import io

import zipfile
import joblib
import numpy as np

MODELSZIP = "Model-Structure.zip"


class CivilsHandler(BaseHandler):
    def initialize(self, context):
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = True
        # self.logger = get_logger()
        self.context = context

        self.map_location, self.device, self.use_gpu = (
            ("cuda", torch.device("cuda:" + str(gpu_id)), True)
            if torch.cuda.is_available()
            else ("cpu", torch.device("cpu"), False)
        )

        # If not already extracted, Extract model source code
        print(model_dir)
        with zipfile.ZipFile(os.path.join(model_dir, MODELSZIP), "r") as zip_ref:
            zip_ref.extractall(model_dir)

        from configs.LoadConfig import load_json_config
        from tasks import eval_protocols

        """加载模型及环境初始化"""

        args = load_json_config(f"{model_dir}/config.json")
        self.args = args

        self.model = self.build_model(args)
        self.model.load(f"{model_dir}/model.pkl")

        if args.eval_protocol == "mlp":
            args.num_cluster = args.num_cluster[0]
            self.task_proto = eval_protocols.Linear_probe(
                args.feat_dim, args.num_cluster
            )
            self.task_proto.load_state_dict(
                torch.load(f"{model_dir}/model_{args.eval_protocol}.pkl")["mlp_model"]
            )
        else:
            self.task_proto = joblib.load(f"{model_dir}/{args.eval_protocol}.joblib")

        import infer_datautils

        # 数据转换与模型信息全局字典
        TRANS_DATA_DICT = {
            "TS_CoT": infer_datautils.format_ts2cot,
            "TS_SEA": infer_datautils.format_ts2sea,
            "TS2Vec": infer_datautils.format_ts2vec,
        }

        self.data_process_func = TRANS_DATA_DICT[self.args.backbone_type]

    def build_model(self, args):
        from algorithm.ts_sea import TS_SEA
        from algorithm.ts_cot import TS_CoT
        from algorithm.ts2vec import TS2Vec

        """根据配置加载模型架构"""
        if args.backbone_type == "TS_CoT":
            return TS_CoT(
                input_dims=args.in_dims,
                output_dims=args.repr_dims,
                device=self.device,
                args=args,
            )
        elif args.backbone_type == "TS_SEA":
            return TS_SEA(
                input_dims=args.in_dims,
                output_dims=args.repr_dims,
                device=self.device,
                args=args,
            )
        elif args.backbone_type == "TS2Vec":
            config = dict(
                batch_size=args.batch_size,
                lr=args.lr,
                output_dims=args.repr_dims,
                max_train_length=args.max_train_length,
            )
            return TS2Vec(input_dims=args.in_dims, device=self.device, **config)
        else:
            raise ValueError("Unknown Backbone Type")

    def preprocess(self, req):
        """预处理输入数据"""
        print("data preprocessing")
        # print(req)
        # 从请求中读取二进制数据
        print(len(req))
        binary_data = req[0].get("body")

        # 将二进制数据反序列化为 PyTorch 张量
        buffer = io.BytesIO(binary_data)
        data = torch.load(buffer)

        # 检查数据形状
        print("接收到的张量形状：", data.shape)

        # 返回响应
        return self.data_process_func(data)

    def inference(self, processed_data):
        """推理阶段"""
        # processed_data = self.preprocess(data)

        data_repr = self.model.encode_online(processed_data)

        if self.args.eval_protocol == "mlp":
            data_repr = torch.from_numpy(data_repr)
            test_pred = self.task_proto(data_repr)
            test_pred = torch.nn.functional.softmax(test_pred, dim=1)
            y_score = torch.argmax(test_pred, dim=1)
            return test_pred.tolist(), y_score.tolist()
        elif self.args.eval_protocol in ["linear", "knn"]:
            y_score = self.task_proto.predict_proba(data_repr)
        else:
            y_score = self.task_proto.score(data_repr)

        return y_score.tolist(), y_score.argmax(axis=1).tolist()

    def postprocess(self, inference_output):
        """后处理输出"""
        y_score, y_label = inference_output
        # ret = []
        # for _scm,_yl in zip(y_score,y_label):
        #     ret.append({"predictions": _yl, "scores": _scm})
        return [{"predictions": y_label, "scores": y_score}]
