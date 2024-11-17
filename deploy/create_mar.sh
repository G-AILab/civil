#!/bin/bash

TARGET_MODEL_PATH="exp_logs/civil_bi/HAR/TS2Vec/20241101_051903"
# 工作空间路径
WORKSPACE="/workspace/Civil"
# 打包模型结构文件名称
MODEL_NAME="Model-Structure.zip"

# 打包的MAR文件名称前缀
MAR_F_NAME="ts2vec-har"

STORE_PATH="model-store/$MAR_F_NAME"

cd $WORKSPACE
# 清除缓存
rm -rf $STORE_PATH 

mkdir -p $STORE_PATH

#复制需要打包的文件到临时空间，直接处理会造成路径错误
cp -r $TARGET_MODEL_PATH/* $STORE_PATH/
# 复制handler到临时空间中
cp -r devlop/civil_handler.py $STORE_PATH/

rm -rf "$STORE_PATH/$MODEL_NAME"

# 打包骨干模型和依赖
zip -r "$STORE_PATH/$MODEL_NAME"  *.py */*.py */*/*.py -i '*.py' 'algorithm/*' 'models/*' 'tasks/*' -x '*/__pycache__/*' 

# mv $STORE_PATH/$MODEL_NAME $STORE_PATH

cd $STORE_PATH

rm -rf "$STORE_PATH.mar"
torch-model-archiver --model-name $MAR_F_NAME --version 1.0 --serialized-file   ./model.pkl \
 --handler ./civil_handler.py --extra-files "$MODEL_NAME,config.json,model.pkl,linear.joblib"  -f 

mv $MAR_F_NAME.mar ../

cd $WORKSPACE

# torch-model-archiver --model-name test-model --version 1.0 --serialized-file   ./model.pkl  --handler civil_handler.py --extra-files "HARS.zip,config.json,model.pkl,linear.joblib,civil_handler.py"  -f  


# torchserve --start --model-store model-store --models Wav2Vec2=test-model.mar --ncs --ts-config ./model-store/config.properties --disable-token-auth
# torchserve --start --model-store model-store  --ncs --ts-config /workspace/model-server/config.properties --disable-token-auth --enable-model-api