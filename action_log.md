# action_log

# 一、openmmlab==>mmaction2

## 1、Log

```python
# 1、MMAction2 为视频理解任务实现了多种多样的算法，包括行为识别，时序动作定位，时空动作检测，基于骨骼点的行为识别，以及视频检索。
https://github.com/open-mmlab/mmaction2
# 环境：
"""
docker run -dit --name action -p 5322:22 -p 5330-5399:5330-5399 -v /home/dsl7625279/dataset:/root/dataset -v /home/dsl7625279/project:/root/project -v /dev/shm:/dev/shm --gpus all --privileged --entrypoint /bin/bash nvidia/cuda:11.8.0-cudnn8-devel-ubuntu18.04-v0.3beta

docker exec -it action /bin/bash

service ssh start

nohup jupyter-notebook --no-browser --ip 0.0.0.0 --port 5340 --allow-root > jupyter.nohub.out &

http://10.8.5.43:5340/tree?
"""
# 依赖安装
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch  # This command will automatically install the latest version PyTorch and cudatoolkit, please check whether they match your environment.
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet  # optional
mim install mmpose  # optional
git clone https://github.com/open-mmlab/mmaction2.git
# scp -r -P 60022 mmaction2.zip dsl7625279@10.8.5.43:/home/dsl7625279/project/research/action/
scp -r -P 5322 mmaction2.zip root@10.8.5.43:/root/project/research/action/
cd mmaction2
pip install -v -e .

# demo: 参考 https://zhuanlan.zhihu.com/p/606907078 https://github.com/open-mmlab/mmaction2/blob/main/demo/README.md
""" 测试tsn
# 模型文件1
scp -r -P 5322 tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth root@10.8.5.43:/root/project/research/action/mmaction2/checkpoints
# demo测试
python demo/demo_inferencer.py demo/demo.mp4 --rec tsn --rec-weights checkpoints/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth --label-file tools/data/kinetics/label_map_k400.txt --vid-out-dir demo_out
 
scp -r -P 5322 root@10.8.5.43:/root/project/research/action/mmaction2/demo_out ./

# 模型文件2 测试
python demo/demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py checkpoints/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --out-filename demo_out/output_video.mp4
# scp -r -P 5322 root@10.8.5.43:/root/project/research/action/mmaction2/demo/demo_out ./

# test
python tools/test.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py checkpoints/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth --dump result.pkl
"""

"""
scp -r -P 5322 i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth root@10.8.5.43:/root/project/research/action/mmaction2/checkpoints

python tools/visualizations/vis_cam.py demo/demo_configs/i3d_r50_32x2x1_video_infer.py \
    checkpoints/i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb_20220812-e213c223.pth demo/demo.mp4 \
    --target-layer-name backbone/layer4/1/relu --fps 10 \
    --out-filename demo_out/demo_gradcam.gif
"""

"""
scp -r -P 5322 tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth root@10.8.5.43:/root/project/research/action/mmaction2/checkpoints

scp -r -P 5322 sample-mp4-file.mp4 root@10.8.5.43:/root/project/research/action/mmaction2/demo/

python demo/long_video_demo.py demo/demo_configs/tsn_r50_1x1x8_video_infer.py checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth demo/sample-mp4-file.mp4 tools/data/kinetics/label_map_k400.txt demo_out/long_video.mp4 --input-step 3 --device cpu --threshold 0.2

scp -r -P 5322 root@10.8.5.43:/root/project/research/action/mmaction2/demo_out ./
"""

"""基于骨架的动作识别
scp -r -P 5322 posec3d_k400.pth root@10.8.5.43:/root/project/research/action/mmaction2/checkpoints

python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo_out/demo_skeleton_out.mp4 \
    --config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --checkpoint checkpoints/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt

人检测：使用Faster-RCNN检测视频中的人，获取每个人的边界框。
姿态估计：对检测到的人体，使用HRNet提取关键点坐标，得到每个人的姿态信息。
动作识别：使用PoseC3D分析提取到的姿态信息，进行动作识别。

python demo/demo_skeleton.py demo/demo_skeleton.mp4 demo_out/demo_skeleton_out_stgcn.mp4 \
    --config configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py \
    --checkpoint checkpoints/stgcn_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d_20221129-484a394a.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --det-cat-id 0 \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --label-map tools/data/skeleton/label_map_ntu60.txt
"""

"""时空动作检测
python demo/demo_spatiotemporal_det.py demo/demo.mp4 demo_out/demo_spatiotemporal_det.mp4 \
    --config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --checkpoint checkpoints/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map tools/data/ava/label_map.txt \
    --predict-stepsize 8 \
    --output-stepsize 4 \
    --output-fps 6
    
python tools/deployment/export_onnx_stdet.py \
    configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    checkpoints/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --output_file slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.onnx \
    --num_frames 8
    
python demo/demo_spatiotemporal_det_onnx.py demo/demo.mp4 demo_out/demo_spatiotemporal_det_onnx.mp4 \
    --config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --onnx-file checkpoints/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.onnx \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --det-score-thr 0.9 \
    --action-score-thr 0.5 \
    --label-map tools/data/ava/label_map.txt \
    --predict-stepsize 8 \
    --output-stepsize 4 \
    --output-fps 6
"""

"""Inferencer
python demo/demo_inferencer.py demo/demo.mp4 \
    --rec tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb \
    --rec-weights checkpoints/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    --label-file tools/data/kinetics/label_map_k400.txt \
    --vid-out-dir demo_out
"""

""" 音频演示
python demo/demo_audio.py \
    configs/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature.py \
    https://download.openmmlab.com/mmaction/v1.0/recognition_audio/resnet/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature/tsn_r18_8xb320-64x1x1-100e_kinetics400-audio-feature_20230702-e4642fb0.pth \
    audio_feature.npy tools/data/kinetics/label_map_k400.txt
"""

"""使用单个视频预测基于骨架和基于 RGB 的动作识别和时空动作检测结果。
python demo/demo_video_structuralize.py \
    --skeleton-stdet-checkpoint checkpoints/posec3d_ava.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --skeleton-config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --skeleton-checkpoint checkpoints/posec3d_k400.pth \
    --use-skeleton-stdet \
    --use-skeleton-recog \
    --label-map-stdet tools/data/ava/label_map.txt \
    --label-map tools/data/kinetics/label_map_k400.txt \
    --out-filename demo_out/test_stdet_recognition_output.mp4
    
Faster RCNN 检测视频中的人。
HRNet 估计这些检测到的人的姿态，生成骨架数据。
posec3d_ava 使用这些骨架数据进行动作识别(stand/walk)。
posec3d_k400 进一步在时空域中检测动作，确保动作识别的连续性和精确性。

python demo/demo_video_structuralize.py \
    --rgb-stdet-config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --rgb-stdet-checkpoint checkpoints/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --rgb-config demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
    --rgb-checkpoint checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
    --label-map-stdet tools/data/ava/label_map.txt \
    --label-map tools/data/kinetics/label_map_k400.txt \
    --out-filename demo_out/test_stdet_recognition_output2.mp4
未做骨架估计

python demo/demo_video_structuralize.py \
    --rgb-stdet-config configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py \
    --rgb-stdet-checkpoint  checkpoints/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --skeleton-config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --skeleton-checkpoint checkpoints/posec3d_k400.pth \
    --use-skeleton-recog \
    --label-map-stdet tools/data/ava/label_map.txt \
    --label-map tools/data/kinetics/label_map_k400.txt \
    --out-filename demo_out/test_stdet_recognition_output3.mp4

Faster RCNN - 人体检测器
HRNetw32 - 姿态估计器
PoseC3D - 基于骨架的动作识别器
SlowOnly-8x8-R101 - RGB 基于RGB的时空动作检测。

python demo/demo_video_structuralize.py \
    --skeleton-stdet-checkpoint checkpoints/posec3d_ava.pth \
    --det-config demo/demo_configs/faster-rcnn_r50_fpn_2x_coco_infer.py \
    --det-checkpoint checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    --pose-config demo/demo_configs/td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py \
    --pose-checkpoint checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth \
    --skeleton-config configs/skeleton/posec3d/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py \
    --rgb-config demo/demo_configs/tsn_r50_1x1x8_video_infer.py \
    --rgb-checkpoint checkpoints/tsn_r50_1x1x3_100e_kinetics400_rgb_20200614-e508be42.pth \
    --use-skeleton-stdet \
    --label-map-stdet tools/data/ava/label_map.txt \
    --label-map tools/data/kinetics/label_map_k400.txt \
    --out-filename demo_out/test_stdet_recognition_output4.mp4
"""


"""mmaction2 tutorial
# MMAction2 Colab: https://colab.research.google.com/github/open-mmlab/mmaction2/blob/master/demo/mmaction2_tutorial.ipynb
# kinetics400_tiny 数据集
scp -r -P 5322 kinetics400_tiny.zip root@10.8.5.43:/root/project/research/action/mmaction2/data
├── kinetics400_tiny
│   ├── kinetics_tiny_train_video.txt
│   ├── kinetics_tiny_val_video.txt
│   ├── train
│   └── val

完成训练测试验证
"""
```

失败遗留相关：

```python
# I3D
# 数据集：https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md
# https://openxlab.org.cn/home
# 数据集hmdb51 https://mmaction2.readthedocs.io/zh-cn/latest/dataset_zoo/hmdb51.html
"""失败
scp -r -P 5322 download_annotations.sh root@10.8.5.43:/root/project/research/action/mmaction2/tools/data/hmdb51
scp -r -P 5322 download_videos.sh root@10.8.5.43:/root/project/research/action/mmaction2/tools/data/hmdb51
bash download_annotations.sh
bash download_videos.sh
scp -r -P 5322 extract_rgb_frames.sh root@10.8.5.43:/root/project/research/action/mmaction2/tools/data/hmdb51
bash extract_rgb_frames.sh
"""

行为识别模型包括TSN/TSM/I3D/R(2+1)D/Slow/SlowFast
时序行为检测模型包括BMN/BSN
```

## 2、mmaction2

### 前置任务：

#### Action Recognition（动作识别）：

动作识别的目标是在给定的视频片段中识别出特定的动作类别。动作识别主要关心视频的全局特征，并对其进行分类。

例如，在短视频中识别出某人正在做“跑步”、“跳舞”或“打篮球”等动作。

输入数据：RGB 视频帧，光流（Optical Flow）数据用于捕捉运动信息

防范：TSN，I3D, TSM, SlowFast等

#### Action Localization（动作定位）：

动作定位的任务是在长视频中定位动作发生的**时间段**。输出结果通常是每个动作发生的开始时间和结束时间，以及该动作的类别标签。

**输入数据**：RGB 视频帧；光流；关键帧；

方法：BMN等

#### Spatio-Temporal Action Detection（时空动作检测）：

时空动作检测的任务不仅要定位视频中动作发生的时间段，还要检测动作在每一帧中的**空间位置**（通常用边界框表示）。输出结果包括动作的时间区间、类别标签，以及视频中每一帧人物的空间位置（Bounding Box）。

**输入数据**：RGB 视频帧；光流；关键帧；对象检测框（Bounding Boxes）；

方法：AVA（Atomic Visual Actions）模型；SlowFast+Fast R-CNN；

#### Skeleton-based Action Recognition（基于骨骼的动作识别）：

基于骨骼的动作识别任务是通过分析人体关键点（通常是人体的关节或骨骼点）的运动轨迹来识别动作类型。这种方法特别适合用于人体姿态分析、健身指导等应用场景。

输入数据：2D 或 3D 人体骨骼关键点数据（通常是通过 OpenPose 或 MediaPipe 等工具获取）；

方法：ST-GCN等

#### Video Retrieval（视频检索）：

视频检索的任务是从一个视频数据库中找到与查询视频（query video）最相似的视频。通常用于基于内容的视频检索、相似场景或动作的查找等。

方法：基于 CNN+LSTM 的视频检索方法； 基于 Transformer 的检索模型

```python
scp -r -P 5322 UCF101 root@10.8.5.43:/root/project/research/action/mmaction2/data
# 1、下载解压：参考 https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ucf101/README_zh-CN.md
# /root/project/research/action/mmaction2/data/UCF101
├── annotations
│   └── ucfTrainTestlist
│           ├── classInd.txt
│           ├── testlist01.txt
│           ├── testlist02.txt
│           ├── testlist03.txt
│           ├── trainlist01.txt
│           ├── trainlist02.txt
│           └── trainlist03.txt
└── videos
    ├── ApplyEyeMakeup
    ├── ApplyLipstick
    ├── Archery
    ├── xxx
    ...
    ├── WritingOnBoard
    └── YoYo
# 2、使用 Python 脚本生成 UCF101 数据集的视频文件列表
# python tools/data/build_file_list.py ucf101 data/ucf101/videos/ --level 2 --format videos --shuffle   
├── annotations
│   ├── classInd.txt
│   ├── testlist01.txt
│   ├── testlist02.txt
│   ├── testlist03.txt
│   ├── trainlist01.txt
│   ├── trainlist02.txt
│   └── trainlist03.txt
├── ucf101_train_split_1_videos.txt
├── ucf101_train_split_2_videos.txt
├── ucf101_train_split_3_videos.txt
├── ucf101_val_split_1_videos.txt
├── ucf101_val_split_2_videos.txt
├── ucf101_val_split_3_videos.txt
└── videos
    ├── ApplyEyeMakeup
    ├── ApplyLipstick
    ├── Archery
	...
    ├── WallPushups
    ├── WritingOnBoard
    └── YoYo
    
# 3、抽取视频帧和光流
# 在视频动作识别任务中，抽取视频帧和光流是一种常见的预处理步骤，主要用于从视频中提取时空信息，以更好地捕捉动作和行为的特征
# 如果用户需要抽取 RGB 帧（因为抽取光流的过程十分耗时），可以考虑运行以下命令使用 denseflow 只抽取 RGB 帧。
# -------------------- 安装denseflow ---------------------------------------
"""
# 1、安装cmake
scp -r -P 5322 cmake-3.27.0-linux-x86_64.sh root@10.8.5.43:/root/project/research/action/
# 2、安装opencv，参考《WDCV库架构》
scp -r -P 5322 opencv-4.6.0.zip root@10.8.5.43:/root/project/research/action/
scp -r -P 5322 opencv_contrib-4.6.0.zip root@10.8.5.43:/root/project/research/action/
scp -r -P 5322 ippicv_2020_lnx_intel64_20191018_general.tgz root@10.8.5.43:/root/project/research/action/

cmake -DOPENCV_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/opencv/ -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.6.0/modules .. -D IPPICV_DISABLE=ON

# 需要安装CUDA版本的opecnv
# 卸载原有的
# 查找
ls /usr/local/lib | grep opencv
ls /usr/local/include | grep opencv
# 卸载，如果在安装过程中使用了 make install，可以使用 make uninstall 来卸载，但这要求之前构建时生成了卸载规则
rm -rf /usr/local/include/opencv4
rm -rf /usr/local/lib/libopencv*
rm -rf /usr/local/bin/opencv*

cmake -DOPENCV_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/opencv/ \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.6.0/modules \
      -D IPPICV_DISABLE=ON \
      -D WITH_CUDA=ON \                   # 启用 CUDA 支持
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \  # CUDA 工具包路径
      -D OPENCV_DNN_CUDA=ON \             # 启用 DNN 模块的 CUDA 支持（如果需要）
      -D CUDA_ARCH_BIN=6.1;6.2;7.0;7.5;8.0 \ # 指定 CUDA 架构（根据你的 GPU 选择合适的架构）
      ..

cmake -DOPENCV_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/opencv/ \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.6.0/modules \
      -D IPPICV_DISABLE=ON \
      -D WITH_CUDA=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      -D OPENCV_DNN_CUDA=ON \
      ..
make -j8
make install

# 安装denseflow

git clone https://github.com/open-mmlab/denseflow.git
cd denseflow && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$HOME/app -DUSE_HDF5=no -DUSE_NVFLOW=no ..
make -j
make install

"""
python tools/data/build_rawframes.py data/ucf101/videos/ data/ucf101/rawframes/ --task rgb --level 2  --ext avi
"""报错 denseflow: error while loading shared libraries: libopencv_cudaoptflow.so.406: cannot open shared object file: No such file or directory
find /usr/local/ -name "libopencv_cudaoptflow*"
/usr/local/lib/libopencv_cudaoptflow.so.406
/usr/local/lib/libopencv_cudaoptflow.so.4.6.0
/usr/local/lib/libopencv_cudaoptflow.so

解决：临时设置  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
永久添加： 
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
"""

```

### 2-1、数据集

#### 1、 kinetics400

#### 2、UCF101

#### 3、Something-Something V2



### 2-2、动作识别-UCF101 Finetune

```python
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()
```

对模型进行微调训练，参考：

```
# mmaction2 训练参考 https://github.com/open-mmlab/mmaction2/blob/main/docs/zh_cn/user_guides/train_test.md
# tsn训练参考：
# csdn blog 相关任务理解参考：https://blog.csdn.net/weixin_61674495/article/details/125928095
# 训练的config配置文件命名含义参考：https://mmaction2.readthedocs.io/zh-cn/latest/user_guides/config.html
# 模型微调：https://mmaction2.readthedocs.io/zh-cn/latest/user_guides/finetune.html
# 数据集准备：https://mmaction2.readthedocs.io/zh-cn/latest/user_guides/prepare_dataset.html
```

#### 1、TSN

“Temporal segment networks: Towards good practices for deep action recognition， 时间段网络：走向深度动作识别的良好实践”

Abstract: 

```
深度卷积网络在静态图像的视觉识别方面取得了巨大的成功。然而，对于视频中的动作识别，相对于传统方法的优势并不那么明显。本文旨在探索设计用于视频动作识别的有效 ConvNet 架构的原理，并在给定有限训练样本的情况下学习这些模型。我们的第一个贡献是时间段网络（TSN），这是一种基于视频的动作识别的新颖框架。它基于远程时间结构建模的思想。它结合了稀疏时间采样策略和视频级监督，可以使用整个动作视频进行高效且有效的学习。另一个贡献是我们对在时间段网络的帮助下在视频数据上学习卷积网络的一系列良好实践的研究。我们的方法在 HMDB51 (69.4%) 和 UCF101 (94.2%) 数据集上获得了最先进的性能。我们还可视化了学习到的 ConvNet 模型，它定性地证明了时间段网络的有效性和所提出的良好实践。
```

使用了 `Recognizer2D` 作为**基础类型**，backbone选择了 `ResNet`，head 选择了 `TSNHead`:

- backbone:resnet
- head: 做动作识别；输入的特征图尺寸其实是 N * num_segs，即包括了 batch size 以及一个clip中的T帧图片。TSN 做的工作就是对每个clip的 num_segs 帧结果取平均，得到最终结果。做的工作就是 N * num_segs, in_channels, h, w 经过reshape与avg pool得到 N, inchannels 的特征，然后通过一个全连接层进行分类得到最终结果。如果有必要的话，再加上一个dropout。

复现：

```python
# 增加 tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_ucf101-rgb.py config文件

# imagenet-pretrained-r50 表示使用加载预训练的resnet50; 
# 8xb32表示num_workers=8, bs=32, 
# 1x1x3表示clip_len=1,frame_interval=1,num_clips=3
# 100e 表示 max_epochs
# kinetics400 代表数据集，  数据模态，例如 rgb、flow、keypoint-2d 等

# 另外修改 tsn_r50.py中的num_classes=101

# python tools/train.py ${CONFIG_FILE} [ARGS]
# ${CONFIG_FILE}: 配置文件的路径 [ARGS]: 可选的命令行参数

python tools/train.py configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_ucf101-rgb.py --work-dir ./work_dirs/ucf101_tsn

"""result
10/21 10:53:37 - mmengine - INFO - Epoch(val) [100][119/119]    acc/top1: 0.8163  acc/top5: 0.9543  acc/mean1: 0.8138  data_time: 0.0655  time: 0.1138
"""
```

#### 2、SlowFast

“SlowFast Networks for Video Recognition”， 用于视频识别网络的SlowFast

Abstract: 

```
我们提出了用于视频识别的 SlowFast 网络。我们的模型涉及（i）慢速路径，以低帧速率运行，以捕获空间语义，以及（ii）快速路径，以高帧速率运行，以精细时间分辨率捕获运动。通过减少其通道容量，快速路径可以变得非常轻量级，但可以学习用于视频识别的有用时间信息。我们的模型在视频中的动作分类和检测方面都实现了强大的性能，并且我们的 SlowFast 概念的贡献明确指出了巨大的改进。我们报告了主要视频识别基准、Kinetics、Charades 和 AVA 的最先进的准确性。
```

**模型结构:**

- **双路径网络**:
  - **Slow Pathway**（慢通道）：用于提取高分辨率（通常是原始帧率）的特征。这一部分负责捕捉视频的细节信息，通常使用较慢的帧率（例如每秒提取 4 帧）。  ----- 使用较低的帧率（如每秒提取 4 帧）来提取高分辨率图像的特征。这可以通过选择视频中均匀间隔的帧来实现。例如，从每个视频的开头、中心和末尾各提取 1 帧。使用卷积神经网络（CNN）模型（例如 ResNet、DenseNet 等）来提取特征。
  - **Fast Pathway**（快通道）：用于提取低分辨率（通常是较高帧率）的特征，以捕捉快速运动信息。快通道通常以较快的帧率提取（例如每秒提取 32 帧）。   ----- 使用较高的帧率（如每秒提取 32 帧）来捕捉快速运动的信息。您可以通过每秒提取固定数量的帧（如每 1/30 秒提取 1 帧）来实现。同样使用 CNN 模型提取这些帧的特征。
- **融合层**：在模型的后期，通过融合慢通道和快通道的特征来获取全面的信息，最终用于动作分类。

```python
# 增加 slowfast_r50_8xb8-4x16x1-256e_ucf101-rgb.py
# 修改 slowfast_r50.py num_classes=101
# 启用 --deterministic 会报错
python tools/train.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_ucf101-rgb.py --work-dir ./work_dirs/ucf101_slowfast --seed=0

"""result

"""
```

#### 3、[MultiModality: Audio](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition_audio/resnet/README.md) (ArXiv'2020)

多模态：“Audiovisual SlowFast Networks for Video Recognition”，用于视频识别的视听 SlowFast 网络

Abstract：

```
我们提出视听SlowFast网络，这是一种集成视听感知的架构。 AVSlowFast 具有慢速和快速视觉路径，它们与更快的音频路径深度集成，以统一的表示方式对视觉和声音进行建模。我们在多个层面融合音频和视觉特征，使音频有助于形成分层视听概念。为了克服因音频和视觉模式的不同学习动态而产生的训练困难，我们引入了 DropPathway，它在训练期间随机删除音频路径，作为一种有效的正则化技术。受先前神经科学研究的启发，我们执行分层视听同步来学习联合视听特征。我们报告了六个视频动作分类和检测数据集的最新结果，进行了详细的消融研究，并展示了 AVSlowFast 的泛化，以学习自监督视听特征。
```

#### 4、[TimeSformer](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/timesformer/README.md) (ICML'2021)

“Is Space-Time Attention All You Need for Video Understanding?”， 时空注意力是视频理解所需要的全部吗？

Abstract：

```
我们提出了一种无卷积的视频分类方法，专门建立在空间和时间上的自注意力基础上。我们的方法名为“TimeSformer”，通过直接从帧级补丁序列进行时空特征学习，将标准 Transformer 架构应用于视频。我们的实验研究比较了不同的自注意力方案，并表明“分散注意力”（即时间注意力和空间注意力分别应用在每个块内）可以在所考虑的设计选择中获得最佳的视频分类准确性。尽管采用了全新的设计，TimeSformer 在多个动作识别基准测试中仍取得了最先进的结果，包括 Kinetics-400 和 Kinetics-600 上报告的最佳准确度。最后，与 3D 卷积网络相比，我们的模型训练速度更快，可以实现显着更高的测试效率（精度略有下降），并且还可以应用于更长的视频剪辑（超过一分钟长）。
```

#### 5、[ActionCLIP](https://github.com/open-mmlab/mmaction2/blob/main/projects/actionclip/README.md) (ArXiv'2021)

“ActionCLIP: A New Paradigm for Video Action Recognition”， 视频动作识别的新范式

Abstract：

```
视频动作识别的规范方法要求神经模型执行经典且标准的 N 中 1 多数投票任务。它们被训练来预测一组固定的预定义类别，从而限制了它们在具有未见过的概念的新数据集上的可转移能力。在本文中，我们通过重视标签文本的语义信息而不是简单地将它们映射为数字，为动作识别提供了新的视角。具体来说，我们将此任务建模为多模态学习框架内的视频文本匹配问题，这通过更多语义语言监督增强了视频表示，并使我们的模型能够进行零样本动作识别，而无需任何进一步的标记数据或参数要求。此外，为了解决标签文本的不足并利用大量网络数据，我们提出了一种基于多模态学习框架的动作识别新范式，我们将其称为“预训练、提示和微调”。该范例首先从大量网络图像文本或视频文本数据的预训练中学习强大的表示。然后，它通过即时工程使动作识别任务更像预训练问题。最后，它对目标数据集进行端到端微调以获得强大的性能。我们给出了新范式 ActionCLIP 的实例，它不仅具有优越且灵活的零样本/少样本迁移能力，而且在一般动作识别任务上达到了顶级性能，在 Kinetics-400 上实现了 83.8% 的 top-1 准确率以ViT-B/16为骨干。
```

#### 6、[VideoSwin](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/swin/README.md) (CVPR'2022)

“ Video Swin Transformer ”

Abstract：

```
视觉社区正在见证从 CNN 到 Transformer 的建模转变，其中纯 Transformer 架构在主要视频识别基准上已达到最高准确度。这些视频模型都建立在 Transformer 层上，这些层在空间和时间维度上全局连接补丁。在本文中，我们提倡视频 Transformer 中局部性的归纳偏差，与之前使用时空分解全局计算自注意力的方法相比，这可以实现更好的速度与准确度权衡。所提出的视频架构的局部性是通过采用为图像域设计的 Swin Transformer 来实现的，同时继续利用预训练图像模型的强大功能。我们的方法在广泛的视频识别基准测试中实现了最先进的准确度，包括动作识别（Kinetics-400 上的 84.9 top-1 准确度和 Kinetics-600 上 85.9 top-1 的准确度，经过约 20 倍的预训练数据和约 3 倍小的模型大小）和时间建模（Something-Something v2 上的准确度为 69.6 top-1）
```

#### 7、[VideoMAE](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomae/README.md) (NeurIPS'2022)

“VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training“， 屏蔽自动编码器是用于自监督视频预训练的数据高效学习器

Abstract：

```
通常需要在超大规模数据集上预训练视频转换器，才能在相对较小的数据集上实现最佳性能。在本文中，我们证明视频屏蔽自动编码器（VideoMAE）是用于自监督视频预训练（SSVP）的数据高效学习器。我们受到最近的 ImageMAE 的启发，提出了具有极高比率的定制视频管掩蔽。这种简单的设计使视频重建成为一项更具挑战性的自我监督任务，从而鼓励在预训练过程中提取更有效的视频表示。我们在 SSVP 上获得了三个重要发现：（1）极高比例的掩蔽比（即 90% 到 95%）仍然可以产生良好的 VideoMAE 性能。时间冗余的视频内容能够实现比图像更高的掩蔽比。 (2) VideoMAE 在非常小的数据集（即大约 3k-4k 视频）上取得了令人印象深刻的结果，而无需使用任何额外的数据。 (3)VideoMAE表明对于SSVP来说数据质量比数据数量更重要。预训练和目标数据集之间的域转移是一个重要问题。值得注意的是，我们的 VideoMAE 与 vanilla ViT 可以在 Kinetics-400 上达到 87.4%，在 Something-Something V2 上达到 75.4%，在 UCF101 上达到 91.3%，在 HMDB51 上达到 62.6%，而无需使用任何额外的数据。
```

#### 8、[MViT V2](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/mvit/README.md) (CVPR'2022)

”MViTv2: Improved Multiscale Vision Transformers for Classification and Detection“， 改进的多尺度ViT，用于分类和检测.

Abstract：

```
在本文中，我们研究多尺度视觉变换器（MViTv2）作为图像和视频分类以及对象检测的统一架构。我们提出了 MViT 的改进版本，它结合了分解的相对位置嵌入和残差池连接。我们以五种尺寸实例化该架构，并对其 ImageNet 分类、​​COCO 检测和动力学视频识别进行评估，其性能优于之前的工作。我们进一步将 MViTv2s 的池化注意力与窗口注意力机制进行比较，它在准确性/计算方面优于后者。没有任何附加功能，MViTv2 在 3 个领域具有最先进的性能：ImageNet 分类准确率为 88.8%，COCO 目标检测准确率为 58.7 boxAP，Kinetics-400 视频分类准确率为 86.1%。
```

#### 9、[UniFormer V1](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformer/README.md) (ICLR'2022)

”UniFormer: Unified Transformer for Efficient Spatiotemporal Representation Learning“，用于高效时空表示学习的统一变压器

Abstract：

```
由于视频帧之间存在大量的局部冗余和复杂的全局依赖性，从高维视频中学习丰富的多尺度时空语义是一项具有挑战性的任务。这项研究的最新进展主要是由 3D 卷积神经网络和视觉转换器推动的。尽管 3D 卷积可以有效地聚合局部上下文以抑制小型 3D 邻域的局部冗余，但由于感受野有限，它缺乏捕获全局依赖性的能力。或者，视觉变换器可以通过自注意力机制有效地捕获长程依赖性，但在通过每层所有令牌之间的盲目相似性比较来减少局部冗余方面存在局限性。基于这些观察，我们提出了一种新颖的统一变换器（UniFormer），它以简洁的变换器格式无缝集成了 3D 卷积和时空自注意力的优点，并在计算和精度之间实现了更好的平衡。与传统的转换器不同，我们的关系聚合器可以通过分别在浅层和深层学习局部和全局令牌亲和力来解决时空冗余和依赖性。我们对流行的视频基准进行了广泛的实验，例如 Kinetics-400、Kinetics-600 和 Something-Something V1&V2。仅通过 ImageNet-1K 预训练，我们的 UniFormer 在 Kinetics-400/Kinetics-600 上即可实现 82.9%/84.8% 的 top-1 准确率，同时所需的 GFLOP 比其他最先进的方法少 10 倍。对于 Something-Something V1 和 V2，我们的 UniFormer 分别实现了 60.9% 和 71.2% 的 top-1 准确率。
```

#### 10、[UniFormer V2](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/uniformerv2/README.md) (Arxiv'2022)

”UniFormerV2: Spatiotemporal Learning by Arming Image ViTs with Video UniFormer”，通过使用视频 UniFormer 武装图像 ViT 进行时空学习

Abstract：

```
学习判别时空表示是视频理解的关键问题。最近，视觉变换器（ViTs）展示了它们在通过自注意力学习长期视频依赖性方面的能力。不幸的是，由于令牌之间盲目的全局比较，它们在解决本地视频冗余方面表现出局限性。 UniFormer 通过将卷积和自注意力统一为 Transformer 格式的关系聚合器，成功缓解了这个问题。然而，在对视频进行微调之前，该模型必须需要烦人且复杂的图像预训练短语。这阻碍了它在实践中的广泛使用。相反，开源 ViT 很容易获得，并且经过良好的预训练，具有丰富的图像监督功能。基于这些观察，我们提出了一个通用范例，通过使用高效的 UniFormer 设计来武装预训练的 ViT，构建强大的视频网络系列。我们称这个系列为 UniFormerV2，因为它继承了 UniFormer 块的简洁风格。但它包含全新的本地和全局关系聚合器，通过无缝集成 ViT 和 UniFormer 的优势，实现更好的精度与计算平衡。没有任何附加功能，我们的 UniFormerV2 在 8 个流行视频基准测试中获得了最先进的识别性能，包括场景相关的 Kinetics-400/600/700 和 Moments in Time、时间相关的 Something-Something V1/V2 、未修剪的 ActivityNet 和 HACS。特别是，据我们所知，它是第一个在 Kinetics-400 上达到 90% top-1 准确率的模型。
```

#### 11、[VideoMAE V2](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/videomaev2/README.md) (CVPR'2023)

“VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking”， 使用双掩蔽缩放视频掩蔽自动编码器

Abstract：

```
规模是构建强大的基础模型的主要因素，该模型可以很好地推广到各种下游任务。然而，训练具有数十亿参数的视频基础模型仍然具有挑战性。本文表明，视频掩码自动编码器（VideoMAE）是一种可扩展的通用自监督预训练器，用于构建视频基础模型。我们通过核心设计在模型和数据方面扩展了 VideoMAE。具体来说，我们提出了一种用于高效预训练的双重掩蔽策略，其中编码器对视频令牌的子集进行操作，解码器对视频令牌的另一个子集进行处理。尽管VideoMAE由于编码器中的高掩蔽比而非常高效，但掩蔽解码器仍然可以进一步降低总体计算成本。这使得视频中十亿级模型的高效预训练成为可能。我们还使用渐进式训练范例，其中涉及对不同的多源未标记数据集进行初始预训练，然后对混合标记数据集进行后预训练。最后，我们成功训练了具有十亿个参数的视频 ViT 模型，该模型在 Kinetics（K400 上为 90.0%，K600 上为 89.9%）和 Something-Something（K400 上为 68.7%）数据集上实现了新的最先进性能。 V1 和 V2 为 77.0%）。此外，我们在各种下游任务上广泛验证了预训练的视频 ViT 模型，证明了其作为通用视频表示学习器的有效性。
```



## 3、notebooks



## 4、源码分析：

```
clip_len=1,  # 每个采样输出剪辑的帧数
frame_interval=1,  # 相邻采样帧的时间间隔
num_clips=3),  # 要采样的剪辑数
```

### 4-1、使用 Registry 机制搭建模型：

模型的每个组件（例如 `Backbone`、`Neck`、`Head`）会通过 `Registry` 注册，之后可以在配置文件中通过字符串指定模型的类型和对应参数。

```python
from .base import AvgConsensus, BaseHead

# HEADS.register_module() 是一个装饰器，表示 TSNHead 类被注册到 HEADS 注册表中。这样，用户可以在配置文件中通过字符串 "TSNHead" 来引用该组件，而框架会自动调用它的实现。
@MODELS.register_module()
class TSNHead(BaseHead):
    pass
```

Registry 的实现原理:  `MMEngine` 中的 `Registry` 实现了一个简单而强大的模块注册和查找机制。主要功能如下：

- **注册模块**：可以使用装饰器 `@Registry.register_module()` 将模块类注册到对应的 `Registry` 实例中。
- **查找模块**：当需要实例化某个模块时，只需提供模块名（如 `TSNHead`），`Registry` 会自动查找对应的类并实例化。

Registry 机制的优势:

- **模块化**：`Registry` 机制使得模型的各个组件解耦，便于扩展和修改。
- **灵活性**：通过配置文件可以轻松更换不同的组件，比如更换 `Backbone` 或 `Head`。
- **易于扩展**：用户可以自定义自己的模型组件，并通过 `Registry` 进行注册和使用。



# 二、video understanding(视频理解)：PySlowFast

resource：

```python
# paperwithcode: https://paperswithcode.com/sota/action-recognition-on-ava-v2-2 
# STAR/L: https://paperswithcode.com/paper/end-to-end-spatio-temporal-action
# 动作识别：slowfast  https://github.com/facebookresearch/SlowFast
```

**PySlowFast**  是 FAIR 的开源视频理解代码库，提供最先进的视频分类模型和高效的训练。

PySlowFast 的目标是提供高性能、轻量级的 pytorch 代码库，为不同任务（分类、检测等）的视频理解研究提供最先进的视频主干。它的设计目的是为了支持新颖的视频研究想法的快速实施和评估。 PySlowFast 包括以下骨干网络架构的实现：

- SlowFast
- Slow
- C2D
- I3D
- Non-local Network
- X3D
- MViTv1 and MViTv2
- Rev-ViT and Rev-MViT





# 三、人体关键点检测：

https://github.com/CMU-Perceptual-Computing-Lab/openpose



# other:

一、train, 训练更新：

1、权重（Weights）与偏置（Biases）：卷积层、全连接层等的权重和偏置；

2、**批归一化参数（Batch Normalization Parameters）**：

（1）**均值（Mean）和方差（Variance）**：在批归一化层中，用于标准化输入的均值和方差在训练阶段会被更新。它们是从当前批次的输入计算而来的。

（2）**可学习参数（Gamma 和 Beta）**：

- **Gamma（缩放因子）**：用于对归一化后的输出进行缩放，通常是一个可学习的参数。
- **Beta（偏移量）**：用于对归一化后的输出进行偏移，也是一个可学习的参数。



二、`backbone` 配置中的 `norm_eval` 参数：

**`norm_eval=True`**：表示在训练时，**Batch Normalization** 层会被设置为评估模式（evaluation mode），即在前向传播时，BN 层会使用预训练模型中的均值和方差，而不再根据当前批次数据动态更新均值和方差。这会冻结 BN 层的参数，使其不参与训练。   场景：预训练模型中的 BN 参数已经基于大量数据进行优化，重新训练时可能不需要再调整这些统计信息，特别是当新的训练数据与预训练数据的分布相似时，冻结 BN 层可以加速训练并保持模型的稳定性。

**`norm_eval=False`**：表示在训练时，BN 层会继续保持训练模式，动态更新每个批次数据的均值和方差，BN 层的均值和方差会随着训练数据变化不断更新。这通常适用于数据分布在训练过程中变化较大的情况。   场景：当训练数据与预训练数据的分布差异较大，或者在小数据集上训练时，使用 `norm_eval=False` 可能会更加有效。因为这种情况下，允许 BN 层动态更新可以帮助模型更好地适应新的数据分布。

BN层有两个部分：

（1）**可学习的参数**（权重与偏置，通常是 `gamma` 和 `beta`），这些参数会随着训练更新；

（2）**均值和方差**，这些是动态统计数据，用于归一化。

​	**`norm_eval=True`**，BN 层进入评估模式。BN 层不会再基于当前的训练数据更新均值和方差，而是使用在预训练时计算好的固定值。也就是说，这些**动态统计信息**（均值和方差）在整个训练过程中保持不变。BN 层的 `gamma` 和 `beta` 参数仍然可以是可学习的，即使 BN 处于评估模式，这些参数也可以继续更新。



三、requires_grad的含义：

用于指示一个张量是否需要计算梯度。其主要作用是在反向传播过程中决定是否对该张量的梯度进行跟踪。

冻结特征提取部分（backbone）:

requires_grad=False: **完全冻结特征提取部分**（包括所有权重，如卷积层、BN 层的 `gamma` 和 `beta` 参数）, 好处是可以利用预训练模型的特征提取能力，而不需要再进行更新，从而减少计算负担并避免过拟合。   冻结特征提取层的情况下，通常会只训练模型的后续部分（如分类器或其他自定义层）。这样可以保持特征提取层的能力，专注于学习新的任务。   迁移学习；减少过拟合；节省计算资源。



四、inference, 起作用的关键值包括：

1. **输入数据**：模型的原始输入。
2. **模型参数**: 推理时，模型使用在训练过程中学到的固定参数来对输入数据进行处理,

```
权重和偏置（Weights & Biases）：推理过程中最关键的值是模型各层的权重和偏置。这些参数决定了输入数据如何被转换，进而生成最终的输出。
	- 卷积层/全连接层的权重：用于计算输入数据的加权和。对于卷积层，权重是卷积核；对于全连接层，权重是矩阵乘法的系数。
	- 偏置：用于调整计算结果的偏移量，结合权重一起影响输出。
Batch Normalization 的 gamma 和 beta 参数（如果模型包含 BN 层）：
	- BN 层在推理时使用训练中学到的 gamma 和 beta 参数，用于对特征进行缩放和平移。

Batch Normalization 的均值和方差
	- 均值和方差（如果模型包含 BN 层）：推理时，BN 层不会再使用推理时批量数据的均值和方差，而是使用在训练阶段通过整个训练数据集计算出的全局均值和方差。它们用于对输入数据进行归一化处理，确保数据的分布与训练时一致。

```

   3.**激活值**：中间层输出的特征图。

```
激活值在训练阶段是动态变化的，因为它依赖于不断更新的权重和输入数据的变化。
在推理阶段，激活值取决于固定的权重和当前输入数据，因此它会根据输入数据的不同而变化，但不会因为模型参数的更新而改变。
```

  4.**模型结构**：各层的组合方式和连接关系。

```
网络层的配置：模型的结构（如卷积层、池化层、全连接层等）决定了输入如何在网络中传递和处理。虽然结构本身不变，但它在推理时起到决定性作用，影响输入数据流经各层的顺序和方式。
```

  5.**最终输出**：模型的预测结果，如分类的概率或回归的数值。

```
通过前面的层级计算，模型的最终输出通常是一个概率分布（如分类任务中的 softmax 输出）或者回归值（如回归任务中的数值输出）。这个结果是推理的主要输出，用于提供模型的预测结果。
```

6. **损失函数（仅在需要评估推理结果时使用）**
   - **推理时不直接参与计算**，但如果我们在推理过程中评估模型性能（如计算测试集上的准确率或误差），会使用损失函数来对推理的结果进行评估，衡量模型的预测效果。

  7.**注意力权重**：如果使用注意力机制，这些权重会影响输入如何被聚焦处理。

```
注意力机制模型（如 Transformer）在推理时，会根据输入数据计算注意力权重，用于决定输入序列中哪些部分对输出贡献较大。这些权重值会影响推理时的中间计算过程和最终结果。
```

五、模型大小：

**参数量**：模型中可训练的参数个数，包括权重、偏置等。参数越多，模型越复杂，也意味着模型对数据的表达能力越强，但也更容易过拟合。

Million (百万)； Billion (十亿)；

- 大模型：参数量超过 **100亿（Billion）** 个参数，这类模型需要大量计算资源（如GPU集群）进行训练，通常适用于大规模数据和高计算需求任务，如大型语言模型（LLM）、高级图像生成等任务。 如 复杂任务、跨领域任务（如GPT、BERT、DALL-E）。**GPT-3**：参数量**1750亿**（Billion）；BERT（Large）：**3.4亿**（340 Million），生成模型：DALL-E 2，约 **35亿**（3.5 Billion）；StyleGAN2，约 **3000万**（30 Million）。
- 中型模型：参数量介于 **几亿到几十亿** 个参数。常规的图像分类、目标检测、自然语言处理等任务。ViT-B/16：86 Million； BERT（Base）：**1.1亿**（110 Million）；Swin Transformer (Swin-T)，约**2900万**（29 Million）。**ResNet-50**：约**2500万**（25 Million）；
- 小模型：参数量在 **几百万到几亿** 个参数。这类模型通常用于低计算资源设备或实时应用，训练速度快，适合小型数据集和轻量化任务。MobileNet、SqueezeNet、EfficientNet-B0
