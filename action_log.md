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


# UCF101数据集
scp -r -P 5322 UCF101 root@10.8.5.43:/root/project/research/action/mmaction2/data

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
```

## 2、notebooks





# 二、动作识别：slowfast，i3d

https://github.com/facebookresearch/SlowFast
https://github.com/google-deepmind/kinetics-i3d

# 三、人体关键点检测：

https://github.com/CMU-Perceptual-Computing-Lab/openpose

