# Pose

# 一、MMPose

MMPose 是一款基于 Pytorch 的姿态估计开源工具箱，是 OpenMMLab 项目的成员之一，包含了丰富的 2D 多人姿态估计、2D 手部姿态估计、2D 人脸关键点检测、133关键点全身人体姿态估计、动物关键点检测、服饰关键点检测等算法以及相关的组件和模块，下面是它的整体框架。

```
https://github.com/open-mmlab/mmpose
# docs: https://mmpose.readthedocs.io/zh-cn/dev-1.x/guide_to_framework.html
```

新旧版本 mmpose、mmdet、mmcv 的对应关系为：

- mmdet 2.x <=> mmpose 0.x <=> mmcv 1.x
- mmdet 3.x <=> mmpose 1.x <=> mmcv 2.x

## Demo:

环境安装：

```python
"""已有 pip list | grep mm/torch
comm                              0.2.2
mmaction2                         1.2.0                /root/project/research/action/mmaction2
mmcv                              2.1.0
mmdet                             3.2.0
mmengine                          0.10.5
mmpose                            1.3.2python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file vis_results.jpg \
    --draw-heatmap
smmap                             5.0.1
timm                              0.6.13
open-clip-torch                   2.0.2
pytorch-ignite                    0.2.0
pytorch-lightning                 1.5.0
torch                             1.13.1+cu117
torchmetrics                      1.4.0.post0
torchvision                       0.14.1+cu1
"""
git clone https://github.com/open-mmlab/mmpose.git
scp -r -P 5322 mmpose.zip root@10.8.5.43:/root/project/research/action/
cd mmpose
pip install -r requirements.txt
pip install -v -e .
# "-v" 表示输出更多安装相关的信息
# "-e" 表示以可编辑形式安装，这样可以在不重新安装的情况下，让本地修改直接生效

# demo
# 下载 td-hm_hrnet-w48_8xb32-210e_coco-256x192 对应的配置文件与权重
# https://mmpose.readthedocs.io/en/latest/model_zoo_papers/backbones.html
# Topdown Heatmap + Hrnet on Coco
# scp -r -P 5322 td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth root@10.8.5.43:/root/project/research/action/mmpose
# lazy import 须 mim download mmpose --config td-hm_hrnet-w48_8xb32-210e_coco-256x192  --dest . 下载的td-hm_hrnet-w48_8xb32-210e_coco-256x192.py 解决
CUDA_VISIBLE_DEVICES=2 python demo/image_demo.py \
    tests/data/coco/000000000785.jpg \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
    td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth \
    --out-file vis_results.jpg \
    --draw-heatmap

# 2D 动物图片姿态识别推理
scp -r -P 5322 faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth root@10.8.5.43:/root/project/research/action/mmpose/checkpoints/

python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py \
    checkpoints/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth \
    configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py \
    checkpoints/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth \
    --input tests/data/animalpose/ca110.jpeg \
    --output-root vis_results --draw-heatmap --det-cat-id=15

# 2D 手部图片关键点识别
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    checkpoints/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input tests/data/onehand10k/9.jpg \
    --output-root vis_results --draw-heatmap
# 000000000110.jpg  /root/project/research/Yolo/ultralytics_yolov11/ultralytics/bus.jpg /root/dataset/action/hand-keypoints/val/images/IMG_00000004.jpg 
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    checkpoints/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input /root/dataset/action/hand-keypoints/val/images/IMG_00000004.jpg \
    --output-root vis_results --draw-heatmap

# 2D 手部视频关键点识别推理
scp -r -P 5322 tests_data_nvgesture_sk_color.avi root@10.8.5.43:/root/project/research/action/mmpose/tests/data

python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    checkpoints/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input tests/data/tests_data_nvgesture_sk_color.avi \
    --output-root vis_results --kpt-thr 0.1

# VID_20240905_104444.mp4
python demo/topdown_demo_with_mmdet.py \
    demo/mmdetection_cfg/rtmdet_nano_320-8xb32_hand.py \
    checkpoints/rtmdet_nano_8xb32-300e_hand-267f9c8f.pth \
    configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py \
    checkpoints/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth \
    --input /root/project/research/action/mmaction2/data/xian_video_dataset/pre/VID_20240905_104444.mp4 \
    --output-root vis_results --kpt-thr 0.1

# scp -r -P 5322 root@10.8.5.43:/root/project/research/action/mmpose/vis_results/VID_20240905_104444.mp4 ./result
# scp -r -P 5322 new_oral2 root@10.8.5.43:/root/dataset/code
# 训练
# coco_tiny  coco格式 标注数据转为 labelme 可展示的每张图片对应json格式文件
scp -r -P 5322 hrnet_w32-36af842e.pth root@10.8.5.43:/root/project/research/action/mmpose/checkpoints/
训练参考：01_mmpose_tutorial_train_inference.ipynb
```

## Docs:

```python
配置文件： https://mmpose.readthedocs.io/zh-cn/dev-1.x/user_guides/configs.html

推理及其相关参数： https://mmpose.readthedocs.io/zh-cn/dev-1.x/user_guides/inference.html

```



## Yolov11 train:

```python
# 1、yolov11 pose
# dataset : https://gitcode.csdn.net/65ec4f5f1a836825ed797cad.html

# coco8-pose train
scp -r -P 7322 coco8-pose root@10.8.5.43:/root/dataset/action
scp -r -P 7322 yolo11s-pose.pt root@10.8.5.43:/root/project/research/Yolo/ultralytics_yolov11/ultralytics
# test
python pose_test.py
# train check
python pose_train.py

# hand keypoints train
# inference 部署可参考：https://github.com/RionDsilvaCS/yolo-hand-pose
scp -r -P 7322 hand-keypoints.zip root@10.8.5.43:/root/dataset/action
python pose_hand_train.py
"""result
hand_YOLO11-pose summary (fused): 257 layers, 2,956,000 parameters, 0 gradients, 7.8 GFLOPs
Class     Images  Instances      Box(P      R      mAP50     mAP50-95)   Pose(P   R        mAP50  mAP50-95): 100%|██████████| 244/244 [00:37<00:00,  6.48it/s]
all       7808       7808      0.984      0.986      0.993     0.934      0.932   0.91      0.936  0.828
Speed: 0.2ms preprocess, 0.8ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to /root/project/research/Yolo/ultralytics_yolov11/ultralytics/runs/pose/train3
"""

# 2、mmpose
pose_log.md
# 3、OpenPose
pose_log.md
```

