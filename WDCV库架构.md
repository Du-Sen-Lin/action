# WDCV

## 一、期望：

### 1、通用AI模型标准化

- 标准化模型输入、输出
- 训练、推理、验证、配置接口保持一致

### 2、可视化分析

- 支持数据集可视化分析
- 支持模型评估
- 支持错图分析

### 3、最新AI算法demo集成

- 集成research algorithm, demo示例

### 4、tdcv集成

- 开发传统CV算法：C++

### 5、部署demo

- 支持 C++ 部署
- 指出 Python 部署

### 6、效果

- 集成常用AI模型，覆盖工业95%以上场景
- 快速分析数据、训练和部署



### 7、工业模型的迁移性、兼容性

- 工业场景分领域（玻璃/镜头/）的有监督大模型；
- 正样本场景的自主训练、快速部署平台；
- gtcv库快速开发
- yolov5集成

## 二、算法模型说明

### 1、classification 分类

- 参考 tsnn 与 https://github.com/Lightning-AI/lightning/blob/master/examples/pytorch/basics/backbone_image_classifier.py
- pytorch-lightning 由 1.2.10 -> 1.9.5；metries模块被移除
- 效果测试：

### 2、detection 目标检测



### 3、segmentation 分割



### 4、anomaly_detection 正样本缺陷检测



### 5、tdcv 传统图像处理算法



### 6、deploy





## 三、深度学习算法基本架构

### AI架构：

```
# 更新pytorch_lightning版本1.2.10 -> 1.9.5：
# pip install lightning==1.9.5 #暂时不需要安装
pip install pytorch-lightning==1.9.5
#安装lightning的时候自动更新，后期新镜像只更新pytorch-lightning查看效果; 对torch也更新torch==1.9.1+cu111 -> 1.13.1+cu117:
torch-1.13.1-cp37-cp37m-manylinux1_x86_64.whl 这个时候torch与torchvison版本不兼容，回退torch版本：pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html 
但是降低后与lightning版本不兼容：lightning 1.9.5 requires torch<4.0,>=1.10.0, but you have torch 1.9.1+cu111 which is incompatible.

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html 

# 使用 Autopep8 自动格式化Python代码（根据PEP 8风格指南）。它可以自动修复缩进、空格、括号、引号等方面的问题，使代码符合PEP 8规范。Autopep8的目标是保留原有代码的布局和结构，只进行必要的调整和修复
conda install --name cv_env autopep8 -y
```



```python
-gtcv
--ui
----basic.py  # 公用的工具函数
----bokeh_ui.py # bokeh相关的画图帮助函数
----pyplot_ui.py # pyplot相关的画图帮助函数
----image_cleaner.py # Jupyter中运行的数据集浏览、清洗工具

--utils
----basic.py #分类模型的基础工具函数

--classification
----transfer_learning.py # 迁移学习分类模型
----backbons ## 分类模型的骨干网络
------timm_backbons.py # 使用timm库，提供大量SOTA的预训练图像模型，且自动支持单通道或多通道图像输入
----heads ## 分类模型的头部网络
------base_head.py # 创建分类网络的头部模型

--common
----model_base.py # gtcv中的模型基础类，继承自 pl.LightningModule；使用pytorch-lightning 库，将研究的模型代码与工程代码分离，高可读性，更容易复现

--detection # 可参考 https://github.com/Iywie/pl_YOLO
----yolo.py # 效果太差，丢弃。 TimmBackboneWithFPN + YoloHead（yolov5）; 调整 anchors等，是有效果的
----yolox.py # TimmBackboneWithFPN + DecoupledHead， 效果可以。
----yolov5.py # TimmBackboneWithPAFPN + YoloHead 效果同yolo一样，较差。TimmBackboneWithPAFPN + DecoupledHead，效果较好。 ？？？？？？？待优化 + 一篇yolohead的文档 
----yolov7.py # to do 

--segmentation
----unet.py # https://github.com/hiepph/unet-lightning;https://github.com/Lightning-AI/lightning/blob/master/examples/pytorch/basics/autoencoder.py; https://catalog.ngc.nvidia.com/orgs/nvidia/resources/nnunet_for_pytorch
----u2net.py # https://zhuanlan.zhihu.com/p/470052977; https://github.com/xuebinqin/U-2-Net
----unet3plus.py # https://blog.csdn.net/Monkey_King_GL/article/details/127820469
----msrf.py #MSRF-Net gated conv（门控卷积，参考：https://www.ngui.cc/el/2580929.html?action=onClick） 代码参考：https://github.com/vikram71198/MSRF_Net  https://github.com/amlarraz/MSRF-Net_PyTorch
```

日志查看：tensorboard --logdir ./lightning_logs/

FPN：

```
FPN是自顶向下的，将高层特征通过上采样和低层特征做融合得到进行预测的特征图
```

![image-20230626143924417](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230626143924417.png)

FPN+PAN：

```
FPN层自顶向下传达强语义特征，而特征金字塔则自底向上传达强定位特征。
FPN 高维度向低维度传递语义信息（大目标更明确）,自上而下，上采样。
PAN 低维度向高维度再传递一次语义信息（小目标也更明确），自下而上，下采样。
深层的feature map携带有更强的语义特征，较弱的定位信息。而浅层的feature map携带有较强的位置信息，和较弱的语义特征。
FPN就是把深层的语义特征传到浅层，从而增强多个尺度上的语义表达。
而PAN则相反把浅层的定位信息传导到深层，增强多个尺度上的定位能力。
```

![image-20230627162117438](C:\Users\Admin\AppData\Roaming\Typora\typora-user-images\image-20230627162117438.png)



### AI架构核心依赖库版本迭代：

```python
# 先前版本
# pip install pytorch-lightning==1.9.5 
# pip install timm==0.6.13
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html 
# python --version 3.7.16

# -------------1、pytoch-lighting-------------------------
# torch / lighting / python 版本兼容性：https://lightning.ai/docs/pytorch/stable/versioning.html#compatibility-matrix
lightning.pytorch   pytorch_lightning   lightning.fabric   torch   torchmetrics   Python
2.0                     2.0            2.0 (GA)         ≥1.11, ≤2.0    ≥0.7.0     ≥3.8, ≤3.10
1.9                     1.9         1.9 (experimental)  ≥1.10, ≤1.13   ≥0.7.0     ≥3.7, ≤3.10
# docs https://lightning.ai/docs/pytorch/stable/
# 可修改一个版本，基础环境更改：python 3.10
# 可替代品


# -------------2、timm-------------------
# 旧版本是作者维护，新版本在hugging face下面，主要接口有小变化，模型下载默认地址变化，下载源比较难下载
# timm 现在在hugging face下，包含一千多个基础模型： timm 是一个包含 SOTA 计算机视觉模型、层、实用程序、优化器、调度器、数据加载器、增强和训练/评估脚本的库。 它包含超过 700 个预训练模型，设计灵活且易于使用。 阅读快速入门指南以启动并运行 timm 库。您将学习如何加载、发现和使用库中包含的预训练模型。
# git: https://github.com/huggingface/pytorch-image-models
# docs: https://huggingface.co/docs/timm/index  https://huggingface.co/docs/hub/timm
# blog: https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055


```







## 四、传统CV集成

### 1、集成方式：

#### 1-1、Cython

使用 Cython 将 Python 和 C++ 结合。Cython中将C++的函数包装为Python可调用的函数。

Cython:

- Cython是一个将Python代码转换为C代码的工具，它可以与C和C++代码无缝集成。
- Cython使用了自己的语法扩展，允许你在Python代码中添加类型声明和C/C++函数的定义。
- Cython编译后生成的扩展模块是纯C代码，可以与Python代码直接链接。
- Cython支持将C/C++代码直接嵌入到Python模块中，也支持使用Python C API进行交互。
- Cython提供了丰富的类型系统和高级特性，可以方便地处理C/C++类、对象和数据结构。

##### demo：编译成功

修改：函数命名inline / 或者 添加头文件

```C++
 // #ifndef DEMO_H 和 #define DEMO_H：这是条件编译的常用技巧，用于避免头文件被重复包含。DEMO_H是一个唯一的标识符，用于确保头文件只被包含一次。

// #ifdef __cplusplus 和 extern "C" {：这部分代码检查编译器是否为C++编译器。如果是C++编译器，它将用 extern "C" 声明将代码包裹起来，以便使用C链接和命名约定。

// #endif：结束了条件编译的代码块

// 使用这种方式，可以在C++代码中包含这个头文件，并与使用C语言编写的代码进行正确的链接和函数调用。这在涉及到混合编程或与其他语言进行交互的情况下非常有用。
```

```c++
#ifndef DEMO_H
#define DEMO_H

#ifdef __cplusplus
extern "C" {
#endif

int add_ab(int a, int b);

#ifdef __cplusplus
}
#endif

#endif
```



##### cython_caliper: 编译成功

```shell
python setup.py build_ext --inplace

gcc -pthread -B /root/conda/envs/ts_env/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/root/conda/envs/ts_env/lib/python3.7/site-packages/numpy/core/include -I/root/conda/envs/ts_env/include -I/root/conda/envs/ts_env/include/python3.7m -c c_caliper.c -o build/temp.linux-x86_64-cpython-37/c_caliper.o -O4 -march=native -mno-avx512f -fopenmp -Wall

g++ -pthread -B /root/conda/envs/ts_env/compiler_compat -Wl,--sysroot=/ -pthread -shared -B /root/conda/envs/ts_env/compiler_compat -L/root/conda/envs/ts_env/lib -Wl,-rpath=/root/conda/envs/ts_env/lib -Wl,--no-as-needed -Wl,--sysroot=/ build/temp.linux-x86_64-cpython-37/c_caliper.o build/temp.linux-x86_64-cpython-37/caliper.o -L./ -lopencv_features2d -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_dnn -o /root/project/wdcv/common/WDCV/wdcv/wdcv/cv/cython_caliper/c_caliper.cpython-37m-x86_64-linux-gnu.so

```

#### 1-2、pybind11

```
https://pybind11.readthedocs.io/en/stable/
https://github.com/pybind
```

pybind11:

- pybind11是一个用于创建Python扩展模块的轻量级库，它具有简单和明确的语法。
- pybind11使用现代C++特性，提供了更简洁的接口来绑定C++代码到Python。
- pybind11可以自动生成Python C API代码，免去了手动编写大量模板代码的麻烦。
- pybind11支持将C++类、函数和对象绑定到Python，并提供了对Python对象的直接访问。
- pybind11与C++代码的集成相对简单，可以在代码中以声明的方式定义Python接口。

##### example:

```
https://github.com/pybind/python_example.git

pip install pybind11

python setup.py install --record log.txt
```

```
# cmake 配置
https://blog.csdn.net/jingtaoaijinping/article/details/109111957
# 环境配置：
wget https://github.com/Kitware/CMake/releases/download/v3.23.2/cmake-3.23.2.tar.gz
tar -zxvf cmake-3.23.2.tar.gz
cd cmake-3.23.2
./bootstrap
make -j4     // 4线程
make install

# 编译：
cd build; 
cmake .. ; 
make ; 
cd ../bin ; 
./*** 
```



### 2、CV_Algo

#### 2-1、shape_based_matching

```markdown
**模板训练**
包含以下四个步骤：
- 模板训练：选取一个黄金模板图像，提取模板图像边缘的特征，对特征进行旋转，缩放然后构建多层金字塔
- 模板保存：将扩展后的特征存为模板文件
- 模板加载：从文件加载模板文件
- 模板匹配：对测试图像进行对应的金字塔缩放，对每层的金子塔图像，使用模板特征进行匹配，满足匹配阈值的放到匹配结果中
```

##### 1、shape_based_matching.py

```python
class ShapeBasedMatching:
    def __init__(self, features_num=128, T=(4,8),
                 weak_threshold=30.0, strong_threshold=60.0,
                 gaussion_kenel=7):
        """
        features_num：提取特征点数量（特征点越多，速度越慢）
        T:金字塔层数，下层是上一层图像的(w/2)*(h/2)，括号数值为金字塔后提取特征进行的梯度扩散区域大小，T越大，速度越快，精度越低，T越小，速度越慢，精度越高。
        weak_threshold 在金字塔提取特征和匹配的提取特征时剔除幅值小于设定值的点，
        strong_threshold 在创建模板特征时剔除赋值小于设置值的点    
        """
        self.c = CShapeBasedMatching(features_num=features_num, T=T, 
                                     weak_threshold=weak_threshold, 
                                     strong_threshold=strong_threshold, 
                                     gaussion_kenel=gaussion_kenel)   
	def add(self, img, mask=None, class_id='default', angle_range=None, scale_range=None, pad=100):
		"""
        根据角度(（结束角度e-起始角度s ） / 角度步长sp+1 = M)，缩放（ （结束缩放e -起始缩放s） / 缩放步长sp+1 = N），金字塔层数K , 生成M * N * K个特征模板。
        如本例程为 (（45 -（-45））/1+1) * (((1.4-1)/0.05)+1) * 1=819
        预估特征通过旋转和缩放可能超出图像区域，需要对四周填充区域pad进行调整（默认为100）		
		"""
        self.c.add(pad_img, pad_mask, class_id=class_id,
                   angle_range=angle_range,
                   scale_range=scale_range,
                   step_cb=step_callback)                
    def show(self, img, class_id='default', template_id=0, pad=100):
        """
        模板可视化
        """
        ret = self.c.show(to_show, class_id=class_id, template_id=template_id)

	def save(self, yaml_dir):
        """
        模板保存
        """
        self.c.save(yaml_dir)

	def load(self, yaml_dir, class_ids="default"):
        """
        加载模板
        """
        self.c.load(yaml_dir, class_ids)

    def find(self, img, score_threshold=90, iou_threshold=0.5, class_ids='default',
             pad=0, topk=-1, subpixel=False, debug=False):
        """
        模板匹配
        """
        matches_arr, matches_ids = self.c.find(pad_img, debug_img, threshold=score_threshold,
                                               iou_threshold=iou_threshold, class_ids=class_ids,
                                               topk=topk, subpixel=subpixel)
	def draw_match_rect(self, img, matches=None, color='b', thickness=3, pad=0, alpha=0.5):
        """
       匹配结果可视化
        """
        
```

##### 2、opencv: matchtemplate

opencv: matchtemplate https://blog.csdn.net/FriendshipTang/article/details/127971323



#### 2-2、caliper

```
！c++中std::cout<<输出 在终端脚本可看到，在notebook中不会输出。
！编译：python setup.py build_ext --inplace
```

caliper.h

```c++
/***
头文件保护（Header Guards）:这部分代码使用了预处理指令来创建头文件保护。这是一种防止头文件被重复包含的机制，防止了在同一个源文件中多次包含同一个头文件，避免了重复的定义和编译错误。
#ifndef CALIPER_ALGOH_H：这个指令检查是否没有定义过 CALIPER_ALGOH_H 这个宏。如果没有定义，表示这是第一次包含这个头文件，代码会继续往下执行。
#define CALIPER_ALGOH_H：这个指令定义了宏 CALIPER_ALGOH_H，以避免再次进入头文件的内容。
***/
#ifndef CALIPER_ALGOH_H
#define CALIPER_ALGOH_H
/***
C++兼容性：这部分代码在C++环境中执行。extern "C" 语句将在C++环境中执行，它告诉编译器这些函数应该按照C的函数命名和调用约定进行处理，以确保与C++的函数名称修饰规则保持一致。extern "C" { ... }：这部分用于设置C++兼容性，告诉编译器其中的代码应该按照C的方式进行链接
#ifdef __cplusplus：这是一个预处理指令，用于检查是否正在编译 C++ 代码。__cplusplus 是一个预定义的宏，在C++环境中会被定义
#endif：这是一个预处理指令，表示条件编译的结束。它与 #ifdef __cplusplus 配对使用，用于结束对C++环境的检查。
***/
#ifdef __cpluscplus
extern "C" {
#endif


#ifdef __cpluscplus
}
#endif

#endif

```





## 五、镜像容器使用：

ubuntu系统取消自动更新方法：

```
https://blog.csdn.net/inthat/article/details/125316389
cat /etc/issue
uname -r
dpkg --list | grep linux-image
dpkg --list | grep linux-headers
dpkg --list | grep linux-modules
vi /etc/apt/apt.conf.d/10periodic # 后面部分全部改成 “0”
vi /etc/apt/apt.conf.d/20auto-upgrades # 后面部分全部改成 “0”
reboot
```

参考 《算法开发服务器（镜像_容器）管理.md》

```
# 版本更新1：pip install pytorch-lightning==1.9.5 
# 版本更新2：pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html 
# 版本更新3：timm:0.4.5 -> 0.6.13 pip install timm==0.6.13
# commit 镜像
docker commit 46a4561f6d49 nvidia/cuda:11.7-cudnn8-devel-ubuntu18.04
# save
docker save -o Ind_Vision_Base.tar nvidia/cuda:11.7-cudnn8-devel-ubuntu18.04
```

#### 121服务器：

```
nvidia-docker run -dit --name cv_algo -p 5322:22 -p 5330-5399:5330-5399 -v /etc/localtime:/etc/localtime -v /var/wdcvlm:/var/wdcvlm -v /var/gtcvlm:/var/gtcvlm -v /etc/machine-id:/etc/machine-id -v /data/algorithm/cv_algo/dataset/public:/root/dataset/public -v /data/algorithm/cv_algo/dataset/cv_algo:/root/dataset/cv_algo -v /data/algorithm/cv_algo/project/cv_algo:/root/project/cv_algo -v /data/algorithm/cv_algo/shared:/root/shared -v /data/algorithm/cv_algo/common/pretrained/_.torch:/root/.torch -v /data/algorithm/cv_algo/common/pretrained/_.cache:/root/.cache -v /dev/shm:/dev/shm --privileged nvidia/cuda:11.7-cudnn8-devel-ubuntu18.04
```

```
nohup jupyter-notebook --no-browser --ip 0.0.0.0 --port 5340 --allow-root > jupyter.nohub.out &
```

#### 192.168.1.6 服务器环境配置满足 shapebasematching 编译环境：opencv 升级为4.6 修改部分代码

传统cv编译环境配置：

```python
apt update
apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3
apt-get install libgl1-mesa-dev

# https://cmake.org/download/
wget https://github.com/Kitware/CMake/releases/download/v3.25.3/cmake-3.25.3.tar.gz

tar -zxvf cmake-3.25.3.tar.gz

apt install gcc
apt install g++
apt install build-essential
apt install libssl-dev

cd cmake-3.25.3
./bootstrap

make -j8
make install
```

**ubuntu 虚拟环境中 安装Eigen库并配置编译环境：**

```python
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar -xzf eigen-3.3.9.tar.gz
cd eigen-3.3.9
mkdir build
cd build
cmake ..  # 安装结果在 /usr/local/include/eigen3
make
make install 
# 安装在虚拟环境的include 下：
cmake .. -DCMAKE_INSTALL_PREFIX=/root/conda/envs/cv_env/include/eigen3
make
make install 
```

```
# 参考 https://blog.csdn.net/u014491932/article/details/124886394 
# 参考 https://blog.csdn.net/gentleman1358/article/details/126955032
# 参考 https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
# https://github.com/opencv/opencv_contrib/releases/tag/4.6.0
apt-get install build-essential 

# libgtk2.0-dev 失败
apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

wget -O opencv-4.6.0.zip https://github.com/opencv/opencv/archive/refs/tags/4.6.0.zip
# 解压 opencv_contrib-4.6.0.zip 并放在opencv-4.6.0下
wget -O opencv_contrib-4.6.0.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.6.0.zip

cd opencv-4.6.0

mkdir -p build && cd build
# sudo cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
# cmake ..
cmake -DOPENCV_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/opencv/ -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.6.0/modules ..
make -j8
make install

ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2
ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
# -- Installing: /usr/local/bin/opencv_version
```

```
python setup.py build_ext --inplace
```

```
注意：
1、安装opencv后需要添加环境变量
~/.bashrc 中： export OpenCV_INCLUDE_DIRS=/usr/local/include/opencv4/opencv2
2、eigen安装后要在setup.py中指定include路径
# eigen_include = os.path.join(cuda_include, "eigen3")
eigen_include = "/usr/local/include/eigen3"
```

opencv 报错： error: ‘CV_BGR2GRAY’ was not declared in this scope
         cv::cvtColor(img, gray, CV_BGR2GRAY);

这个错误是因为在 OpenCV 4.x 版本中，颜色转换的常量名称已经发生了变化，不再使用类似 `CV_BGR2GRAY` 的形式。在 OpenCV 4.x 中，你应该使用 `cv::COLOR_BGR2GRAY` 这种形式来表示颜色转换常量。

解决这个问题的方法是将你的代码中的颜色转换常量名称替换为新的名称。以下是对应关系：

| 旧版本 (2.x/3.x) 常量 | 新版本 (4.x) 常量    |
| --------------------- | -------------------- |
| `CV_BGR2GRAY`         | `cv::COLOR_BGR2GRAY` |
| `CV_RGB2GRAY`         | `cv::COLOR_RGB2GRAY` |
| `CV_GRAY2BGR`         | `cv::COLOR_GRAY2BGR` |
| `CV_GRAY2RGB`         | `cv::COLOR_GRAY2RGB` |
| ...                   | ...                  |

#### 镜像版本更新：

```shell
docker commit 44c47e5eb249 nvidia/cuda:11.7-cudnn8-devel-ubuntu18.04-v1
```

```shell
nvidia-docker run -dit --name test_algo -p 6322:22 -p 6330-6399:6330-6399 -v /etc/localtime:/etc/localtime -v /var/wdcvlm:/var/wdcvlm -v /var/tscvlm:/var/tscvlm -v /etc/machine-id:/etc/machine-id -v /mnt/data/algorithm/bp_algo/dataset/public:/root/dataset/public -v /mnt/data/algorithm/bp_algo/dataset/bp_algo:/root/dataset/bp_algo -v /mnt/data/algorithm/bp_algo/project/bp_algo:/root/project/bp_algo -v /mnt/data/algorithm/bp_algo/shared:/root/shared -v /mnt/data/algorithm/bp_algo/common/pretrained/_.torch:/root/.torch -v /mnt/data/algorithm/bp_algo/common/pretrained/_.cache:/root/.cache -v /dev/shm:/dev/shm --privileged nvidia/cuda:11.7-cudnn8-devel-ubuntu18.04-v1
```

```shell
docker exec -it test_algo bash
service ssh start
```

```
# 配置jupyter密码： getech
jupyter notebook password 
# 在~/⽬录下，启动jupyter服务： 
nohup jupyter-notebook --no-browser --ip 0.0.0.0 --port 6340 --allow-root > jupyter.nohub.out &
```

```
find ./ -name libopencv_features2d.so.406
```

镜像版本更新保存：

```
docker commit 9e787816e502 nvidia/cuda:11.7-cudnn8-devel-ubuntu18.04-v2

docker save -o Ind_Vision_Base_V2.tar nvidia/cuda:11.7-cudnn8-devel-ubuntu18.04-v2
```



#### 7322 C++ 环境配置：

```python
cp -r cmake-3.25.3.tar.gz  /path
cp -r eigen-3.3.9.tar.gz /path
cp -r opencv-4.6.0.zip /path
cp -r opencv_contrib-4.6.0.zip /path

# -------cmake -----------
apt update
apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3
apt-get install libgl1-mesa-dev
# https://cmake.org/download/
wget https://github.com/Kitware/CMake/releases/download/v3.25.3/cmake-3.25.3.tar.gz
tar -zxvf cmake-3.25.3.tar.gz
apt install gcc
apt install g++
apt install build-essential
apt install libssl-dev
cd cmake-3.25.3
./bootstrap
make -j8
make install



# -----------eigen---------
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.tar.gz
tar -xzf eigen-3.3.9.tar.gz
cd eigen-3.3.9
mkdir build
cd build
# cmake ..  # 安装结果在 /usr/local/include/eigen3
# make
# make install 
# 安装在虚拟环境的include 下：
cmake .. -DCMAKE_INSTALL_PREFIX=/root/conda/envs/cv_env/include/eigen3
make
make install 


# ---------- opencv4.6.0------------
# 参考 https://blog.csdn.net/u014491932/article/details/124886394 
# 参考 https://blog.csdn.net/gentleman1358/article/details/126955032
# 参考 https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
# https://github.com/opencv/opencv_contrib/releases/tag/4.6.0
apt-get install build-essential 

# libgtk2.0-dev 失败
apt-get install libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

wget -O opencv-4.6.0.zip https://github.com/opencv/opencv/archive/refs/tags/4.6.0.zip
# 解压 opencv_contrib-4.6.0.zip 并放在opencv-4.6.0下
wget -O opencv_contrib-4.6.0.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.6.0.zip
cp -r opencv_contrib-4.6.0 ./opencv-4.6.0
cd opencv-4.6.0

mkdir -p build && cd build
# sudo cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..
# cmake ..
cmake -DOPENCV_DOWNLOAD_URL=https://mirrors.tuna.tsinghua.edu.cn/opencv/ -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.6.0/modules ..
make -j8
make install

# ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2
# ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
# -- Installing: /usr/local/bin/opencv_version
# --.bashrc--
PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_PATH
export OpenCV_INCLUDE_DIRS=/usr/local/include/opencv4/opencv2
export EIGEN3_INCLUDE_DIR=/usr/local/include/eigen3

# lib 配置
vim /etc/ld.so.conf.d/opencv.conf 输入 /usr/local/lib 再退出 ldconfig

# 配置 /usr/local/lib/pkgconfig opencv.pc文件中加入
prefix=/usr/local
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib

Name: opencv
Description: The opencv library
Version:4.6.0
Cflags: -I${includedir}/opencv4
Libs: -L${libdir} -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres -lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann  -lopencv_core


#vim /etc/bash.bashrc 文件最后添加 虚拟环境就添加再~/.bashrc即可
# PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig 
# export PKG_CONFIG_PATH
# source /etc/bash.bashrc

# 查看版本
pkg-config opencv --modversion

```



