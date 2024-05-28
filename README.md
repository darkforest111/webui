本项目是基于[CoMoSvc](https://github.com/Grace9994/CoMoSVC/)编写的webui，将会持续更新以支持更多功能


# 0.使用之前

#### 任何国家，地区，组织和个人使用此项目必须遵守以下法律

#### 《民法典》

##### 第一千零一十九条

任何组织或者个人不得以丑化、污损，或者利用信息技术手段伪造等方式侵害他人的肖像权。未经肖像权人同意，不得制作、使用、公开肖像权人的肖像，但是法律另有规定的除外。未经肖像权人同意，肖像作品权利人不得以发表、复制、发行、出租、展览等方式使用或者公开肖像权人的肖像。对自然人声音的保护，参照适用肖像权保护的有关规定。

##### 第一千零二十四条

【名誉权】民事主体享有名誉权。任何组织或者个人不得以侮辱、诽谤等方式侵害他人的名誉权。

##### 第一千零二十七条

【作品侵害名誉权】行为人发表的文学、艺术作品以真人真事或者特定人为描述对象，含有侮辱、诽谤内容，侵害他人名誉权的，受害人有权依法请求该行为人承担民事责任。行为人发表的文学、艺术作品不以特定人为描述对象，仅其中的情节与该特定人的情况相似的，不承担民事责任。

#### 《[中华人民共和国宪法](http://www.gov.cn/guoqing/2018-03/22/content_5276318.htm)》

#### 《[中华人民共和国刑法](http://gongbao.court.gov.cn/Details/f8e30d0689b23f57bfc782d21035c3.html?sw=中华人民共和国刑法)》

#### 《[中华人民共和国民法典](http://gongbao.court.gov.cn/Details/51eb6750b8361f79be8f90d09bc202.html)》

#### 《[中华人民共和国合同法](http://www.npc.gov.cn/zgrdw/npc/lfzt/rlyw/2016-07/01/content_1992739.htm)》


# 1.环境配置
Python 3.8环境下创建Conda虚拟环境:

```shell
conda create -n Your_Conda_Environment_Name python=3.8
```
安装相关依赖库:

```shell
pip install -r requirements.txt
```
## 下载checkpoints
### 1. m4singer_hifigan
下载vocoder [m4singer_hifigan](https://drive.google.com/file/d/10LD3sq_zmAibl379yTW5M-LXy2l_xk6h/view) ,并解压到`comosvc`文件夹下`m4singer_hifigan`路径

```shell
unzip m4singer_hifigan.zip
```

vocoder的checkoint将在`m4singer_hifigan`目录中

### 2. ContentVec

下载 [ContentVec](https://ibm.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) 解压在`comosvc`文件夹下`Content`路径，以提取歌词内容特征。

### 3. m4singer_pe

下载pitch_extractor [m4singer_pe](https://drive.google.com/file/d/19QtXNeqUjY3AjvVycEt3G83lXn2HwbaJ/view) ,并解压到`comosvc`文件夹下`m4singer_pe`路径

```shell
unzip m4singer_pe.zip
```

# 2.训练和推理
## 使用webUI
在文件根目录下新建文本文件，输入以下内容
```shell
@echo off
call conda activate your_env_name
python webui.py
```
其中`your_env_name`是你创建的conda环境的名字

将上述代码保存为`webui.bat`文件。在保存时，将`保存类型`设置为`所有文件`，并将文件名以`.bat`结尾

进行点击`webui.bat`打开webui页面，你可以在里面进行训练和推理

## 使用命令行
如要使用命令行进行训练和推理，请参考[CoMoSvc的README文档](https://github.com/Grace9994/CoMoSVC/blob/main/README.md)

# 3.其他小功能
## suno的非官方api
### 使用
在浏览器中获取cookie和session_id,并填写到`.env`文件中
![](D:\CoMoSVC-main\webui_comosvc\image\example.png)
打开webui即有对应功能

更多功能敬请期待！！！


# 致谢
 [CoMoSvc](https://github.com/Grace9994/CoMoSVC)


 