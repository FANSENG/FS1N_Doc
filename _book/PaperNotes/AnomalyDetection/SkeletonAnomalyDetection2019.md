# Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos

## Metadata

- **CiteKey**: SkeletonAnomalyDetection2019
- **Type**: conferencePaper
- **Title**: Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos
- **Author**: Morais, Romero; Le, Vuong; Tran, Truyen; Saha, Budhaditya; Mansour, Moussa; Venkatesh, Svetha
- **Year**: 2019
- **Journal**: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
- **Pages**: 11988-11996
- **Publisher**: IEEE
- **DOI**: 10/ghj5xf
---
- **Url**: [Open online](https://ieeexplore.ieee.org/document/8953884/)
- **zotero entry**: [Zotero](zotero://select/library/items/AKGNN9KM)
- **open pdf**: [Morais 等。 - 2019 - Learning Regularity in Skeleton Trajectories for A.pdf](file:///C:%5CUsers%5C15750%5CZotero%5Cstorage%5CJKUFYHM5%5CMorais%20%E7%AD%89%E3%80%82%20-%202019%20-%20Learning%20Regularity%20in%20Skeleton%20Trajectories%20for%20A.pdf)
- **Keywords**: ObsCite
---
## Abstract
> Appearance features have been widely used in video anomaly detection even though they contain complex entangled factors. We propose a new method to model the normal patterns of human movements in surveillance video for anomaly detection using dynamic skeleton features. We decompose the skeletal movements into two sub-components: global body movement and local body posture. We model the dynamics and interaction of the coupled features in our novel Message-Passing Encoder-Decoder Recurrent Network. We observed that the decoupled features collaboratively interact in our spatio-temporal model to accurately identify human-related irregular events from surveillance video sequences. Compared to traditional appearancebased models, our method achieves superior outlier detection performance. Our model also offers “open-box” examination and decision explanation made possible by the semantically understandable features and a network architecture supporting interpretability.
- 数据来自监控视频
- 分解骨骼运动为两部分
	1.  全局身体行为
	2.  局部身体姿态
- 使用不同feature之间的相互关系，用于对正常行为和异常行为建模
- 在时空模型中使用解耦特征间的相互关系可以有效的提高对异常行为的识别
---
## Summary
- 本文解决的是 视频异常检测 问题，其输入数据并不是监控视频，而是从监控视频中提取得到的 人体骨骼检测结果，具体划分为两部分，global 和 local，其中 global 指人体中心在整个图像中的位置， local 指检测到的各个关节点相对于人体中心的位置，global 对位置敏感而不对身体具体姿态敏感， local 对身体姿态敏感对位置不敏感。
- 本实验的表现十分依赖于骨骼检测的结果，可解释性强，对噪音不敏感，包含丰富语义，特征维度低的优点。
- 模型以 GRU 为基础，引入了Message Passing机制，没有完全孤立 global 和 local，并且使 global 和 local 可以互相传递信息，协同辨别异常行为。通过对决策权重可视化能帮助实验者了解判断异常的具体因素，有助于深入了解模型的内部决策逻辑。
- 模型架构为单编码器双解码器架构，要求模型即可以还原出输入的骨骼架构，还能对模型不可见的未来骨骼结构做出预测。

- 本模型具有如下缺陷：
	1. 依赖于高质量的骨骼检测和跟踪，对骨骼检测跟踪有极高要求。
	2. 只关注人体骨骼情况，会忽视人与环境交互的内容。如环境中存在草坪，行走直接踏入草坪也应判断异常行为。
	3. 对于Message Passing机制的可解释性差。
	4. 模型并行度低。
---
## 1 Introduction
- 视频异常检测的挑战：
	1. 缺乏人类监督
	2. 对视频中异常行为的定义不明确
- 大多数的方案都是基于像素外观和特定动作特征
	- 高维非结构化
	- 对噪音敏感
	- 存在信息冗余，会增大训练和区分噪音的难度。
	- 缺乏可解释性，在视觉特征和行为的真实含义中存在语义差距。
- 本文使用2D人类骨骼轨迹，骨骼特征相比于上述方案优点如下：
	1. 简洁小型
	2. 强结构性
	3. 富语义
	4. 可高度描述人类行为
- 本文将骨骼运动分解为两个子块
	1. 全身运动 -> 身体在屏幕中的位置
	2. 局部身体姿态 -> 各关节点 相对于 全身检测框中心点的位置
- 模型称为MPED-RNN，由两个RNN分支组成，分别对应global和local。
	- 模型可对影响决策的因素和权重进行可视化来提供内部推理的开放式解释。
---
## 2 Relate Work

### 2.1 Video anomaly detection
- 传统方法 : 单分类SVM、混合概率PCA
- 新技术 : 
	1. CNN
	2. 光流预测(成本高)
	3. 结构化表示(物体轨迹来指导视觉特征池化，给予重点区域更多attention)
	4. 使用目标检测、属性检测和动作检测来理解异常分数(易被检测到的不相关信息影响，且label sets不能覆盖所有内容)
- 本文技术 : 
	1. 低维特征(骨骼点)
	2. 语义丰富
	3. 可解释性强
---
### 2.2 Human trajectory modeling

- 本文采取 提取主要的正常特征模式来排除异常 的方法。
- Du[^1]等人提出了划分骨骼点为五个部分，五个 双向神经网络 联合建模。
- 本文自然分解人类行为为 全身运动 和 局部变形 两部分，并使用 交互循环网络 联合建模。
---
## 3 Method

- 本文实验前提是骨骼特征已经从视频中提取出来，即本文的输入数据为骨骼特征序列。
- 每个时间步的输入数据为如下，其中$k$为关节点的数量。
$$
f_{t}=(x_{t}^{i},y_{t}^{i})_{i=1\dots k}
$$
---
### 3.1 Skeleton Motion Decomposition

- [^2][^3]中没有将骨骼特征分解为 全身运动 和 局部姿势，在具有统一骨架尺度和活动类型的视频中表现良好，其中两个因素的贡献是一致的。但在监控视频中，人体的骨架尺度很大程度上依赖于人的位置和动作，具体如下。
	1. 当人在靠近摄像头的区域，观察到的运动主要受局部因素影响。
	2. 当人在远离摄像头的区域，观察到的运动主要受全局因素影响，容易忽略局部因素。
- 本文将骨骼特征分解为 `global` 和 `local`两部分
	1. global : 主要包含人体检测框的 形状、大小和刚性运动。
	2. local : 包含骨骼内部变形的信息，忽略骨骼在屏幕中的绝对位置，只考察关节点相对于人体检测框中心的位置。
---
- 下图为本部分内容具体图示。
	- 规范参考系：以 检测框中心 为 原点，右方向和上方向分别代表 x,y 正方向。
		- 规范参考系将 局部分量 的值 标准化 为其 相对于 人体检测框 的大小，不去考虑原始真实值。如下方公式2，3。
	- 全局分量：由 监控区域左下角 指向 规范参考系原点 的向量。
	- 原始关节点：由 监控区域左下角 指向 具体关节点 的向量。
	- 局部分量：由 规范参考系原点 指向 具体关节点 的向量。如下方公式1。
	![](https://raw.githubusercontent.com/FANSENG/Figure-bed/master/20220728002555.png)
- 全局动作 和 局部动作 在两个并发的子进程中建模。
- 在特定环境中（如走廊），两个分量之间一般都具有较强的相关性。
---
- 本部分公式如下
$$
f_{t}^{l,i}=f_{t}^{i} - f_{t}^{g} \tag{1}
$$
$$
x^{g} = \frac{max(x^i)+min(x^i)}{2};\ \ \ \ y^{g} = \frac{max(y^i)+min(y^i)}{2};\tag{2-1}
$$
$$
w = max(x^i) - min(x^i);\ \ \ \ h = max(y^i) - min(y^i) \tag{2-2}
$$
$$
x^{l,i} = \frac{x^i-x^g}{w};\ \ \ \ y^{l,i} = \frac{y^i-y^g}{h} \tag{3}
$$
---
### 3.2 MPED-RNN Architecture

- MPED-RNN 对 全局分量 和 局部分量 建模时，使用了两个可以交互的子进程，互相传递 内部状态 作为对方下一个时间步的输入。
- MPED-RNN 包含两个 循环编码解码网络 分支，分别负责处理 全局变量 和 局部变量， 每个分支都包含三个RNN， 架构为 单编码器双解码器， 分别为 Encoder、Reconstructing Decoder 和 Predicting Decoder。
- 架构图具体如下：
![](https://raw.githubusercontent.com/FANSENG/Figure-bed/master/20220728235330.png)
---
- Encoder：
	1. 输入长度为 `T` 的骨骼序列，首先将所有GRU的隐藏状态初始化为 null。
	2. 对于每个时间步 `t`，骨骼信息 $f_{t}$被分解为$f_{t}^g$和$f_{t}^l$，如公式1、2和3。
	3. 两个分支间互相传递按照下方 公式4和5 计算出的消息。
	4. 对于每个时间步，全局分量和局部分量通过下方 公式6和7 进行编码。

- Reconstructing Decoders：
	1. 标准化隐藏状态为 $h_{T}^{gr} = h_{T}^{ge}$ 和 $h_{T}^{lr} = h_{T}^{le}$。
	2. 对于 $t=T,T-1,\dots ,1$，使用下方 公式8，9 更新隐藏状态。

- Predicting Decoders
	1. 标准化隐藏状态为 $h_{T}^{gp} = h_{T}^{ge}$ 和 $h_{T}^{lp} = h_{T}^{le}$。
	2. 对于 $t=T+1,T+2,\dots ,T+P$，使用下方 公式10，11 更新隐藏状态。
---
- 双解码器架构共同强制编码器学习紧凑的表示，保证编码结果可以重建输入并预测未来。
- 在测试过程中，异常的行为不会被预测到，因为模型从没有见过异常数据并且异常数据不符合正常动作模式。
- 在解码器中，投影特征 $f_{t}^g$ 和 $f_{t}^l$ 由 $h_{t}^g$ 和 $h_{t}^l$ 通过两个独立的全连接层生成，然后将两个投影特征拼接起来输入另一个全连接层得到$f_{t}$。在理想情况下，$f_{t}^g$ 、 $f_{t}^l$  和 $f_{t}$ 三者满足 公式1，2和3。
- 使用全连接层学习逆映射可以提高模型鲁棒性，能过滤掉噪音；？==投影特征可以用于评估输入的骨骼序列和学习到的正常行为的一致性，因此被用来构建损失函数和测试得分==？。
---
- 本部分公式如下
$$
m_{t}^{l\to g}=\sigma(W^{l \to g}h_{t-1}^l+b^{l \to g}) \tag{4}
$$
$$
m_{t}^{g\to l}=\sigma(W^{g \to l}h_{t-1}^g+b^{g \to l}) \tag{5}
$$
$$
E^{ge}:h_{t}^{ge}=GRU([f_{t}^g,m_{t}^{le\to ge}],h_{t-1}^{ge}) \tag{6}
$$
$$
E^{le}:h_{t}^{le}=GRU([f_{t}^l,m_{t}^{ge\to le}],h_{t-1}^{le}) \tag{7}
$$
$$
D_{r}^g:h_{t-1}^{gr}=GRU(m_{t}^{lr\to gr},h_{t}^{gr}\tag{8})
$$
$$
D_{r}^l:h_{t-1}^{lr}=GRU(m_{t}^{gr\to lr},h_{t}^{lr}\tag{9})
$$
$$
D_{p}^g:h_{t}^{gp}=GRU(m_{t}^{lp\to gp},h_{t-1}^{gp} \tag{10})
$$
$$
D_{p}^l:h_{t}^{lp}=GRU(m_{t}^{gp\to lp},h_{t-1}^{lp} \tag{11})
$$
---
### 3.3 Training MPED-RNN
- Training setup
	- 骨骼检测轨迹在视频中出现的帧数不同，但模型需要输入固定长度的序列，所以使用了滑动窗口策略解决此问题，将完整的骨骼轨迹分割为多段。如 公式12、13，其中`s`为步长，`T`为片段长度。

- Loss functions
	- 本文基于三个坐标系定义了三个损失函数，分别为Perceptual loss $L_{p}$、Global loss $L_{g}$和Local loss $L_{p}$
	- Perceptual loss 约束模型在图像坐标系中生成正常序列。
	- Global loss 和 Local loss 负责约束模型中的两个encoder-decoder分支。
	- 公式14 为三个损失函数的通式，`*`指代`p`、`g`和`l`其中之一。当`*`指代`p`的情况下，若预测序列的长度达到真实轨迹的长度，则会截断。
		- 公式中双竖线为范数符号，下方2表示L2范数，上方2表示平方。
		-  L2范数定义为向量所有元素的平方和的开平方。
	- 公式15 为总损失函数。其中${ \lambda _{g}, \lambda _{l}, \lambda _{p}} ≥ 0$。
---
- 本部分公式如下
$$
 seg_{i} = \{f_{t}\}_{t=b_{i} \dots e_{i}} \tag{12}
$$
$$
b_{i} = s \times i;\ \ \ \ e_{i}=s\times i+T \tag{13}
$$
$$
L_{*}(seg_{i})=\frac{1}{2}( \frac{1}{T} \sum_{t=b_{i}}^{e_{i}}\mid\mid \hat{f_{t}^*}-f_{t}^*\mid\mid_{2}^2+\frac{1}{P}\sum_{t=e_{i}+1}^{e_{i}+P}\mid\mid \hat{f_{t}^*}-f_{t}^*\mid\mid_{2}^2) \tag{14}
$$
$$
L(seg_{i})=\lambda_{g}L_{g}(seg_{i})+\lambda_{l}L_{l}(seg_{i})+\lambda_{p}L_{p}(seg_{i}) \tag{15}
$$
---
### 3.4 Detecting Video Anomalies
- 评价异常分数的4个步骤
$$
\hat{f_{t}} \to  S_{t} \to [S_{t}^g, S_{t}^l] \to \alpha _{f_{t}} \to \alpha _{v_{t}}
$$
1. 提取片段

	- 针对每个轨迹，使用步长为 S ，长度为 T 的滑动窗口选择重叠的骨骼片段。类似上文 公式12、13 。
2. 评估片段loss : 

	- 使用 公式1 分解片段，然后将所有的特征喂入模型，得到 公式 15 输出的Loss。
3. 收集骨骼异常得分 : 

	- 设置一个投票方案，根据 公式16 总结各片段中的Loss来得到骨骼一场得分。其中 u 为骨骼片段。
4. 计算帧异常分数 : 

	- 根据 公式17 计算帧异常分数，其中$Skel(v_{t})$代表帧中出现的骨骼集合，使用max的原因是为了忽略正常行为的异常分数。
---
- 本部分公式如下
$$
\alpha_{f_{t}}=\frac{\sum_{u \in S_{t} L_{p}(u) }}{\lvert S_{t} \rvert} \tag{16}
$$
$$
\alpha_{v_{t}} = max(\alpha_{f_{t}})_{f_{t}\in Skel(v_{t})} \tag{17}
$$
---
### 3.5 Implementation Details
- 本文使用 Alpha Pose[^4]进行骨骼检测。
- 结合稀疏光流和检测到的骨骼，在相邻的帧中，给予每对骨骼相似的分数，并使用匈牙利算法解决了匹配问题。（目标追踪）
- 对 Gloabl 和 Local 标准化时，使用原值减去特征的中值，然后忽略最大和最小的10%的值进行标准化，即取排序后 10%处 和 90%处 的值进行标准化。
---
## 4 Experiments
- 本文使用了两个数据集评估本算法，分别是`ShanghaiTech Campus`和`CUHK Avenue`。他们在数据来源、视频质量和异常类型方面都有所不同。
---
### 4.1 ShanghaiTech Campus Dataset
- 本数据集被认为是当时最全面和现实的视频异常检测的数据集之一，包含上海科技大学13个摄像头的数据，异常类型广泛。
- 大部分异常都与人相关，本文丢弃了测试数据中6个与人无关的异常视频，保留了其余101个与人相关的异常视频。
---
#### 4.1.1 Comparison with Appearance-based Methods

|            | HR-ShanghaiTech | ShanghaiTech |
|:----------:|:---------------:|:------------:|
|  Conv-AE   |      0.698      |    0.704     |
|  TSC sRNN  |       N/A       |    0.680     |
| Liu et al. |      0.727      |    0.728     |
|  MPED-RNN  |    **0.754**    |  **0.734**   |
- 上表为本文方法和其他前沿方法的ROC AUC对比。
![|450](https://raw.githubusercontent.com/FANSENG/Figure-bed/master/20220731235109.png)
- 上图为三个方法的比较，可以发现本文方法只专注于骨骼，不关注屏幕中其他与人无关的事务；而另外两个方法尝试在整个屏幕上去做异常检测，会更容易受到噪音的影响。
---
#### 4.1.2 Interpreting Open-box MPED-RNN
- 如下图本文对生成的 $\hat{f_{t}^g}$ 和 $\hat{f_{t}^l}$ 进行了可视化处理，并且在原图像空间中显示了预测的骨骼。红色为预测骨骼，黑色为输入骨骼。
- 本数据集采集自校园，下方Normal指的动作为 站立 或者 随意走 ，像奔跑、骑自行车等情况容易检测为异常。
- 左侧为正常行走情况下的对比，可以看到两者骨骼情况相似，且全局检测框位置基本吻合。
- 右侧为奔跑情况下的对比，可以观察到骨骼的差别较大，且全局检测框的吻合度也较差。
![](https://raw.githubusercontent.com/FANSENG/Figure-bed/master/20220731235613.png)
---
#### 4.1.3 Ablation Study
![|450](https://raw.githubusercontent.com/FANSENG/Figure-bed/master/20220801001401.png)
- 上表为MPED-RNN的各个简化变体在数据集上的表现。
---
#### 4.1.4 Error Mode Analysis
- 对本模型中识别错误的情况进行了考察，发现 骨骼检测和跟踪 的错误是识别错误的重要原因，骨骼检测和跟踪 出错的原因大部分是因为人类区域分辨率低、缺乏光照、 颜色对比问题、阴影问题等。并且对于很多人的情况下，tracking ID 很容易丢失或错误分配，这一点也会误导模型。
- 下方为骨骼检测出错的实例。左边误识别了玻璃中人的倒影，右边误将骑车的人的骨架识别为走路的骨架。
![](https://raw.githubusercontent.com/FANSENG/Figure-bed/master/20220801002204.png)
---
### 4.2 CUHK Avenue dataset
- 本文同样在CUHK Avenue数据集上进行了实验，包含16个训练视频和21个测试视频，均来自一个摄像头。
- 鉴于在上一个数据集的经验，本文手动对输入数据进行了处理，删除掉了与人无关的异常、人无法被检测和跟踪的帧。
- 对数据经过处理后，最后的 ROC AUC 为0.863。
---
## 5 Dicussion
- 每帧的骨骼特征平均长度小于100
- 模块化体系架构
- 依赖于高质量的骨骼检测和跟踪
- message-passing方案可以扩展于其他问题
- global-local方法可以扩展于其他项目上
---
## 6 Conclusion
- MPED-RNN简单、并且高度可解释
- 特征工程中进行了如下工作
	1. 检查人际互动的规律性
	2. 结合骨骼特征和外观
	3. 将基于组件的模型扩展到非人类对象
---
## Notes
- GRU原理图：
![](https://raw.githubusercontent.com/FANSENG/Figure-bed/master/20220802125424.png)
- 与LSTM相比，GRU有如下特点：
	1. GRU相比于LSTM少了输出门，其参数比LSTM少。
	2. GRU在复调音乐建模和语音信号建模等特定任务上的性能和LSTM差不多，在某些较小的数据集上，GRU相比于LSTM表现出更好的性能。
	3. LSTM比GRU严格来说更强，因为它可以很容易地进行无限计数，而GRU却不能。这就是GRU不能学习简单语言的原因，而这些语言是LSTM可以学习的。
	4. GRU网络在首次大规模的神经网络机器翻译的结构变化分析中，性能始终不如LSTM。
- GRU更适合于对速度要求高，数据集较小的任务上。
---
## References
[^1]:Yong Du, Yun Fu, and Liang Wang. Representation learning of temporal dynamics for skeleton-based action recognition. IEEE Transactions on Image Processing, 25(7):3010–3022, 2016. 2.2
[^2]:Katerina Fragkiadaki, Sergey Levine, Panna Felsen, and Jitendra Malik. Recurrent network models for human dynamics. In IEEE International Conference on Computer Vision, pages 4346–4354, 2015. 2.2, 3.1
[^3]:Ruben Villegas, Jimei Yang, Yuliang Zou, Sungryull Sohn, Xunyu Lin, and Honglak Lee. Learning to generate longterm future via hierarchical prediction. In International Conference on Machine Learning, pages 3560–3569, 2017. 2.2, 3.1
[^4]:Hao-Shu Fang, Shuqin Xie, Yu-Wing Tai, and Cewu Lu. RMPE: Regional multi-person pose estimation. In IEEE International Conference on Computer Vision, pages 23532362, 2017. 3.5