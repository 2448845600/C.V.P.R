# 多目标跟踪算法2019
最近一轮顶会的 MOT 论文
[TOC]

## Multi-Object Tracking with Multiple Cues and Switcher-Aware Classification
arxiv 1901

### Design
![overall design](.\imgs\design.png)
>Overall design of the proposed MOT framework. Short term cues are captured by SOT sub-net and long term cues are captured by ReID sub-net. The switcher aware classifer (SAC) predicts whether the detection result matches the target using the short term and long term cues of the target, the detection result, and the switcher.

跟踪流程分为9步，$I_{t}$表示第 $i$ 帧，$S$ 表示目标跟踪集，$I_{t,i}^{X}$ 表示目标 $i$ 在 $I_t$ 中的特征，具体步骤如下：

1. 初始化，目标跟踪集 $S$ 为空，$t＝1$。
2. 对于 $I_t$ 中的目标 $X$，通过 SOT sub-net 在 $I_{t+1}$ 找到的目标模板 $E_{X}$，输出其出现的最大概率位置$D_{track}$。
3.  对于 $I_{t+1}$ 的检测结果 $D_{det}$，获取它们在图中对应区域的特征 $I_{(t+1),D_{det}}^{X}$, l利用 ReID sub-net 得到每个跟踪目标的 long term features，$\{I_{t,i}^{X}\}, i=1,2,...,K$。
4. 将 $D_{det}, D_{track}, \{I_{t,i}^{X}\}$ 三者送入联合匹配器中。
5. 找出目标 $\Lambda$ 在匹配阶段可能出现的 potential switcher，提取其 SOT 和 ReID 特征。
6. 借助 switcher 的特征，利用 switcher-aware classifier 生成每个 detection 与跟踪目标的匹配分数，以判断检测结果与跟踪目标是否匹配。$I_{t+1}$中每个检测结果均需如此操作，以获得匹配分数。
7. 使用匹配分数构建跟踪目标和检测结果的二分图。利用最小费用流找到最佳匹配结果。
8. 对于匹配的目标，使用 matched detection 信息更新位置和模板；对于不匹配的目标，使用SOT 结果更新跟踪器位置，并丢弃不可靠或丢失的目标；对于孤立的检测结果，如果它们的置信度分数满足新目标的条件，它们将被添加到 $S$ 中。
9. 设置 $t = t + 1$，重复 step2 ~ 8 直到结束。

### Using SOT Tracker for Short Term Cues

<img src=".\imgs\Siamese-RPN.png" style="zoom:80%;" />
>Siamese-RPN architecture for SOT

将 Siamese-RPN 作为 SOT Tracker baseline。Siamese-RPN 有两个 branch，一个输出 score，记为 $p$，一个输出 box，称为 SOT box，记为 $D_{track}$。对应的，detection 输出的 box 称为 detection box，记为 $D_{det}$。Short term feature $f_s$：
$$
f_s (D_{track}, D_{det}) = IoU(D_{track}, D_{det})
$$

利用 SOT tracker 存在一个较大的问题：无法判断目标何时丢失，即出现 tracker 漂移到背景干扰者时，无法停止跟踪的问题。对应 step7，这里优化了跟踪得分，对于目标 $X$，其跟踪质量 $q_X$:
$$
q_{X, t+1}=\left\{\begin{aligned}
\frac{q_{X, t}+I o U\left(D_{t r a c k}, D_{d e t}\right) \cdot p}{2}, \text { if matched, } & \\
q_{X, t} \cdot \text { decay } \cdot p^{k}, \text { otherwise }
\end{aligned}\right.
$$

$\text {decay, k}$ 都是超参数，当 $q_X < \text{threshold}$，丢弃该目标。

### Using ReID Network for Long Term Cues

step3中提到，long-term features 包含 K images。 设计 quality-aware mechanism to select best K images。

$$
t_{i}=\underset{t-i \delta<\hat{t} \leq t-(i-1) \delta}{\arg \max } Q\left(I_{\hat{t}}^{X}\right), i=1,2, \ldots, K
$$

$\Q$ 是一个网络，输入是目标所在区域 (应该是 crop from RGB-image)，输出是 quality score。即对于目标 $X$，我们利用超参数 $\delta$ 将历史帧分为 $K$ 个间隔，选取每个间隔中 quality score 最大的一帧，得到 $X$ 的 $K$ 张 images。

利用 ReID sub-net 将 images 转化为 features，最终得到 long term features $\{A_{t_{i}}^{X}\}, i=1,2,...,K$。 $A_{det}$ 是 detection box 经过 ReID 的特征。余弦距离，得到 $\mathcal{F}_{l}^{X}$ ：
$$
\begin{array}{l}
\mathcal{F}_{l}^{X}=\left\{f_{l}\left(A_{t_{i}}^{X} A_{d e t}\right) | i=1, \ldots, K\right\} \\

\text { where } f_{l}\left(A_{t_{i}}^{X}, A_{\text {det}}\right)=\frac{A_{t_{i}}^{X^{\mathrm{T}}} \cdot A_{\text {det}}}{\left|A_{t_{i}}^{X}\right|\left|A_{\text {det}}\right|}
\end{array}
$$


### Switcher-Aware Classifier
先验：本文观察得到：身份转换（IDS）一般发生在两个目标重叠较大时。所以将与当前目标最大重叠的另一个目标标记为最可能的 potential switcher，$\Lambda$：
$$
\begin{aligned}
&\Lambda=\arg \max _{Y \in \mathcal{S} \text { s.t. } Y \neq X} \operatorname{Io} U\left(X_{t}, Y_{t}\right)
\end{aligned}
$$

其中 $S$ 即 step1 中的 $S$。

### Conclusion
本文比较复杂，但是思路也很清晰：
1. 得到更多高质量的框，这里选用SOT，Siamese-RPN
2. 处理IDS，即SAC

## Tracking without bells and tricks
ICCV2020

### Pipeline

![](.\imgs\Tracking_wo_b.png)
>The presented Tracktor accomplishes multi-object tracking only with an object detector and consists of two primary processing steps, indicated in blue and red, for a given frame $t$. First, the regression of the object detector aligns already existing track bounding boxes $b^k_{t−1}$ of frame $t−1$ to the object’s new position at frame $t$. The corresponding object classification scores $s^k_t$ of the new bounding box positions are then used to kill potentially occluded tracks. Second, the object detector (or a given set of public detections) provides a set of detections $D_t$ of frame t. Finally, a new track is initialized if a detection has no substantial Intersection over Union with any bounding box of the set of active tracks $B_t = \{b^{k_1}_t , b^{k_2}_t , · · · \}$.

本文的亮点如题所示，即 without bells and tricks，直接利用 detector 做 tracking：将上一帧得到的 bboxes 和当前帧的图片作为 detector 的输入，直接得到这些 bboxes 在当前帧的位置。

在具体实现上，将上一帧的 pred bboxes 作为当前帧的 region proposals，借助 Faster R-CNN 的回归器，得到tracker 在当前帧的位置。

### Tracking extensions

在此基础之上，增加了两个常见的tricks：motion model and re-id algorithm。


## How To Train Your Deep Multi-Object Tracker
CVPR2020

以《Tracking without bells and tricks》为 baseline，实现了一个 multi-task 的 End-to-End tracker。已有的 MOT Model，其 tracker 部分一般采用图论相关的优化方法，而不是 end-to-end 的神经网络，原因如下：
- 神经网络不能很好地处理 Data Association，只能用组合优化方法
- 没有合适的 loss 满足 MOT 任务，只能分阶段训练（Detection，ReID）

所以本文实现了一个 DHN，利用深度学习实现匈牙利算法；将 MOT 的评估指标 MOTA/MOTP 融入 loss function设计中。这里的匹配似乎没有用到特征，只用了中心点距离和IoU。

### Pipeline
整体结构如下：

![pipeline](.\imgs\pipelineV3.png)


### Deep Hungarian Net: DHN

传统算法在 MOT 中的最后堡垒：Data Association，现在，也被深度学习攻破！本文提出 DHN，Deep Hungarian Net，基于深度学习的匈牙利算法。如图所示：

![](.\imgs\DHN.png)
>Structure of the proposed Deep Hungarian Network. The row-wise and column-wise flattening are inspired by the original Hungarian algorithm, while the Bi-RNN allows for all decisions to be taken globally, thus is accounting for all input entries.

输入的距离矩阵 distance matrix， $D$；网络输出的分配矩阵 proxy assignment matrix， $\tilde{A}$；最优化方法计算的最优分配矩阵，optimal assignment matrix，$A^{*}$，即我们的优化目标、label。DHN 的设计需要满足：

- $\tilde{A}$ 是 $A^{*}$ 的有效近似
- 近似方法必须 differentiable（可微的）
- 输入输出尺寸与 Hungarian 相同
- 与 Hungarian 相同，基于全局信息做决策

对于 $D$，他是中心点距离（Euclidean distance，$f$）和IoU距离（Jaccard distance，$\mathcal{J}$）的平均值，是可微的。
$$
d_{n, m}=\frac{f\left(\mathbf{x}^{n}, \mathbf{y}^{m}\right)+\mathcal{J}\left(\mathbf{x}^{n}, \mathbf{y}^{m}\right)}{2}
$$

由于 DHN 是由 Bi-RNN 组成的，所以到 $\tilde{A}$ 的全过程亦可微。

### Differentiable MOTA and MOTP

现在有了 $\tilde{A}$，需要设计 loss function，评估其优劣。本文希望 loss function 可以和评估指标一样，训练时候的优化方向比较好。先介绍一下常见的评估指标，再选择 MOTA 和 MOTP，设计网络结构实现。

MOTChallenge 的官方评估指标如下：

![](.\imgs\MOT_评估指标.png)

FP、FN、IDS是三个比较基础的指标：
- FP 误判数，即在第 t 帧中给出的假设位置 $h_j$ 没有跟踪目标与之匹配；
- FN 漏检数，即在第 t 帧中目标 $o_j$ 没有预测框与之匹配；
- IDS 误配数，即在第 t 帧中跟踪目标发生 ID 切换的次数，错误多发生在这类情况下。

以此为基础，我们可以推导出最重要、综合的两个指标，MOTA 和 MOTP：

MOTA，multiple object tracking accuracy，是多目标跟踪的准确度，体现在确定目标的个数，以及有关目标的相关属性方面的准确度，用于统计在跟踪中的误差积累情况。$M$ 表示 ground truth 的数量。
$$
MOTA = 1 - \frac{\mathrm{FP}+\mathrm{FN}+ \mathrm{IDS}}{M}
$$

MOTP，multiple object tracking precision，多目标跟踪的精确度，体现在确定目标位置上的精确度，用于衡量目标位置确定的精确程度。$c_t$ 表示第 t 帧目标 $o_i$ 和假设 $h_j$ 的匹配个数；$b_t^i$ 表示第 t 帧目标 $o_i$ 与其配对假设位置之间的距离，即匹配误差。
$$
MOTP=\frac{\sum_{i, t} d_{t}^{i}}{\sum_{t} c_{t}}
$$

我们希望用神经网络拟合这两个指标，并且整个过程可微，使得参数可训练，故设计如下网络结构和损失函数：

![](.\imgs\DeepMOT-loss.png)
>DeepMOT loss: dMOTP (top) is computed as the average distance of matched tracks and dMOTA (bottom) is composed with FP, IDS and FN.

MOTP和MOTA可以由 FP，FN和IDS推算出来。
$$
dMOTA = 1 - \frac{\tilde{\mathrm{FP}} + \tilde{\mathrm{FN}} + \gamma \tilde{\mathrm{IDS}}}{M}
$$

$$
dMOTP = 1 - \frac{\left\|\mathbf{D} \odot \mathbf{B}^{\mathrm{TP}}\right\|_{1}}{\left\|\mathbf{B}^{\mathrm{TP}}\right\|_{0}}
$$

### How To Train

训练步骤如下，红色框是 Hungarian algorithm 模块，用于计算 $A^{*}$，作为训练的标签。

<img src=".\imgs\Train.png" alt="Train"  />
>The proposed MOT training strategy (bottom) accounts for the track-to-object assignment problem, that is solved by the proposed deep Hungarian network, and approximates the standard MOT losses, as opposed to the classical training strategies (top) using the non-differentiable Hungarian algorithm.


## Towards Real-Time Multi-Object Tracking
arxiv 1909


## DASOT: A Uniﬁed Framework Integrating Data Association and Single Object Tracking for Online Multi-Object Tracking
AAAI 2020

没有找到paper，只是看到了 list

## Learning a Neural Solver for Multiple Object Tracking
CVPR 2020

这项工作采用 MOT 的经典网络流（network flow ）公式来定义基于消息传递网络（MPN，Message Passing Networks）的完全可微分框架。 通过直接在图域上操作，该方法可以在整个检测范围内进行全局推理并预测最终解决方案。 

### Overview

![](.\imgs\Learning_a_slover.png)
>Overview of our method. (a) We receive as input a set of frames and detections. (b) We construct a graph in which nodes represent detections, and all nodes at different frames are connected by an edge. (c) We initialize node embeddings in the graph with a CNN, and edge embeddings with an MLP encoding geometry information (not shown in figure). (c) The information contained in these embeddings is propagated across the graph for a fixed number of iterations through neural message passing. (d) Once this process terminates, the embeddings resulting from neural message passing are used to classify edges into active (colored with green) and non-active (colored with red). During training, we compute the cross-entropy loss of our predictions w.r.t. ground truth labels and backpropagate gradients through our entire pipeline. (e) At inference, we follow a simple rounding scheme to binarize our classification scores and obtain final trajectories.


## 参考文献及博客
