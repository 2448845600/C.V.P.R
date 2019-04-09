# Anchor or Anchor-Free
Anchor or Anchor-Free, that's a problem. But I want to choose Anchor-Free.

[TOC]

## History of Anchor and Anchor-Free
目标检测本无anchor，用的人多了，便有了anchor。

R-CNN是基于深度学习的目标检测算法开山之作，利用Selective Search算法生成候选区域，归一化后送入网络提取特征，最后分类得到类别、回归得到精细位置。YOLO v1首次提出one-stage目标检测算法，利用卷积神经网络直接输出$(x,y,w,h,c,cls_1......cls_n)$。这些目标检测“老将”面临以下问题：
- 多尺度问题
- R-CNN需要更好的proposals
- YOLO v1一个cell只能预测一个物体，所以面对密集物体漏检严重

Faster-RCNN提出anchor应对上述问题。anchor通常是人为设计的一组框，作为类别分类、位置回归的基准，被one-stage和two-stage检测算法广泛使用。two-stage算法一般利用包含anchor的RPN生成proposal，再对proposals分类和精细回归，如Faster R-CNN。one-stage算法一般在feature map的每个位置，根据anchor预设的参数scale和ratios，预测bbox相对于anchor的改变量，如SSD、YOLO v2。anchor大致发挥了以下四点作用：
- 对于two-stage算法，anchor得到的候选框质量更高，数量可以调整
- 对于one-stage算法，anchor便于匹配predict box与ground-truth box，实现一个cell/location预测多个bbox
- 设置不同尺寸（scale）不同形状（ratios）anchor，多尺度预测
- 预设anchor形状，相当于增加了先验知识，使特定类别与特定scale&ratios匹配

但是anchor本质上只是对暴力搜索proposals的一种剪枝操作，存在以下问题：
- 需要预设大量超参（number、scale、ratio等），对性能影响大
- 固化了模型的检测方向，不利于检测极端长宽比或形状多变的物体，不利于任务迁移
- 为保证检测效果，需要大量anchor boxes，但多分布在背景区域为负样本，导致正负样本不均衡
- 训练过程需要计算所有anchor boxes与ground truth boxes的IoU，计算量大

当然也存在anchor-free的思路。一种是YOLO v1的延续，如CornerNet。另一种以DenseBox为代表。下面几篇CVPR-2019论文从不同角度思考anchor的缺点，提出了anchor或anchor-free的解决思路。

## Guide the Better Anchor
论文：Region Proposal by Guided Anchoring，CVPR2019

### Motivation
传统的anchor需要生成大量anchor boxes以达到检测效果，这导致大量boxes分布在背景区域，导致正负样本不均衡；而预设参数的anchor难以检测极端长宽比的物体。所以，**目标检测需要相对稀疏、形状可变的anchor**。本文总结了anchor的**设计准则**：alignment（中心对齐）和consistency（特征一致）。
- Alignment
  从Faster R-CNN来看，每个anchor对应feature map上一个像素点，所以anchor的中心点最好就是这个像素点。在实际操作中，two-stage的方案会对anchor进行回归修正，导致这两个点偏移；one-stage的方案直接回归得到anchor中心，也存在偏移。偏移有没有坏处呢？或者是alignment有没有好处？有，但是都是实验/先验的推断。

- consistency
  每个类别的anchor应当具有一致性，否则难以学习到特征。其核心在于anchor与feature map匹配，如果不同图片相同位置预测的anchor大小千差万别，造成不匹配，则使得网络难以收敛，预测结果可能不够精确。

### Guided Anchoring and Feature Adaption
我们现在希望干的事情很简单：不利用超参数（尺寸和长宽比），预测anchor的x、y、w、h。我们要对feature map中每个点，直接预测这四个值吗？这里似乎有点问题。本文提出一个**先验**：
$$
p(x, y, w, h |I) = p(x,y | I) p(w,h | x, y, I) 
$$
**即anchor的预测分为两个部分，位置预测和形状预测**。如图，在FPN的基础上，采用两个分支分别预测位置和形状，组合得到anchor。其中，位置预测分支完成分类任务，即该像素点是否是物体中心，如果是物体的中心，就认为这是x,y，否则pass；形状预测分支是回归任务，即在该点是物体中心的前提下，预测形状。这部分就是红色框的内容。

![A5q4Ig.png](https://s2.ax1x.com/2019/04/09/A5q4Ig.png)

但是该方式造成一个问题：对于不同的输入，特征图同一个位置预测的anchor形状各不相同，但是特征图所有区域的感知野（receiptive field）却是一样的，造成anchor形状与feature map不匹配，违背前文提到的consistency准则。所以，**本文设计Feature Adaptation模块，将Guided Anchoring模块得到的形状信息融入feature map，实现匹配**。

我的理解：从consistency角度看，传统的anchor利用预设的scale，使得feature map每个点预测多组相同形状的框，达到“匹配”，而Guided anchoring预测完全不同scale的框，将scale与feature map融合，达到“匹配”。这里采用了3\*3大小的deformable convolution（可形变卷积），但是每个位置的offset是由该位置的(w, h)通过1\*1的卷积得到的。

简单提一下**训练**，详见论文。
$$
L = λ_1L_{loc} + λ_2L_{shape} + L_{cls} + L_{reg}
$$

### The Use of High-quality Proposals

通过Guided Anchoring，我们得到了更加精准的anchor，预选框的质量显著提升。但是detector性能提升有限。本文通过以下两个改进继续提高检测性能：
- 减少 proposal 数量
- 增大训练时正样本的IoU阈值（这个更重要）

这些改进来源于下图中对预测框的统计分析。top300的预测框里面包含大量高IoU的proposal，所以不需要用1000个框来训练和测试；proposal的IoU都很高，所以有必要提高IoU阈值。将包含这两个改进的Guided Anchoring增加到Faster R-CNN中，性能提升2.7个点（无其他trick），其他方法上也有大幅提升。

![A5qbMq.png](https://s2.ax1x.com/2019/04/09/A5qbMq.png)

### Conclusion
- 在anchor设计中，alignment和consistency这两个准则十分重要；
- 采用两个branch分别预测anchor的位置和形状，不需要预先定义；
- 利用anchor形状来adapt特征图；
- 高质量proposal可以使用更少的数量和更高的IoU进行训练；
- 即插即用，无缝替换。

## Match Feature with Box
论文：Feature Selective Anchor-Free Module for Single-Shot Object Detection，CVPR2019

Without anchor, we can match the best level of feature.

### Motivation
本文从scale variation角度出发，欲抑先扬，褒奖一波feature pyramid后，开始挑刺。基于anchor的检测器存在两个局限：
- heuristic-guided feature selection（启发式地选择特征）
- overlap-based anchor sampling（利用重叠的anchor采样）

含有anchor的检测模型通过IoU度量anchor与instance的距离，完成训练过程中的anchor-instance匹配。这种做法具有内在的启发式，导致训练过程中每个instance匹配的anchor不属于最优的特征金字塔层级（即instance对应的anchor与instance对应的特征图不够匹配）。所以，本文希望训练时每个instance能够选择最优层级的特征，为此不惜除去anchor。在这个指导思想下，设计实现了FSAF（Feature Selective Anchor-Free）模块。

### Feature Selective Anchor-Free Module

**Network Architecture**
其实没有多少大改，就是再RetinaNet的基础上做了点工作。对于A个anchor，K个class，原本预测类别概率W\*H\*KA，位置W\*H\*4A，现在预测类别概率W\*H\*A，位置W\*H\*4，类似于yolov1。区别在于类别和位置再两个分支预测。

![A5qvoF.png](https://s2.ax1x.com/2019/04/09/A5qvoF.png)

**Online Feature Selection**

前文提到，希望网络可以学习到用特定层特征预测特定问题，所以设计了该模块。本模块解决的是在训练过程中，如何将instance与bbox对应。对于一个instance，在特征金字塔的每个层级均预测一个bbox，计算loss（loss = focal loss + IoU loss），取min loss所在bbox作为该instance对应的bbox，进行反向传播。在消融实验中，还采用启发式特征选择策略（heuristic feature selection process）对比实验，即$l^{\prime}=\left\lfloor l_{0}+\log_{2}(\sqrt{w h} / 224)\right\rfloor$。本文实验部分表明Online Feature Selection优于Heuristic Feature Selection。

![A5L9zR.png](https://s2.ax1x.com/2019/04/09/A5L9zR.png)

**Joint Inference and Training**
本文并没有用anchor-free branch取代anchor-based branch，而是结合在一起。再推断过程中，通过不同的nms/confidence阈值，将这两个分支预测的bbox过滤后再混合。训练过程中，采用联合loss，$L = L^{ab} + \lambda(L^{af}_{cls} + L^{af}_{reg})$，其中$L^{af}_{cls}$采用focal loss（RetinaNet），$L^{af}_{reg}$采用IoU loss（UnitBox）。在文章的实验部分，有如下阐述：如果仅使用anchor-free branched，基本上只能达到anchor-based branch的效果（AP提升0.2%）；采用两者联合后，效果提升较大（AP提升2.5%）。

### Conclusion
- IoU-based anchor-instance match mechanism is heuristic 
- FSAF use min loss to select best level of feature
- anchor-free and anchor-based jointly can get the best result

## Unify to Fully-conv & Per-pixel Prediction
论文：FCOS: Fully Convolutional One-Stage Object Detection

### Motivation
本文按照惯例批判一波anchor后，提出一个有趣的论断：语义分割、关键点检测、深度估计都广泛采用**全卷积逐像素预测框架**（fully convolutional per-pixel prediction framework），而目标检测由于使用了anchor，偏离这个思路。

本文提出anchor-free的目标检测算法，FCOS。它的主要优点如下：
- 采用统一的FCN per-pixel prediction，可以复用semantic segmentation的tricks
- proposal free和anchor free，显著减少超参
- 可以应用在two-stage detector，取代FPN
- 可以修改FCOS的输出分支，用于解决instance segmentation和keypoint detection任务

### Fully Convolutional One-Stage
FCOS的整体结构如图

![A4qf0O.png](https://s2.ax1x.com/2019/04/08/A4qf0O.png)

**Fully Convolutional One-Stage Object Detector**
FCOS预测l,t,r,b作为物体的位置。分别表示上下左右到中心点的距离，训练就按照如下公式进行。

$$
\begin{aligned} 
l^{*} &=x-x_{0}^{(i)}, \quad t^{*}=y-y_{0}^{(i)} \\ 
r^{*} &=x_{1}^{(i)}-x, \quad b^{*}=y_{1}^{(i)}-y 
\end{aligned}
$$

loss具体描述见论文。对于特征图中的位置(x, y)（文章中用**location**(x, y)表示），$\boldsymbol{p}_{x, y}^{*}$表示预测的类别概率，$\boldsymbol{t}_{x, y}^{*}$表示预测的位置。

$$
\begin{aligned} 
L\left(\left\{\boldsymbol{p}_{x, y}\right\},\left\{\boldsymbol{t}_{x, y}\right\}\right) 
= \frac{1}{N_{\mathrm{pos}}} \sum_{x, y} L_{\mathrm{cls}}\left(\boldsymbol{p}_{x, y}, c_{x, y}^{*}\right)
&+ \frac{\lambda}{N_{\mathrm{pos}}} \sum_{x, y} \mathbb{1}_{\left\{c_{x, y}^{*}>0\right\}}     L_{\mathrm{reg}}\left(\boldsymbol{t}_{x, y}, \boldsymbol{t}_{x, y}^{*}\right) 
\end{aligned}
$$

![A5uaTO.png](https://s2.ax1x.com/2019/04/08/A5uaTO.png)

**Multi-level Prediction**
在训练中，必然会遇到“模糊样本（ambiguous sample）”问题，即一个ground-truth落在多个边界框中。简单方法是选择面积最小的边界框作为其回归目标。但是，本文采用多级预测（Multi-level Prediction），显著减少模糊样本的数量。在Feature Pyramid的$\left\{P_{3}, P_{4}, P_{5}, P_{6}, P_{7}\right\}$层特征图中，直接限制每层预测的$\left(l^{*}, t^{*}, r^{*}, b^{*}\right)$大小。$m_i$ 是第i层特征图预测结果的最大值。$m_{2}, m_{3}, m_{4}, m_{5}, m_{6}$ $m_{7}$ 被设置为 0，64，128，256，512，$\infty$。如果预测结果满足下式，这被视为negative sample，被抛弃。

$$
\begin{aligned}
& \max \left(l^{*}, t^{*}, r^{*}, b^{*}\right)>m_{i} \\
& \max \left(l^{*}, t^{*}, r^{*}, b^{*}\right)<m_{i-1}
\end{aligned}
$$

这个做法极大的减小了模糊样本数量。文章还提到了Head，没太明白是啥意思。

**center-ness**
现在，还有一个问题待解决：预测结果存在大量远离中心的低质量bbox。解决方式很简单，文章定义center-ness（the distance from the location to the center of the object that the location is responsible for，特征图中location到预测物体中心的距离），具体定义：

$$
centerness^{*} = \sqrt{\frac{\min \left(l^{*}, r^{*}\right)}{\max \left(l^{*}, r^{*}\right)} \times \frac{\min \left(t^{*}, b^{*}\right)}{\max \left(t^{*}, b^{*}\right)}}
$$

center-ness可以理解为给中心点打分，它度量当前像素点（feature location）是否处于ground truth target的中心区域。以下面的热力图为例，红色部分表示center-ness值为1，蓝色部分表示center-ness值为0，其他部分的值介于0和1之间。从公式可以看出，我们希望x，y是中心点。网络增加了一个分支预测center-ness，取代常见的confidence，过滤大量的远离中心点的低质量框。

![center](https://s2.ax1x.com/2019/04/08/A5JEp8.png)

### Conclusion
- predict (l,r,t,b) instead (x,y,w,h)
- use multi-level prediction to decrease ambiguous sample
- define center-ness to filter low-quality predicted bbox
- fully conv and per-pixvl design reuse trick of semantic segmentation

## Reference
[作者解读 Guided Anchoring](https://zhuanlan.zhihu.com/p/55854246)
[Anchor Boxes in Practice](https://medium.com/@andersasac/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9)