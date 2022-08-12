# Transformer

## 杂点

- Transformer在训练时是并行进行的，但在预测时不可并行。原因在于训练时已经知晓所有字符，如使用`['我爱你', 'I love you']`去训练模型，在训练翻译`你`的时候，我们已经知晓了前面的翻译结果`I`和`love`，所以加入掩码后就可以去并行训练；但在预测时，必须按照顺序先翻译出`I`和`love`，才能去翻译`你`。

- Teacher Force: 在每一轮预测时，不使用上一轮预测的输出，而强制使用正确的单词，过这样的方法可以有效的避免因中间预测错误而对后续序列的预测，从而加快训练速度，而Transformer采用这个方法，为并行化训练提供了可能，因为每个时刻的输入不再依赖上一时刻的输出，而是依赖正确的样本，而正确的样本在训练集中已经全量提供了。

- BatchNormal: 针对每一个Batch，将Batch中每一个特征标准化为 均值为$\lambda$, 方差为$\beta$ 的分布（跨样本）。

- LayerNormal: 将每一个样本标准化为 均值为$\lambda$, 方差为$\beta$ 的分布，只考察样本的均值和方差，不去看全局的均值方差（不跨样本），因为每个Seq中的token数量不相同。

- 为什么Attention公式（如下）中要除以$\sqrt{d_k}$：当$d_k$的值很大时，$QK^T$中值的差距可能较大，$softmax$后权值会更趋近于0和1，之后求导时会出现梯度差异过大的问题，会出现收敛困难的问题。极大的点积值会将整个$softmax$推向梯度平缓区，使得收敛困难，所以需要scaled。Add 是天然地不需要 scaled，Mul 在$d_k$较大的时候必须要做 scaled。
$$
Attention(Q, K, V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

- Multi-Head是将Q、K、V通过线性层投影到了低纬度($\frac{d_k}{h}$)，然后通过h个`Scaled Dot-Product Attention`得到h个长度为$\frac{d_k}{h}$的向量，将h个向量进行连接，得到的便是注意力权重。注：其中用来投影的线性层参数是可以学习的。（类似卷积中的多个filter）
- Positional Encoding负责了Transformer中序列关系的表达。
- 为什么Positional Encoding直接相加而不是Concat：在原始向量上concat一个代表位置信息的向量，再经过变换，最终的效果等价于：先对原始输入向量做变换，然后再加上位置嵌入。另一方面，concat会使网络进入深层之后参数量剧增，这是很不划算的。
- Transformer学习率方案为Warm up，计算公式如下（$warmup\_steps$设置为4000），学习率，Warm up会把Learning Rate从小变大然后再线性减小。优化器使用Adam，其中$\beta_1$为0.9，$\beta_2$为0.98。

$$
lrate=d_{model}^{-0.5}·min(step\_num^{-0.5}, step\_num·warmup\_steps^{-1.5})
$$



---

## 参考

1. 为什么Position Encoding中使用三角函数可以表示token间的位置关联:[Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/#the-intuition)
2. Transformer细枝末节的信息:[碎碎念：Transformer的细枝末节](https://zhuanlan.zhihu.com/p/60821628)
3. Transformer的Demo:[Universal-Transformer-Pytorch](https://github.com/DDzzxiaohongdou/Universal-Transfromer)
4. [Transformer之十万个为什么？](https://blog.csdn.net/air__Heaven/article/details/123663323)
5. [Transformer在训练、评估时编码器，解码器分别如何工作的？](https://zhuanlan.zhihu.com/p/405543591)
6. [Transformer源码详解（Pytorch版本）](https://zhuanlan.zhihu.com/p/398039366)
7. [Transformer你问我答](https://zhuanlan.zhihu.com/p/429061708)
8. [Transformer论文逐段精读【论文精读】](https://www.bilibili.com/video/BV1pu411o7BE?spm_id_from=333.999.0.0&vd_source=974cc80f4976d6736be76a21f52d4a2b)
9. 为什么Attention函数要除以$\sqrt{d_k}$:[深入分析transformer](https://zhuanlan.zhihu.com/p/359203426)
10. 为什么Positional Encoding可以直接相加而不是Concat:[为什么Transformer / ViT 中的Position Encoding能和Feature Embedding直接相加？](https://blog.csdn.net/qq_38890412/article/details/124581338)
11. Transformer学习率：[简化Transformer模型训练技术简介](https://zhuanlan.zhihu.com/p/438150240)
12. [从 0 开始学习 Transformer 上篇：Transformer 搭建与理解](https://gitee.com/LilithSangreal/LilithSangreal-Blog/blob/master/NLP/Transformer.md)
13. [从 0 开始学习 Transformer 下篇：Transformer 训练与评估](https://zhuanlan.zhihu.com/p/97451231)
14. [从 0 开始学习 Transformer 番外：Transformer 如何穿梭时空？](https://gitee.com/LilithSangreal/LilithSangreal-Blog/blob/master/NLP/Transformer3.md)
15. [从 0 开始学习 Transformer 拾遗：文章本身的与解释](https://gitee.com/LilithSangreal/LilithSangreal-Blog/blob/master/NLP/Transformer4.md)
