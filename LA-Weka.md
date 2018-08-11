# LA-Weka

## Intro

### Data Mining

Identifying implicit, previously unknown, potentially useful patterns in data.

* Interactively or automatically done
* Descriptive or predictive

### 数据挖掘

找到不明的，之前未知的，且潜在有价值的数据模式。

* 交互式地或自动地完成
* 描述性或预测性



### Machine Learning

Programs that induce structural descriptions from observations.

* Supervised learning: Based on labeled examples & used for predicting labels of new observations.
* Unsupervised learning: Based on unlabeled data.

### 机器学习

通过程序从各观察中得到结构性描述。

* 监督学习：基于标记了的例子并用来为新观察预测其标记。
* 非监督学习：基于未标记的数据。



### Structural Description

`if-then` Rules

* Classification Rule: Predicts value of a given attribute
* Association Rule: Predicts value of arbitrary attribute

### 结构性描述

`if-then`规则

* 分类规则：预测已知的属性
* 关联规则：预测任意的属性



## Input

### Concept 概念

Structures that can be learned. 可用于学习的结构。

### Instance 实例 

Individual, independent examples of a concept, possibly corrupted by noise.

单独的，独立的概念例子，可能被噪音损坏。

### Attribute 属性

Measuring aspects of an instance. 实例中可供测量的方面。

### Concept Description 概念描述

Output of learning scheme. 学习方案的输出。



### Classification Learning 分类学习

Supervised, outcome is called class, success is subjectively measured.

监督学习，输出被称为类，成功的预测可被客观地测量。

### Association Learning 关联学习

Unsupervised, no class is specified, any interesting kinds of structure.

非监督学习，没有明确的类，任何有趣的结构都可以。

### Clustering 聚类

Unsupervised, find similar groups of items, success is subjectively measured.

非监督学习，寻找能够相似的样本成组，成功可以被客观地测量。

### Numeric Prediction 数值预测

Supervised, regression,  predict a numeric value rather than a discrete class

监督学习，回归，预测一个数值量而不是离散类



### Denormalization 反规范化

Process of flattening, any finite set of finite relations.

对一些存在有限关系的有限集合的一个平面化的处理过程。



### Attribute Types

* Nominal: Values are distinct symbol, no relation is implied, only equality test
* Ordinal: Impose order on values, no distance between data
* Interval: Numeric, fixed & equal unit, difference is accepted
* Ratio: Defined non-arbitrary zero point, multiplication & division are permitted

### 属性类型

* 名目：值为独特的符号，之间没有隐晦任何关系，只能平等地测试
* 序数：值间有顺序，但没有距离
* 区间：数值类型，固定和相等的单位，可以取差值
* 比率：定义了一个并不随便的零点，可以进行乘除



### ARFF File 文件

```ARFF
@relation relation-name
@attribute attribute-name-1 {nominal/ordinal}
@attribute attribute-name-2 numeric/string/date
@data
attribute-name-1, attribute-name-2
```

* Sparse Data: In some applications most attribute values in a data set are zero.

  ```ARFF
  0, x, 0, 0, 0, 0, y
  0, 0, 0, w, 0, 0, 0
  ```

  ARFF supports Sparse Data as:

  ```ARFF
  {1, x, 6, y}
  {3, w}
  ```



### Missing Value

* Indicated using `?` for unknown, unrecorded, irrelevant value.
* Missing value may have significance in itself.

### 缺失值

* 使用`?`来表示未知的，未记录的，不相关的值。
* 缺失值有时有重要意义。



### Unbalanced Data

One class is much more prevalent than the others. 

**Solution: ** Assign different costs to different types of misclassification

### 不平衡数据

一个类的数据明显比其他类更加流行。

**解决方案：** 为未分类的不同类型赋予不同的权值。



## Output

### Decision Table 决策表

The simplest way of representing the output of learning. 最简单的表达学习输出的方式。



### Linear Models for Regression 线性模型

Output is a sum of all attributes with its weights. 输出是各加权重属性求和。



### Decision Tree

Divide-and-conquer approach to processing the training data produces a decision tree.

1. Internal nodes in the tree test attributes
2. Attribute values are compared to constant
3. Comparing values of two attributes
4. Using a function of one or more attributes
5. Leaves assign classification or probability distribution to instances
6. To make prediction, instance with unknown class label is routed down the tree

### 决策树

使用分治法来训练数据以建立一个决策树。

1. 内部的节点用于树测试属性
2. 属性值与常数比较？
3. 比较两个属性值
4. 使用函数的一个或多个属性？
5. 为叶节点赋予分类或实例的概率分布
6. 预测时，有未知类的实例由根节点到叶节点决策



### Nominal and Numeric Attribute
* Nominal:
  1. One branch for every value, number of children is equal to number of attributes. 
  2. Just test once only
  3. (Alternative) Division into two subsets of nominal attribute values
* Numeric:
  1. Test whether value is greater or less than constant
  2. Test more than once
  3. (Alternative) Three-way split

### 名目和数值类属性

* 名目：
  1. 一个分支对应所有值，子节点的数目与属性数目一致。
  2. 只测试一遍。
  3. 另外的方法：分裂到两个名目子集
* 数值：
  1. 测试值是否大于或低于一个常量
  2. 测试多次
  3. 另外的方法：分裂成三个分支



### Missing Value

Absence of value have some significance?

* Yes! Missing should be a separate value
* No! Missing must be treated in a special way
  * Solution A: Assign instance to most popular branch
  * Solution B: Split instance into pieces with one piece per branch extending from node

### 缺失值

缺失值很重要吗？

* 是的！缺失值需要是分离的值。
* 不是！缺失值必须通过一个特殊的方法对待
  * 方法A：为最流行的分支分配缺失值的实例
  * 方法B：分裂实例，从节点一个分支衍生一部分缺失值的实例出来



### Classification Rule

Popular alternative to decision trees.

Antecedent (Pre-condition): A series of tests, logically ANDed together.

Consequent (Conclusion): Class or probability distribution assigned by rule, logically ORed together.

### 分类规则

另一种流行

先例 (先决条件)：一系列的测试，通常以`和`逻辑合并

后果 (结论)：类，或者概率分布，通常以`或`逻辑合并



### From Trees to Rules

One rule for each leaf.

* Antecedent contains a condition for every node on the path from root to leaf.
* Consequent is class assigned by the leaf.

Produces rules that are unambiguous. Rules can be unnecessarily complex which need pruning.

### 从树到规则

一个叶节点，一条规则。

* 先例包含了从根到叶的所有节点的条件。
* 结论是叶节点的类。

生成的规则并不模糊，规则没有必要那么复杂，所以需要修剪。



### From Rules to Trees

* More difficult than from trees to rules.
* Symmetry needs to be broken.
* Corresponding tree contains identical subtrees. `(Replicated Subtree Problem)`

### 从规则到树

* 比起从树到规则更难
* 对称需要被打破
* 相应的树包含了相同的子树 `重复子树问题`



### Nuggets of Knowledge

Question: Are rules independent pieces of knowledge?

Two ways of executing a rule set:

* Ordered set of rules (Decision list): Order is important for interpretation.
* Unordered set of rules: Rules may overlap and lead to different conclusions for the same instance.

### 知识块

问题：规则是独立的知识片段吗？

两种执行规则集的方式：

* 顺序的规则集：顺序对于解释极为重要。
* 无序的规则集：规则可能覆盖或者导致对于相同的实例产生不同的结论。



### Association Rules

* Predict any attribute and combinations of attributes.
* Not intended to be used together as a set.

Problem: Immense number of possible associations.

Solution: Restrict to show only the most predictive associations with high support and high confidence.

### 关联规则

* 预测任意属性或者属性集合
* 并不意味着被用作一个集合

问题：可能存在大量的可行关联规则。

解决方案： 限制只输出高支持度和高置信度的预测性关联规则。



### Support & Confidence

Support: Number of instances predicted correctly.

Confidence: Number of correct predictions, as proportion of all instances that rule applies to.

Normally minimum support and confidence are pre-specified.

### 支持度和置信度

支持度：正确预测的实例个数。

置信度：正确预测的实例的个数，占据总实例的比例。

通常最小支持度和置信度都是预先指定了的。



### Instance-Based Representation

Lazy learning, AKA Rote Learning

* Training instance are searched for instances which are most closely resemble test instance.
* The instances themselves represent the knowledge.

Methods: `Nearest-Neighbor`, `K-Nearest-Neighbor`

### 基于实例的表达

懒惰学习，又称死记硬背法学习

* 训练的实例由那些最接近于测试实例的实例构成。
* 实例本身就表达了知识。

方法：`Nearest-Neighbor`, `K-Nearest-Neighbor`

### 

### Distance Function

* Numeric: Euclidean distance with normalization
* Nominal: 1 - different, 0 - equal

### 距离函数

* 数值：规范的欧氏距离
* 名目：1 - 不同， 0 - 相同



## Inferring Simple Rules



## Practical Data Mining



## Decision Tree Learning



## Mining Association Rules

