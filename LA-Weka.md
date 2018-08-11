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

**Solution: **Assign different costs to different types of misclassification

### 不平衡数据

一个类的数据明显比其他类更加流行。

**解决方案：**为未分类的不同类型赋予不同的权值。



## Output



## Inferring Simple Rules



## Practical Data Mining



## Decision Tree Learning



## Mining Association Rules

