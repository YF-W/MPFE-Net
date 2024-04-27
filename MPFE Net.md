# A Feature Enhancement Network Based on Image Partitioning in a Multi-Branch Encoder-Decoder Environment

***The core feature of MPFE Net lies in the partitioning strategy applied to images. This mechanism enables the network to overcome the main issues of lack of attention to critical information and relatively sparse information transmission. We have implemented this partitioning strategy at various critical points in the model, allowing the network to fully achieve semantic partition decoding and effectively perform pixel classification and image restoration in the upsampling process.***

# Paper: MPFE Net（Multi-branch Partition Feature Enhancement Network）

**Authors : Yuefei Wanga, Yutong Zhanga, Li Zhanga, Yuxuan Wanb, Zhixuan Chenb, Yuquan Xub, Ronghui Fengb, Yixi Yangc, Xi Yubg **

## **1.Multi-branch Partition-guided Decoding Network（MPD Net)**



![image-20240427174453949](C:\Users\lp2e\AppData\Roaming\Typora\typora-user-images\image-20240427174453949.png)

***In the decoding stage of the baseline we proposed, the image is divided into four partitions, and these partitions are used as basic units for decoding and upsampling, and finally, these partitions are merged back into the original image. This idea constructs a "dual encoding - quad decoding" structural system, ensuring that the network focuses sufficiently on the original semantic information while ensuring that semantic transmission during decoding is most effective. Figure 2 shows the comparison results of the Dice index between the traditional baseline and the MPD Net (our baseline) on different datasets. On the dataset with seven lesion classes, MPD Net outperforms the traditional baseline on five classes, with an average performance improvement of 1.1307%, validating the correctness of the "partition" idea.***

## 2.Architecture Overview

![image-20240427174521726](C:\Users\lp2e\AppData\Roaming\Typora\typora-user-images\image-20240427174521726.png)

***This structure uses MPD Net as the basic network architecture, constructing a class of network foundation system called "dual encoder - multi decoder." This architecture is based on the basic mode of building a partition strategy for images at the decoding end, realizing a network system of "divide and conquer decoding." Additionally, on this basis, deep research has been conducted on the key link of semantic extraction, supplementing two important modules: MPIG and MFES ViT.***

## 3.Module 1:   Multi-semantic Progressive Interaction Guider（MPIG）

![image-20240427174539789](C:\Users\lp2e\AppData\Roaming\Typora\typora-user-images\image-20240427174539789.png)

***Our Bottleneck structure employs multi-branch techniques to process channel information, and avoids channel information imbalance through hierarchical interaction and fusion with the original input. In addition to receiving semantic information processed by CNN, we also integrate global information from ViT, enabling the model to comprehensively understand the image. Furthermore, we partition the image after feature extraction and pass it to the decoding end separately, achieving fine processing of local details while ensuring the full utilization of features in the transmission and integration process, thereby improving the performance and generalization ability of the model.***

## 4.Module 2:   Multi-branched Feature Enhancement with Shared ViT（MFES ViT）

![image-20240427174552004](C:\Users\lp2e\AppData\Roaming\Typora\typora-user-images\image-20240427174552004.png)

***We introduce convolution and pooling to extract local features, enhancing the understanding of details and improving translational invariance. By replacing linear layers with global average pooling, we aggregate spatial information in the image, achieving feature dimensionality reduction while preserving important local features, thus compensating for the shortcomings of linear layers.***

# **Datasets**:

1. LUNG Dataset: https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data
2. 2018 Data Science Bowl Dataset: https://www.kaggle.com/competitions/data-science-bowl-2018/data
3. MICCAI - Tooth Dataset: https://tianchi.aliyun.com/dataset/156596
4. MICCAI2015-CVC ClinicDB Dataset: https://polyp.grand-challenge.org/CVCClinicDB/
5. Skin Lesion Dataset: https://www.kaggle.com/datasets/ojaswipandey/skin-lesion-dataset
6.  ISIC 2017-Melanoma Dataset: https://challenge.isic-archive.com/data/#2017
7.  ISIC 2017-Nevus Dataset: https://challenge.isic-archive.com/data/#2017
8. ISIC2017-Seborrheic Keratosis Dataset: https://challenge.isic-archive.com/data/#2017

