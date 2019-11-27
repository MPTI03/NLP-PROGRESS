# Sentiment analysis

Sentiment analysis adalah tugas mengklasifikasikan polaritas teks yang diberikan.

### IMDb

[IMDb dataset](https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf) adalah biner
dataset analisis sentimen yang terdiri dari 50.000 ulasan dari Internet Movie Database (IMDb) berlabel positif atau
negatif. Dataset berisi bahkan jumlah tinjauan positif dan negatif. Hanya ulasan yang sangat terpolarisasi yang dipertimbangkan.
Ulasan negatif memiliki skor ≤ 4 dari 10, dan ulasan positif memiliki skor ≥ 7 dari 10. Tidak lebih dari 30 ulasan
termasuk per film. Model dievaluasi berdasarkan akurasi.

| Model           | Accuracy  |  Paper / Source |
| ------------- | :-----:| --- |
| XLNet (Yang et al., 2019) | 96.21 | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) |
| BERT_large+ITPT (Sun et al., 2019) | 95.79 | [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) |
| BERT_base+ITPT (Sun et al., 2019) | 95.63 | [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) |
| ULMFiT (Howard and Ruder, 2018) | 95.4 | [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) |
| Block-sparse LSTM (Gray et al., 2017) | 94.99 | [GPU Kernels for Block-Sparse Weights](https://s3-us-west-2.amazonaws.com/openai-assets/blocksparse/blocksparsepaper.pdf) |
| oh-LSTM (Johnson and Zhang, 2016) | 94.1 | [Supervised and Semi-Supervised Text Categorization using LSTM for Region Embeddings](https://arxiv.org/abs/1602.02373) |
| Virtual adversarial training (Miyato et al., 2016) | 94.1 | [Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725) |
| BCN+Char+CoVe (McCann et al., 2017) | 91.8 | [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107) |

### SST

[Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) 
berisi 215.154 frasa dengan label sentimen berbutir halus di pohon parse
dari 11.855 kalimat dalam ulasan film. Model dievaluasi baik pada butiran halus
(lima arah) atau klasifikasi biner berdasarkan akurasi.

Fine-grained klasifikasi (SST-5, 94,2k examples):

| Model           | Accuracy |  Paper / Source |
| ------------- | :-----:| --- |
| EDD-LG shared (Patro et al., 2018) | 64.4 | [Learning Semantic Sentence Embeddings using Pair-wise Discriminator](https://arxiv.org/pdf/1806.00807.pdf) |
| BCN+Suffix BiLSTM-Tied+CoVe (Brahma, 2018) | 56.2 | [Improved Sentence Modeling using Suffix Bidirectional LSTM](https://arxiv.org/pdf/1805.07340v2.pdf) |
| BCN+ELMo (Peters et al., 2018) | 54.7 | [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) |
| BCN+Char+CoVe (McCann et al., 2017) | 53.7 | [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107) |

Binary klasifikasi (SST-2, 56.4k examples):

| Model           | Accuracy  |  Paper / Source | Code |
| ------------- | :-----:| --- | --- |
| XLNet-Large (ensemble) (Yang et al., 2019) | 96.8 | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) | [Official](https://github.com/zihangdai/xlnet/) |
| MT-DNN-ensemble (Liu et al., 2019) | 96.5 | [Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding](https://arxiv.org/pdf/1904.09482.pdf) | [Official](https://github.com/namisan/mt-dnn/) |
| Snorkel MeTaL(ensemble) (Ratner et al., 2018) | 96.2 | [Training Complex Models with Multi-Task Weak Supervision](https://arxiv.org/pdf/1810.02840.pdf) | [Official](https://github.com/HazyResearch/metal) |
| MT-DNN (Liu et al., 2019) | 95.6 | [Multi-Task Deep Neural Networks for Natural Language Understanding](https://arxiv.org/abs/1901.11504) | [Official](https://github.com/namisan/mt-dnn/) |
| Bidirectional Encoder Representations from Transformers (Devlin et al., 2018) | 94.9 | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | [Official](https://github.com/google-research/bert) |
| Block-sparse LSTM (Gray et al., 2017) | 93.2 | [GPU Kernels for Block-Sparse Weights](https://s3-us-west-2.amazonaws.com/openai-assets/blocksparse/blocksparsepaper.pdf) | [Offical](https://github.com/openai/blocksparse) |
| bmLSTM (Radford et al., 2017) | 91.8 | [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/abs/1704.01444) | [Unoffical](https://github.com/NVIDIA/sentiment-discovery) |
| Single layer bilstm distilled from BERT (Tang et al., 2019)| 90.7 |[Distilling Task-Specific Knowledge from BERT into Simple Neural Networks](https://arxiv.org/abs/1903.12136)|  |
| BCN+Char+CoVe (McCann et al., 2017) | 90.3 | [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107) | [Official](https://github.com/salesforce/cove) |
| Neural Semantic Encoder (Munkhdalai and Yu, 2017) | 89.7 | [Neural Semantic Encoders](http://www.aclweb.org/anthology/E17-1038) | |
| BLSTM-2DCNN (Zhou et al., 2017) | 89.5 | [Text Classification Improved by Integrating Bidirectional LSTM with Two-dimensional Max Pooling](http://www.aclweb.org/anthology/C16-1329) | |

### Yelp

[Yelp Review dataset](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
terdiri dari lebih dari 500.000 ulasan Yelp. Ada biner dan berbutir halus (kelas lima)
versi dataset. Model dievaluasi berdasarkan kesalahan (1 - akurasi; lebih rendah lebih baik).

Fine-grained klasifikasi : 

| Model           | Error  |  Paper / Source |
| ------------- | :-----:| --- |
| XLNet (Yang et al., 2019) | 27.80 | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) |
| BERT_large+ITPT (Sun et al., 2019) | 28.62 | [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) |
| BERT_base+ITPT (Sun et al., 2019) | 29.42 | [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) |
| ULMFiT (Howard and Ruder, 2018) | 29.98 | [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) |
| DPCNN (Johnson and Zhang, 2017) | 30.58 | [Deep Pyramid Convolutional Neural Networks for Text Categorization](http://aclweb.org/anthology/P17-1052) |
| CNN (Johnson and Zhang, 2016) | 32.39 | [Supervised and Semi-Supervised Text Categorization using LSTM for Region Embeddings](https://arxiv.org/abs/1602.02373) |
| Char-level CNN (Zhang et al., 2015) | 37.95 | [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) |

Binary klasifikasi:

| Model           | Error |  Paper / Source |
| ------------- | :-----:| --- |
| XLNet (Yang et al., 2019) | 1.55 | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) |
| BERT_large+ITPT (Sun et al., 2019) | 1.81 | [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) |
| BERT_base+ITPT (Sun et al., 2019) | 1.92 | [How to Fine-Tune BERT for Text Classification?](https://arxiv.org/pdf/1905.05583.pdf) |
| ULMFiT (Howard and Ruder, 2018) | 2.16 | [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) |
| DPCNN (Johnson and Zhang, 2017) | 2.64 | [Deep Pyramid Convolutional Neural Networks for Text Categorization](http://aclweb.org/anthology/P17-1052) |
| CNN (Johnson and Zhang, 2016) | 2.90 | [Supervised and Semi-Supervised Text Categorization using LSTM for Region Embeddings](https://arxiv.org/abs/1602.02373) |
| Char-level CNN (Zhang et al., 2015) | 4.88 | [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) |


### SemEval
SemEval (International Workshop on Semantic Evaluation) memiliki tugas khusus untuk analisis Sentimen.
Ikhtisar tahun terakhir dari tugas tersebut (Tugas 4) dapat diperoleh di: http://www.aclweb.org/anthology/S17-2088

SemEval-2017 Tugas 4 terdiri dari lima subtugas, masing-masing ditawarkan untuk bahasa Arab dan Inggris:

1.Subtask A: Diberikan tweet, putuskan apakah itu mengekspresikan POSITIF, NEGATIF, atau NETRAL
sentimen.

2. Subtask B: Diberikan tweet dan topik, mengklasifikasikan sentimen yang disampaikan ke sana
topik pada skala dua poin: POSITIF vs NEGATIF.

3. Subtask C: Diberikan tweet dan sebuah topik, mengklasifikasikan sentimen yang disampaikan dalam
tweet menuju topik itu pada skala lima poin: STRONGLYPOSITIVE, WEAKLYPOSITIVE,
NETRAL, LEMAH NEGATIF, dan SANGAT NEGATIF.

4. Subtask D: Diberikan satu set tweet tentang suatu topik, perkirakan distribusi tweet
di seluruh kelas POSITIF dan NEGATIF.

5. Subtask E: Given a set of tweets about a topic, estimate the distribution of tweets
across the five classes: STRONGLYPOSITIVE, WEAKLYPOSITIVE, NEUTRAL, WEAKLYNEGATIVE, and STRONGLYNEGATIVE.

Hasil subtask A  :

| Model           | F1-score |  Paper / Source |
| ------------- | :-----:| --- |
| LSTMs+CNNs ensemble with multiple conv. ops (Cliche. 2017) | 0.685 | [BB twtr at SemEval-2017 Task 4: Twitter Sentiment Analysis with CNNs and LSTMs](http://www.aclweb.org/anthology/S17-2094) |
| Deep Bi-LSTM+attention (Baziotis et al., 2017) | 0.677 | [DataStories at SemEval-2017 Task 4: Deep LSTM with Attention for Message-level and Topic-based Sentiment Analysis](http://aclweb.org/anthology/S17-2126) |


## Aspect-based sentiment analysis

### Sentihood

[Sentihood](http://www.aclweb.org/anthology/C16-1146) adalah dataset untuk analisis sentimen berbasis aspek yang ditargetkan (TABSA), yang bertujuan
untuk mengidentifikasi polaritas berbutir halus terhadap aspek tertentu. Dataset terdiri dari 5.215 kalimat,
3,862 di antaranya berisi target tunggal, dan sisanya beberapa target.

Dataset mirror: https://github.com/uclmr/jack/tree/master/data/sentihood

| Model           | Aspect (F1) | Sentiment (acc) |  Paper / Source |  Code |
| ------------- | :-----:| :-----:| --- | --- |
| Sun et al. (2019) | 87.9 | 93.6 | [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/pdf/1903.09588.pdf) | [Official](https://github.com/HSLCY/ABSA-BERT-pair)
| Liu et al. (2018) | 78.5 | 91.0 | [Recurrent Entity Networks with Delayed Memory Update for Targeted Aspect-based Sentiment Analysis](http://aclweb.org/anthology/N18-2045) | [Official](https://github.com/liufly/delayed-memory-update-entnet)
| SenticLSTM (Ma et al., 2018) | 78.2 | 89.3 | [Targeted Aspect-Based Sentiment Analysis via Embedding Commonsense Knowledge into an Attentive LSTM](http://sentic.net/sentic-lstm.pdf) | 
| LSTM-LOC (Saeidi et al., 2016) | 69.3 | 81.9 | [Sentihood: Targeted aspect based sentiment analysis dataset for urban neighbourhoods](http://www.aclweb.org/anthology/C16-1146) |

### SemEval-2014 Task 4

[SemEval-2014 Task 4](http://alt.qcri.org/semeval2014/task4/) berisi dua dataset khusus domain untuk laptop dan restoran, yang terdiri dari lebih dari 6 ribu kalimat dengan anotasi manusia tingkat aspek berbutir halus.

Tugas ini terdiri dari subtugas berikut:

- Subtask 1: Aspek istilah ekstraksi

- Subtask 2: Polaritas istilah aspek

- Subtask 3: Deteksi kategori aspek

- Subtask 4: Polaritas kategori aspek

Preprocessed dataset: https://github.com/songyouwei/ABSA-PyTorch/tree/master/datasets/semeval14   
https://github.com/howardhsu/BERT-for-RRC-ABSA (with both subtask 1 and subtask 2)

Subtask 1 results (SemEval-2014 Task 4 for Laptop and SemEval-2016 Task 5 for Restaurant):

| Model           | Laptop (F1) | Restaurant (F1) |  Paper / Source |  Code |
| ------------- | :-----:| :-----:| --- | --- |
| BERT-PT (Hu, Xu, et al., 2019) | 84.26 | 77.97 | [BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://arxiv.org/pdf/1904.02232.pdf) | [official](https://github.com/howardhsu/BERT-for-RRC-ABSA)
| DE-CNN (Hu, Xu, et al., 2018) | 81.59 | 74.37 | [Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction](https://www.aclweb.org/anthology/P18-2094) | [official](https://github.com/howardhsu/DE-CNN)
| MIN (Li, Xin, et al., 2017) | 77.58 | 73.44 | [Deep Multi-Task Learning for Aspect Term Extraction with Memory Interaction] | 
| RNCRF (Wang, Wenya. et al., 2016) | 78.42 | 69.74 | [Recursive Neural Conditional Random Fields for Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/papers/D/D16/D16-1059/) | [official](https://github.com/happywwy/Recursive-Neural-Conditional-Random-Field)

Subtask 2 results:

| Model           | Restaurant (acc) | Laptop (acc) |  Paper / Source |  Code |
| ------------- | :-----:| :-----:| --- | --- |
| BERT-ADA (Rietzler, Alexander, et al., 2019) | 87.89 | 80.23 | [Adapt or Get Left Behind: Domain Adaptation through BERT Language Model Finetuning for Aspect-Target Sentiment Classification](https://arxiv.org/pdf/1908.11860.pdf) | [official](https://github.com/deepopinion/domain-adapted-atsc)
| LCF-BERT (Zeng, Yang, et al., 2019) | 87.14 | 82.45 | [LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification](https://www.mdpi.com/2076-3417/9/16/3389/pdf) | [official](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/lcf_bert.py)
| BERT-PT (Hu, Xu, et al., 2019) | 84.95 | 78.07 | [BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis](https://arxiv.org/pdf/1904.02232.pdf) | [official](https://github.com/howardhsu/BERT-for-RRC-ABSA)
| AOA (Huang, Binxuan, et al., 2018) | 81.20 | 74.50 | [Aspect Level Sentiment Classification with Attention-over-Attention Neural Networks](https://arxiv.org/pdf/1804.06536.pdf) | [Link](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/aoa.py)
| TNet (Li, Xin, et al., 2018) | 80.79 | 76.01 | [Transformation Networks for Target-Oriented Sentiment Classification](http://aclweb.org/anthology/P18-1087) | [Official](https://github.com/lixin4ever/TNet) / [Link](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/tnet_lf.py)
| RAM (Chen, Peng, et al., 2017) | 80.23 | 74.49 | [Recurrent Attention Network on Memory for Aspect Sentiment Analysis](http://www.aclweb.org/anthology/D17-1047) | [Link](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/ram.py)
| MemNet (Tang, Duyu, et al., 2016) | 80.95 | 72.21 | [Aspect Level Sentiment Classification with Deep Memory Network](https://pdfs.semanticscholar.org/b28f/7e2996b6ee2784dd2dbb8212cfa0c79ba9e7.pdf) | [Official](https://drive.google.com/open?id=1Hc886aivHmIzwlawapzbpRdTfPoTyi1U) / [Link](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/memnet.py)
| IAN (Ma, Dehong, et al., 2017) | 78.60 | 72.10 | [Interactive Attention Networks for Aspect-Level Sentiment Classification](http://www.ijcai.org/proceedings/2017/0568.pdf) | [Link](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/ian.py)
| ATAE-LSTM (Wang, Yequan, et al. 2016) | 77.20 | 68.70 | [Attention-based lstm for aspect-level sentiment classification](https://aclweb.org/anthology/D16-1058) | [Link](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/atae_lstm.py)
| TD-LSTM (Tang, Duyu, et al., 2016) | 75.63 | 68.13 | [Effective LSTMs for Target-Dependent Sentiment Classification](https://www.aclweb.org/anthology/C/C16/C16-1311.pdf) | [Official](https://drive.google.com/open?id=17RF8MZs456ov9MDiUYZp0SCGL6LvBQl6) / [Link](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/td_lstm.py)

## Sentiment classification with user and product information

This is the same task on sentiment classification, where the given text is a review, but we are also additionally given (a) the *user* who wrote the text, and (b) the *product* which the text is written for. There are three widely used datasets, introduced by [Tang et. al (2015)](http://aclweb.org/anthology/P15-1098): IMDB, Yelp 2013, and Yelp 2014. Evaluation is done using both accuracy and RMSE, but for brevity, we only provide the accuracy here. Please look at the papers for the RMSE values.

| Model              | IMDB (acc) | Yelp 2013 (acc) | Yelp 2014 (acc) | Paper / Source | Code |
| ------------------ | :--------: | :-------------: | :-------------: | -------------- | ---- |
| BiLSTM+CHIM (Amplayo, 2019) | 56.4 | 67.8 | 69.2 | [Rethinking Attribute Representation and Injection for Sentiment Classification](https://arxiv.org/pdf/1908.09590.pdf) | [Link](https://github.com/rktamplayo/CHIM) |
| BiLSTM + linear-basis-cust (Kim, et al., 2019) | - | 67.1 | - | [Categorical Metadata Representation for Customized Text Classification](https://arxiv.org/pdf/1902.05196.pdf) | [Link](https://github.com/zizi1532/BasisCustomize) |
| CMA (Ma, et al., 2017) | 54.0 | 66.4 | 67.6 | [Cascading Multiway Attention for Document-level Sentiment Classification](http://aclweb.org/anthology/I17-1064) | - |
| DUPMN (Long, et al., 2018) | 53.9 | 66.2 | 67.6 | [Dual Memory Network Model for Biased Product Review Classification](http://aclweb.org/anthology/W18-6220) | - |
| HCSC (Amplayo, et al., 2018) | 54.2 | 65.7 | - | [Cold-Start Aware User and Product Attention for Sentiment Classification](http://aclweb.org/anthology/P18-1236) | [Link](https://github.com/rktamplayo/HCSC) |
| NSC (Chen, et al., 2016) | 53.3 | 65.0 | 66.7 | [Neural Sentiment Classification with User and Product Attention](http://aclweb.org/anthology/D16-1171) | [Link](https://github.com/thunlp/NSC) |
| UPDMN (Dou, 2017) | 46.5 | 63.9 | 61.3 | [Capturing User and Product Information for Document Level Sentiment Analysis with Deep Memory Network](http://aclweb.org/anthology/D17-1054) | - |
| UPNN (Tang, et al., 2016) | 43.5 | 59.6 | 60.8 | [Learning Semantic Representations of Users and Products for Document Level Sentiment Classification](http://aclweb.org/anthology/P15-1098) | [Link](http://ir.hit.edu.cn/~dytang/paper/acl2015/UserTextNN.zip) |

# Subjectivity analysis

Tugas terkait dengan analisis sentimen adalah analisis subjektivitas dengan tujuan memberi label opini sebagai subyektif atau objektif.

### SUBJ

[Subjectivity dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/) termasuk 5.000 kalimat subjektif dan 5.000 kalimat yang diproses secara objektif. 

| Model           | Accuracy |  Paper / Source |
| ------------- | :-----:| --- |
| AdaSent (Zhao et al., 2015) | 95.50 | [Self-Adaptive Hierarchical Sentence Model](https://arxiv.org/pdf/1504.05070.pdf) |
| CNN+MCFA (Amplayo et al., 2018) | 94.80 | [Translations as Additional Contexts for Sentence Classification](https://arxiv.org/abs/1806.05516) |
| Byte mLSTM (Radford et al., 2017) | 94.60 | [Learning to Generate Reviews and Discovering Sentiment](https://arxiv.org/pdf/1704.01444.pdf) |
| USE (Cer et al., 2018) | 93.90 | [Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf) |
| Fast Dropout (Wang and Manning, 2013) | 93.60 | [Fast Dropout Training](http://proceedings.mlr.press/v28/wang13a.pdf) |

[Kembali ke README](../README.md)
