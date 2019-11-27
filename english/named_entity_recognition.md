# Named entity recognition

Named entity recognition (NER) adalah tugas memberi tag entitas dalam teks dengan tipe yang sesuai.
Pendekatan biasanya menggunakan notasi BIO, yang membedakan awal (B) dan bagian dalam (I) entitas.
O digunakan untuk token non-entitas.

Example:

| Mark | Watney | visited | Mars |
| --- | ---| --- | --- |
| B-PER | I-PER | O | B-LOC |

### CoNLL 2003 (English)

[CoNLL 2003 NER task](http://www.aclweb.org/anthology/W03-0419.pdf) terdiri dari teks berita baru dari Reuters RCV1
corpus ditandai dengan empat jenis entitas yang berbeda (PER, LOC, ORG, MISC). Model dievaluasi berdasarkan F1 berbasis bentang pada set tes. ♦ menggunakan kereta dan pemisahan pengembangan untuk pelatihan.

| Model           | F1  |  Paper / Source | Code |
| ------------- | :-----:| --- | --- |
| CNN Large + fine-tune (Baevski et al., 2019) | 93.5 | [Cloze-driven Pretraining of Self-attention Networks](https://arxiv.org/pdf/1903.07785.pdf) | |
| Flair embeddings (Akbik et al., 2018)♦ | 93.09 | [Contextual String Embeddings for Sequence Labeling](https://drive.google.com/file/d/17yVpFA7MmXaQFTe-HDpZuqw9fJlmzg56/view) | [Flair framework](https://github.com/zalandoresearch/flair)
| BERT Large (Devlin et al., 2018) | 92.8 | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | |
| CVT + Multi-Task (Clark et al., 2018) | 92.61 | [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/abs/1809.08370) | [Official](https://github.com/tensorflow/models/tree/master/research/cvt_text) |
| BERT Base (Devlin et al., 2018) | 92.4 | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | |
| BiLSTM-CRF+ELMo (Peters et al., 2018) | 92.22 | [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) | [AllenNLP Project](https://allennlp.org/elmo) [AllenNLP GitHub](https://github.com/allenai/allennlp) |
| Peters et al. (2017) ♦| 91.93 | [Semi-supervised sequence tagging with bidirectional language models](https://arxiv.org/abs/1705.00108) | |
| CRF + AutoEncoder (Wu et al., 2018) | 91.87 | [Evaluating the Utility of Hand-crafted Features in Sequence Labelling](http://aclweb.org/anthology/D18-1310) | [Official](https://github.com/minghao-wu/CRF-AE) | 
| Bi-LSTM-CRF + Lexical Features (Ghaddar and Langlais 2018) | 91.73 | [Robust Lexical Features for Improved Neural Network Named-Entity Recognition](https://arxiv.org/pdf/1806.03489.pdf) | [Official](https://github.com/ghaddarAbs/NER-with-LS) |
| Chiu and Nichols (2016) ♦| 91.62 | [Named entity recognition with bidirectional LSTM-CNNs](https://arxiv.org/abs/1511.08308) | |
| HSCRF (Ye and Ling, 2018)| 91.38 | [Hybrid semi-Markov CRF for Neural Sequence Labeling](http://aclweb.org/anthology/P18-2038) | [HSCRF](https://github.com/ZhixiuYe/HSCRF-pytorch) |
| IXA pipes (Agerri and Rigau 2016) | 91.36 | [Robust multilingual Named Entity Recognition with shallow semi-supervised features](https://doi.org/10.1016/j.artint.2016.05.003)| [Official](https://github.com/ixa-ehu/ixa-pipe-nerc)|
| NCRF++ (Yang and Zhang, 2018)| 91.35 | [NCRF++: An Open-source Neural Sequence Labeling Toolkit](http://www.aclweb.org/anthology/P18-4013) | [NCRF++](https://github.com/jiesutd/NCRFpp) |
| LM-LSTM-CRF (Liu et al., 2018)| 91.24 | [Empowering Character-aware Sequence Labeling with Task-Aware Neural Language Model](https://arxiv.org/pdf/1709.04109.pdf) | [LM-LSTM-CRF](https://github.com/LiyuanLucasLiu/LM-LSTM-CRF) |
| Yang et al. (2017) ♦| 91.26 | [Transfer Learning for Sequence Tagging with Hierarchical Recurrent Networks](https://arxiv.org/abs/1703.06345) | |
| Ma and Hovy (2016) | 91.21 | [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354) | |
| LSTM-CRF (Lample et al., 2016) | 90.94 | [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) | |

### Long-tail emerging entities

[WNUT 2017 Emerging Entities task](http://aclweb.org/anthology/W17-4418) beroperasi dalam berbagai bahasa Inggris
teks dan fokus pada generalisasi di luar penghafalan dalam lingkungan varians tinggi. Skor diberikan keduanya
instance potongan entitas, dan bentuk permukaan entitas unik, untuk menormalkan dampak bias entitas yang sering terjadi.

| Feature | Train | Dev | Test |
| --- | --- | --- | --- |
| Posts | 3,395 | 1,009 | 1,287 |
| Tokens | 62,729 | 15,733 | 23,394 |
| NE tokens | 3,160 | 1,250 | 1,589 |

Data tersebut dianotasi untuk enam kelas - orang, lokasi, kelompok, karya kreatif, produk, dan perusahaan.

Links: [WNUT 2017 Emerging Entity task page](https://noisy-text.github.io/2017/emerging-rare-entities.html)(termasuk link unduhan langsung untuk data dan skrip penilaian)

| Model         | F1  | F1 (surface form) |  Paper / Source |
| ---           | --- | ---               | --- |
| Flair embeddings (Akbik et al., 2018) | 49.59 | | [Pooled Contextualized Embeddings for Named Entity Recognition](http://alanakbik.github.io/papers/naacl2019_embeddings.pdf) / [Flair framework](https://github.com/zalandoresearch/flair) |
| Aguilar et al. (2018) | 45.55 | | [Modeling Noisiness to Recognize Named Entities using Multitask Neural Networks on Social Media](http://aclweb.org/anthology/N18-1127.pdf) |
| SpinningBytes | 40.78 | 39.33 | [Transfer Learning and Sentence Level Features for Named Entity Recognition on Tweets](http://aclweb.org/anthology/W17-4422.pdf) | 

### Ontonotes v5 (English)

[Ontonotes corpus v5](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf) adalah corpus kaya beranotasi dengan beberapa lapisan anotasi, termasuk entitas bernama, coreference, bagian dari pidato, pengertian kata, proposisi, dan pohon parse sintaksis. Anotasi ini adalah lebih dari sejumlah besar token, bagian lintas domain yang luas, dan 3 bahasa (Inggris, Arab, dan Cina). Dataset NER (yang menarik di sini) mencakup 18 tag, terdiri dari 11 _types_ (PERSON, ORGANISASI, dll) dan 7 _values_ (DATE, PERCENT, dll), dan berisi 2 juta token. Dataset umum yang digunakan dalam NER didefinisikan dalam [Pradhan et al 2013](https://www.semanticscholar.org/paper/Towards-Robust-Linguistic-Analysis-using-OntoNotes-Pradhan-Moschitti/a94e4fe6f475e047be5dcc9077f445e496240852) dan dapat ditemukan [di sini](http://cemantix.org/data/ontonotes.html).

| Model           | F1  |  Paper / Source | Code |
| ------------- | :-----:| --- | --- |
| Flair embeddings (Akbik et al., 2018) | 89.71 | [Contextual String Embeddings for Sequence Labeling](http://aclweb.org/anthology/C18-1139) | [Official](https://github.com/zalandoresearch/flair) |
| CVT + Multi-Task (Clark et al., 2018) | 88.81 | [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/abs/1809.08370)  | [Official](https://github.com/tensorflow/models/tree/master/research/cvt_text) |
| Bi-LSTM-CRF + Lexical Features (Ghaddar and Langlais 2018) | 87.95 | [Robust Lexical Features for Improved Neural Network Named-Entity Recognition](https://arxiv.org/pdf/1806.03489.pdf) | [Official](https://github.com/ghaddarAbs/NER-with-LS)|
| BiLSTM-CRF (Strubell et al, 2017) | 86.99 | [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/pdf/1702.02098.pdf)  | [Official](https://github.com/iesl/dilated-cnn-ner) |
| Iterated Dilated CNN (Strubell et al, 2017) | 86.84 | [Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/pdf/1702.02098.pdf)  | [Official](https://github.com/iesl/dilated-cnn-ner) |
| Chiu and Nichols (2016) | 86.28 | [Named entity recognition with bidirectional LSTM-CNNs](https://arxiv.org/abs/1511.08308) | |
| Joint Model (Durrett and Klein 2014) | 84.04 | [A Joint Model for Entity Analysis: Coreference, Typing, and Linking](https://pdfs.semanticscholar.org/2eaf/f2205c56378e715d8d12c521d045c0756a76.pdf) |
| Averaged Perceptron (Ratinov and Roth 2009) | 83.45 | [Design Challenges and Misconceptions in Named Entity Recognition](https://www.semanticscholar.org/paper/Design-Challenges-and-Misconceptions-in-Named-Ratinov-Roth/27496a2ee337db705e7c611dea1fd8e6f41437c2) (These scores reported in ([Durrett and Klein 2014](https://pdfs.semanticscholar.org/2eaf/f2205c56378e715d8d12c521d045c0756a76.pdf))) | [Official](https://github.com/CogComp/cogcomp-nlp/tree/master/ner) |



[Go back to the README](../README.md)
