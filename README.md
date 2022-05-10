# sequential learning interpretation

Deep learning, and in particular, recurrent neural networks, has in recent years gained nonstop interest with its successful application in a broad range of areas. These include handwriting recognition, natural language processing, speed recognition and so on. However, with the ever expanding use of such models, their interpretability or the mechanism of their decision making process have been understudied. Such interpretation can not only help users trust the models and predictions more, but also provide valuable insights into various areas, such as genetic modeling and linguistics, and help with model designs.

Here, we organized papers and articles from difference sources to provide a somewhat full-around overview of developments in this area.

## directions in sequential learning interpretation

### definition

- [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/pdf/1702.08608.pdf)\
        **keywords**: definition of interpretability, incompleteness, taxonomy of evaluation, latent dimensions

## to start with

### papers

- [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf?ref=morioh.com)\
        **keywords**: model agnostic, text and image, local approximation, LIME
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf)\
        **keywords**: model agnostic, data agnostic, MNIST, SHAP
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf?ref=https://codemonkey.link)\
        **keywords**: RNN encoder-decoder, natural language, novel hidden unit
- [Visualizing and understanding recurrent networks](https://arxiv.org/pdf/1506.02078.pdf?ref=https://codemonkey.link)\
        **keywords**: LSTM, natural language, revealed cells that identify interpretable and high-level patterns, long-range dependency, error analysis
- [Techniques for Interpretable Machine Learning](https://arxiv.org/pdf/1808.00033.pdf)\
        **keywords**: overview of different models and interpretation techniques
- [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable&ref=https://githubhelp.com)\
        **keywords**: nonlinear classifiers, image, Layer-Wise Relevance Propagation
- [Benchmarking Deep Learning Interpretability in Time Series Predictions](https://proceedings.neurips.cc/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf)\
        **keywords**: RNN, Temporal Convolutional Networks, Transformers, synthetic time series data, saliency-based interpretability methods, two-step temporal saliency
rescaling (TSR)


### articles

- [Interpreting recurrent neural networks on multivariate time series](https://towardsdatascience.com/interpreting-recurrent-neural-networks-on-multivariate-time-series-ebec0edb8f5a)\
        **keywords**: RNN, multivariate time series, SHAP, instance importance, efficiency


## based on interpretation methods

### SHAP

- [Interpreting a Recurrent Neural Network’s Predictions of ICU Mortality Risk](https://arxiv.org/pdf/1905.09865.pdf)\
        **keywords**: LSTM, dt-patient-matrix, Learned Binary Masks (LBM), KernelSHAP



### composition

- [On Attribution of Recurrent Neural Network Predictions via Additive Decomposition](https://arxiv.org/pdf/1903.11245.pdf)\
        **keywords**: LSTM, GRU, Bidirectional GRU, sentiment text, Stanford Sentiment Treebank 2 (SST2), Yelp Polarity (Yelp), decomposition, REAT
- [Visualizing and Understanding Neural Models in NLP](https://arxiv.org/pdf/1506.01066.pdf)\
        **keywords**: RNN, LSTM, Bidirectional LSTM, sentiment text, Stanford Sentiment Treebank, compositionality, unit salience


### gradient

- [Interpretation of Prediction Models Using the Input Gradient](https://arxiv.org/pdf/1611.07634.pdf?ref=https://githubhelp.com)\
        **keywords**: model agnostic, Bag of Words, gradient


### sparse constraint


### backpropagation

- [Explaining Recurrent Neural Network Predictions in Sentiment Analysis](https://arxiv.org/pdf/1706.07206.pdf)\
        **keywords**: LSTM,  bidirectional LSTM, sentiment text, Stanford Sentiment Treebank, Layer-wise Relevance Propagation (LRP)


### attention

- [Interpretability of time-series deep learning models: A study in cardiovascular patients admitted to Intensive care unit](https://www.sciencedirect.com/science/article/pii/S1532046421002057)\
        **keywords**: LSTM, EHRs data-stream, attention, activation maps

