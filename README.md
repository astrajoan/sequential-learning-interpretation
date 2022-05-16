# sequential learning interpretation

Deep learning, and in particular, recurrent neural networks, has in recent years gained nonstop interest with its successful application in a broad range of areas. These include handwriting recognition, natural language processing, speed recognition and so on. However, with the ever expanding use of such models, their interpretability or the mechanism of their decision making process have been understudied. Such interpretation can not only help users trust the models and predictions more, but also provide valuable insights into various areas, such as genetic modeling and linguistics, and help with model designs.

Here, we organized papers and articles from difference sources to provide a somewhat full-around overview of developments in this area.

## directions in sequential learning interpretation

### definition

- [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/pdf/1702.08608.pdf) (arXiv, 2017)\
        **keywords**: definition of interpretability, incompleteness, taxonomy of evaluation, latent dimensions

## to start with

### papers

- [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf?ref=morioh.com) (SIGKDD 2016)\
        **keywords**: model agnostic, text and image, local approximation, LIME
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) (NIPS, 2017)\
        **keywords**: model agnostic, data agnostic, MNIST, SHAP
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf?ref=https://codemonkey.link) (EMNLP, 2014)\
        **keywords**: RNN encoder-decoder, natural language, novel hidden unit
- [Explainable Artificial Intelligence (XAI) on Time Series Data: A Survey](https://arxiv.org/pdf/2104.00950.pdf) (arXiv, 2021)\
        **keywords**: CNN, RNN, explainable AI methods, time series, natural language, backpropagation-based methods, perturbation-based methods, attention, Symbolic Aggregate Approximation (SAX), Fuzzy Logic
- [Visualizing and understanding recurrent networks](https://arxiv.org/pdf/1506.02078.pdf?ref=https://codemonkey.link) (ICLR 2016)\
        **keywords**: LSTM, natural language, revealed cells that identify interpretable and high-level patterns, long-range dependency, error analysis
- [Techniques for Interpretable Machine Learning](https://arxiv.org/pdf/1808.00033.pdf) (CACM, 2019)\
        **keywords**: overview of different models and interpretation techniques
- [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0130140&type=printable&ref=https://githubhelp.com) (PloS one, 2015)\
        **keywords**: nonlinear classifiers, image, Layer-Wise Relevance Propagation
- [Benchmarking Deep Learning Interpretability in Time Series Predictions](https://proceedings.neurips.cc/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf) (NIPS, 2020)\
        **keywords**: RNN, Temporal Convolutional Networks, Transformers, synthetic time series data, saliency-based interpretability methods, two-step temporal saliency rescaling (TSR)

### articles

- [Interpreting recurrent neural networks on multivariate time series](https://towardsdatascience.com/interpreting-recurrent-neural-networks-on-multivariate-time-series-ebec0edb8f5a)\
        **keywords**: RNN, multivariate time series, SHAP, instance importance, efficiency


## based on interpretation methods

### SHAP

- [Interpreting a Recurrent Neural Network’s Predictions of ICU Mortality Risk](https://arxiv.org/pdf/1905.09865.pdf) (Journal of Biomedical Informatics, 2021)\
        **keywords**: LSTM, dt-patient-matrix, Learned Binary Masks (LBM), KernelSHAP

### composition

- [On Attribution of Recurrent Neural Network Predictions via Additive Decomposition](https://arxiv.org/pdf/1903.11245.pdf) (WWW, 2019)\
        **keywords**: LSTM, GRU, Bidirectional GRU, sentiment text, Stanford Sentiment Treebank 2 (SST2), Yelp Polarity (Yelp), decomposition, REAT
- [Visualizing and Understanding Neural Models in NLP](https://arxiv.org/pdf/1506.01066.pdf) (NAACL-HLT, 2016)\
        **keywords**: RNN, LSTM, Bidirectional LSTM, sentiment text, Stanford Sentiment Treebank, compositionality, unit salience


### gradient

- [Interpretation of Prediction Models Using the Input Gradient](https://arxiv.org/pdf/1611.07634.pdf?ref=https://githubhelp.com) (arXiv, 2016)\
        **keywords**: model agnostic, Bag of Words, gradient


### backpropagation

- [Explaining Recurrent Neural Network Predictions in Sentiment Analysis](https://arxiv.org/pdf/1706.07206.pdf) (EMNLP, 2017)\
        **keywords**: LSTM,  bidirectional LSTM, sentiment text, Stanford Sentiment Treebank, Layer-wise Relevance Propagation (LRP)

### attention

- [Interpretability of time-series deep learning models: A study in cardiovascular patients admitted to Intensive care unit](https://www.sciencedirect.com/science/article/pii/S1532046421002057) (Journal of Biomedical Informatics, 2021)\
        **keywords**: LSTM, EHRs data-stream, attention, activation maps
- [Show Me What You’re Looking For: Visualizing Abstracted Transformer Attention for Enhancing Their Local Interpretability on Time Series Data](https://martin.atzmueller.net/paper/VisualizingAbstractedTransformerAttentionLocalInterpretability-SchwenkeAtzmueller-2021-preprint.pdf) (The International FLAIRS Conference Proceedings, 2021)\
        **keywords**: Transformer, Synthetic Control Chart, ECG5000, attention, data abstraction, Symbolic Aggregate Approximation (SAX), according visualization
- [Focusing on What is Relevant: Time-Series Learning and Understanding using Attention](https://arxiv.org/pdf/1806.08523.pdf) (ICPR, 2018)\
        **keywords**: temporal contextual layer, time series, motion capture, key frame detection, action classification

### saliency

- [Benchmarking Deep Learning Interpretability in Time Series Predictions](https://proceedings.neurips.cc/paper/2020/file/47a3893cc405396a5c30d91320572d6d-Paper.pdf) (NIPS, 2020)\
        **keywords**: RNN, Temporal Convolutional Networks, Transformers, synthetic time series data, saliency-based interpretability methods, two-step temporal saliency, rescaling (TSR)
- [Two Birds with One Stone: Series Saliency for Accurate and Interpretable Multivariate Time Series Forecasting](https://www.ijcai.org/proceedings/2021/0397.pdf) (IJCAI, 2021)\
        **keywords**: model agnostic, time series, electricity, air quality, industry data, series saliency
- [Series Saliency: Temporal Interpretation for Multivariate Time Series Forecasting](https://arxiv.org/pdf/2012.09324.pdf) (arXiv, 2020)\
        **keywords**: model agnostic, series saliency, multivariate time series, temporal feature importance, heatmap visualization

### erasure

- [Understanding Neural Networks through Representation Erasure](https://arxiv.org/pdf/1612.08220.pdf?ref=https://githubhelp.com) (arXiv, 2016)\
        **keywords**: Bi-LSTM, Uni-LSTM, RNN, natural language, lexical, sentiment, document, computing impact of erasure on evaluation metrics, reinforcement learning, erase minimum set of input words to flip a decision

### interpretable model

- [Electric Energy Consumption Prediction by Deep Learning with State Explainable Autoencoder](https://www.mdpi.com/1996-1073/12/4/739/htm) (Energies, 2019)\
        **keywords**: LSTM, projector and predictor, energy consumption prediction, state transition, t-SNE algorithm
- [Explaining Deep Classification of Time-Series Data with Learned Prototypes](https://arxiv.org/pdf/1904.08935.pdf) (CEUR workshop proceedings, 2019)\
        **keywords**: autoencoder-prototype, 2-D time series, ECG or respiration or speech waveforms, prototype diversity and robustness
- [Explainable Tensorized Neural Ordinary Differential Equations for Arbitrary-step Time Series Prediction](https://arxiv.org/pdf/2011.13174.pdf) (IEEE Transactions on Knowledge and Data Engineering, 2022)\
        **keywords**: ETN-ODE, tensorized GRU, multivariate time series, tandem attention, arbitrary-step prediction, multi-step prediction

## interpretation evaluation metrics

- [Don’t Get Me Wrong: How to apply Deep Visual Interpretations to Time Series](https://arxiv.org/pdf/2203.07861.pdf) (arXiv, 2022)\
        **keywords**: gradient- or perturbation-based post-hoc visual interpretation, sanity, faithfulness, sensitivity, robustness, stability, localization
