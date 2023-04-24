# sequential learning interpretation

Deep learning, and in particular, recurrent neural networks, has in recent years gained nonstop interest with its successful application in a broad range of areas. These include handwriting recognition, natural language processing, speed recognition and so on. However, with the ever expanding use of such models, their interpretability or the mechanism of their decision making process have been understudied. Such interpretation can not only help users trust the models and predictions more, but also provide valuable insights into various areas, such as genetic modeling and linguistics, and help with model designs.

Here, we organized papers and articles from difference sources to provide a somewhat full-around overview of developments in this area.

## directions in sequential learning interpretation

### definition

- [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/pdf/1702.08608.pdf) (arXiv, 2017)\
        **keywords**: definition of interpretability, incompleteness, taxonomy of evaluation, latent dimensions

## to start with

### papers

- [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf?ref=morioh.com) (SIGKDD, 2016)\
        **keywords**: model agnostic, text and image, local approximation, LIME
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) (NIPS, 2017)\
        **keywords**: model agnostic, data agnostic, kernel SHAP, linear regression
- [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://arxiv.org/pdf/1406.1078.pdf?ref=https://codemonkey.link) (EMNLP, 2014)\
        **keywords**: RNN encoder-decoder, natural language, novel hidden unit
- [Explainable Artificial Intelligence (XAI) on Time Series Data: A Survey](https://arxiv.org/pdf/2104.00950.pdf) (arXiv, 2021)\
        **keywords**: CNN, RNN, explainable AI methods, time series, natural language, backpropagation-based methods, perturbation-based methods, attention, Symbolic Aggregate Approximation (SAX), Fuzzy Logic
- [Towards a Rigorous Evaluation of XAI Methods on Time Series](https://arxiv.org/pdf/1909.07082.pdf) (ICCVW, 2019)\
        **keywords**: model-agnostic, time series, perturbation and sequence based evaluation, SHAP, DeepLIFT, LRP, Saliency Map, LIME
- [Visualizing and understanding recurrent networks](https://arxiv.org/pdf/1506.02078.pdf?ref=https://codemonkey.link) (ICLR, 2016)\
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

### controversies

- [Attention is not Explanation](https://arxiv.org/pdf/1902.10186.pdf) (arXiv, 2019)\
        **keywords**: RNN, BiLSTM, binary text classification, question answering, feature importance, Kendall &#964; correlation, counterfactual attention weights, adversarial attention
- [Attention is not not Explanation](https://arxiv.org/pdf/1908.04626.pdf) (arXiv, 2019)\
        **keywords**: LSTM, binary text classification, uniform attention weights, model variance, MLP diagnostic tool, model-consistent adversarial training, TVD/JSD plots

## based on interpretation methods

### SHAP

- [Interpreting a Recurrent Neural Network Predictions of ICU Mortality Risk](https://arxiv.org/pdf/1905.09865.pdf) (Journal of Biomedical Informatics, 2021)\
        **keywords**: LSTM, dt-patient-matrix, Learned Binary Masks (LBM), KernelSHAP
- [TimeSHAP: Explaining recurrent models through sequence perturbations](https://arxiv.org/pdf/2012.00073.pdf) (SIGKDD, 2021)\
        **keywords**: model-agnostic recurrent explainer based on KernelSHAP, feature/event/cell wise explanation, pruning method by grouping older events

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
- [Show Me What You’re Looking For: Visualizing Abstracted Transformer Attention for Enhancing Their Local Interpretability on Time Series Data](https://martin.atzmueller.net/paper/VisualizingAbstractedTransformerAttentionLocalInterpretability-SchwenkeAtzmueller-2021-preprint.pdf) (FLAIRS, 2021)\
        **keywords**: Transformer, Synthetic Control Chart, ECG5000, attention, data abstraction, Symbolic Aggregate Approximation (SAX), according visualization
- [Focusing on What is Relevant: Time-Series Learning and Understanding using Attention](https://arxiv.org/pdf/1806.08523.pdf) (ICPR, 2018)\
        **keywords**: temporal contextual layer, time series, motion capture, key frame detection, action classification
- [Spatiotemporal Attention for Multivariate Time Series Prediction and Interpretation](https://ieeexplore.ieee.org/document/9413914) (ICASSP, 2021)\
        **keywords**: spatial interpretation, spatiotemporal attention mechanism
- [Uncertainty-Aware Attention for Reliable Interpretation and Prediction](https://proceedings.neurips.cc/paper/2018/file/285e19f20beded7d215102b49d5c09a0-Paper.pdf) (NIPS, 2018)\
        **keywords**: RNN, risk prediction, attention, variational inference
- [Topological Attention for Time Series Forecasting](https://proceedings.neurips.cc/paper_files/paper/2021/file/d062f3e278a1fbba2303ff5a22e8c75e-Paper.pdf) (NIPS, 2021)\
        **keywords**: N-BEATS, univariate time series data, M4 dataset, topological attention
- [RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism](https://proceedings.neurips.cc/paper_files/paper/2016/file/231141b34c82aa95e48810a9d1b33a79-Paper.pdf) (NIPS, 2016)\
        **keywords**: RNN, Electronic Health Records, reverse time order
- [Preserving Dynamic Attention for Long-Term Spatial-Temporal Prediction](https://dl.acm.org/doi/pdf/10.1145/3394486.3403046) (KDD, 2020)\
        **keywords**: CNN, crowd flow prediction, service utilization prediction, Dynamic Switch-Attention Network (DSAN), Multi-Space Attention (MSA)
- [Attention based multi-modal new product sales time-series forecasting](https://dl.acm.org/doi/pdf/10.1145/3394486.3403362) (KDD, 2020)\
        **keywords**: multi-modal encoder-decoder, sales, self-attention
- [Non-stationary Time-aware Kernelized Attention for Temporal Event Prediction](https://dl.acm.org/doi/pdf/10.1145/3534678.3539470) (KDD, 2022)\
        **keywords**: Kernelized attention, Electricity Transformer Temperature, PM2.5, Generalized Spectral Mixture Kernel (GSMK) 

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

### prototypes

- [Interpretable and steerable sequence learning via prototypes](https://arxiv.org/pdf/1907.09728.pdf) (SIGKDD, 2019)\
        **keywords**: prototype sequence network, criteria for explainable prototypes, refining with user knowledge by creating/updating/deleting prototypes

### interpretable model

- [Electric Energy Consumption Prediction by Deep Learning with State Explainable Autoencoder](https://www.mdpi.com/1996-1073/12/4/739/htm) (Energies, 2019)\
        **keywords**: LSTM, projector and predictor, energy consumption prediction, state transition, t-SNE algorithm
- [Explaining Deep Classification of Time-Series Data with Learned Prototypes](https://arxiv.org/pdf/1904.08935.pdf) (CEUR, 2019)\
        **keywords**: autoencoder-prototype, 2-D time series, ECG or respiration or speech waveforms, prototype diversity and robustness
- [Explainable Tensorized Neural Ordinary Differential Equations for Arbitrary-step Time Series Prediction](https://arxiv.org/pdf/2011.13174.pdf) (IEEE Transactions on Knowledge and Data Engineering, 2022)\
        **keywords**: ETN-ODE, tensorized GRU, multivariate time series, tandem attention, arbitrary-step prediction, multi-step prediction
- [TSXplain: Demystification of DNN Decisions for Time-Series using Natural Language and Statistical Features](https://arxiv.org/pdf/1905.06175.pdf) (ICANN, 2019)\
        **keywords**: model-agnostic, time series, textual explanation, statistical feature extraction, anomaly detection
- [Multilevel wavelet decomposition network for interpretable time series analysis](https://arxiv.org/pdf/1806.08946.pdf) (SIGKDD, 2018)\
        **keywords**: time series forecasting, multi-frequency LSTM, decomposition into small sub-series, importance score of middle layer
- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/pdf/1905.10437.pdf) (ICLR, 2020)\
        **keywords**: fully-connected layers with doubly residual stacking, interpretable architecture with trend or seasonality model
- [Exploring interpretable LSTM neural networks over multi-variable data](https://arxiv.org/pdf/1905.12034.pdf) (ICML, 2019)\
        **keywords**: interpretable multi-variable LSTM, mixture attention mechanism, training method to learn network parameter and variable/temporal importance

### adversarial training

- [Explainability and Adversarial Robustness for RNNs](https://arxiv.org/pdf/1912.09855.pdf) (BigDataService, 2020)\
        **keywords**: LSTM, network packet flows, adversarial robustness, feature sensitivity, Partial Dependence Plot (PDP), adversarial training
- [Adversarial Detection with Model Interpretation](https://people.engr.tamu.edu/xiahu/papers/kdd18liu.pdf) (SIGKDD, 2018)\
        **keywords**: model-agnostic, Twitter/YelpReview dataset, evasion-prone sample selection, local interpretation, defensive distillation, adversarial training

### counterfactual

- [Counterfactual Explanations for Machine Learning on Multivariate Time Series Data](https://arxiv.org/pdf/2008.10781.pdf) (ICAPAI, 2021)\
        **keywords**: model-agnostic, multivariate time series, HPC system telemetry datasets, heuristic algorithm, measuring good explanation
- [Instance-based Counterfactual Explanations for Time Series Classification](https://arxiv.org/pdf/2009.13211.pdf) (ICCBR, 2021)\
        **keywords**: model-agnostic, time series, UCR archive, properties of good counterfactuals, Native Guide method, w-counterfactual, NUN-CF

## interpretation evaluation metrics

- [Don’t Get Me Wrong: How to apply Deep Visual Interpretations to Time Series](https://arxiv.org/pdf/2203.07861.pdf) (arXiv, 2022)\
        **keywords**: gradient- or perturbation-based post-hoc visual interpretation, sanity, faithfulness, sensitivity, robustness, stability, localization

- [Evaluation of interpretability methods for multivariate time series forecasting](https://link.springer.com/article/10.1007/s10489-021-02662-2)(Applied Intelligence, 2021)\
        **keywords**: time series forcasting, Area Over the Perturbation Curve, Ablation Percentage Threshold, local fidelity, local explanation

- [Validation of XAI explanations for multivariate time series classification in the maritime domain](https://www.sciencedirect.com/science/article/abs/pii/S1877750321001976)(Journal of Computational Science, 2022)\
        **keywords**: LIME, time-slice mapping, SHAP, Path Integrated Gradient, heatmap, perturbation, sequence analysis, noval evaluation technique
