
# UniFlow

[\[📂 GitHub\]](https://github.com/ZhengrongYue/UniFlow)
<!-- [\[📃 QLIP Tech Report\]](http://arxiv.org/abs/2502.05178) -->
<!-- [\[🔗 Project Page\]](http://nvlabs.github.io/QLIP/) -->
[\[🤗 HF Model\]](https://huggingface.co/yuezhengrong/UniFlow)

## Introduction

We present **UniFlow**, a single visual tokenizer that delivers state-of-the-art performance for both high-fidelity image generation and visual understanding. UniFlow combines a pre-trained vision foundation encoder with a lightweight patch-wise flow decoder, eliminating the traditional conflict between discriminative and generative objectives. A layer-wise adaptive self-distillation mechanism preserves semantic richness, while flow-matching in pixel space guarantees superior reconstruction without the bottlenecks of frozen VAEs. Training converges in only 30 epochs on ImageNet-1K, yet the resulting model serves as a drop-in replacement for both CLIP-style visual backbones and VAE-style tokenizers, outperforming larger specialized counterparts. With UniFlow, one encoder, one decoder, and one training run are all you need for unified visual understanding and generation.
