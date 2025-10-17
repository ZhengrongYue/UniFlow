<div align="center">
<h1>⚡ UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2508.05599-b31b1b.svg)](https://arxiv.org/abs/2510.10575)
[![Github](https://img.shields.io/badge/Github-UniFlow-blue)](https://github.com/ZhengrongYue/UniFlow)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/yuezhengrong/UniFlow)



</div>

This project presents **UniFlow** (Unified Pixel Flow Tokenizer), a unified continuous visual tokenizer that eliminates the conflict between semantic understanding and high-fidelity reconstruction. By marrying a pretrained vision encoder with a patch-wise flow decoder, UniFlow delivers state-of-the-art reconstruction and multimodal understanding via a single encoder tokenizer, while surpassing both discrete and continuous counterparts. <br><br>



<p align="center">
  <img src="./assets/intro.jpg" width="90%">
  <br>
  <em>
  Comparison of different training paradigms for unified tokenizers. UniFlow achieves a new state-of-the-art in reconstruction fidelity and multimodal understanding, surpassing both discrete and continuous unified tokenizers, while offering high compression ratios. (All multimodal large language models are trained on LLaVA-v1.5 data with Vicuna-7B, except that TokenFlow uses Vicuna-13B.)</em>
</p>


<p align="center">
  <img src="./assets/vis.jpg" width="90%">
  <br>
  <em>Various downstream tasks demonstrate UniFlow's robust visual representation.</em>
</p>

## 🔥 Updates

* **[2025.10.01]** 🚀 🚀 🚀 We are excited to release **UniFlow**, a powerful unified tokenizer featuring our novel **Layer-wise Adaptative Distillation** and a **Patch-wise Pixel Flow Decoder**. Code and pretrained models are now available!

## 📖 Quick Start

### Inference

Simply test the effect of reconstruction:
```
quick_start.ipynb
```



## ❤️ Acknowledgement
Our work builds upon the foundations laid by many excellent projects in the field. We would like to thank the authors of [MAR](https://github.com/LTH14/mar). We also drew inspiration from the methodologies presented in [FlowMo](https://github.com/kylesargent/FlowMo/), [InternVideo2](https://github.com/OpenGVLab/InternVideo/tree/main) and [InternVL](https://github.com/OpenGVLab/InternVL). We are grateful for their contributions to the community.


## 📘 Citation
```bibtex
@misc{yue2025uniflowunifiedpixelflow,
      title={UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation}, 
      author={Zhengrong Yue and Haiyu Zhang and Xiangyu Zeng and Boyu Chen and Chenting Wang and Shaobin Zhuang and Lu Dong and KunPeng Du and Yi Wang and Limin Wang and Yali Wang},
      year={2025},
      eprint={2510.10575},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.10575}, 
}
```

## ✨ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ZhengrongYue/UniFlow&type=date&legend=top-left)](https://www.star-history.com/#ZhengrongYue/UniFlow&type=date&legend=top-left)
