# SFT-DataSelection-at-scale
Code of paper "Rethinking Data Selection at Scale: Random Selection is Almost All You Need"

## Overview
In this paper, we replicated several self-scoring methods those that do not rely on external model assistance on two million scale datasets, and found that nearly all methods struggled to significantly outperform random selection when dealing with such large-scale data pools. Moreover, our comparisons suggest that, during SFT, diversity in data selection is more critical than simply focusing on high quality data. We also analyzed the limitations of several current approaches, explaining why they perform poorly on large-scale datasets and why they are unsuitable for such contexts. Finally, we found that filtering data by token length offers a stable and efficient method for improving results.

## Training Framework
We use the **0.8.2.dev0** version of [LLama-factory](https://github.com/hiyouga/LLaMA-Factory) to uniformly fine-tune the Large Language Models (LLM). Our fine-tuning script is shown in ``llama-factory-train.sh`` file. 

## Citation
If you finding our work interesting or helpful to you, please cite this repo.
```
@article{xia2024rethinking,
  title={Rethinking data selection at scale: Random selection is almost all you need},
  author={Xia, Tingyu and Yu, Bowen and Dang, Kai and Yang, An and Wu, Yuan and Tian, Yuan and Chang, Yi and Lin, Junyang},
  journal={arXiv preprint arXiv:2410.09335},
  year={2024}
}
```
