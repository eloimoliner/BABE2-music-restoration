# A Diffusion-Based Generative Equalizer for Music Restoration


This repository aims to create a conditional version of the CQT-Diff+ model while retaining its unconditional features (inpainting and audio generation). At the moment, it is NOT ready for use; this modification is in an Alpha state. The license for this modification is the same as that of the original BABE-2 project.

> E. Moliner, M. Turunen, F. Elvander and V. Välimäki,, "A Diffusion-Based Generative Equalizer for Music Restoration", submitted to DAFx24, Mar, 2022



![](manuscript/figures/hawaii-trend.png)


![alt text](http://research.spa.aalto.fi/publications/papers/dafx-babe2/media/pipeline.png)

## Abstract

This paper presents a novel approach to audio restoration, focusing on the enhancement of low-quality music recordings, and in particular historical ones. 
Building upon a previous algorithm called BABE, or Blind Audio Bandwidth Extension, we introduce BABE-2, which presents a series of significant improvements.
This research broadens the concept of bandwidth extension to \emph{generative equalization}, a novel task that, to the best of our knowledge, has not been explicitly addressed in previous studies. 
BABE-2 is built around an optimization algorithm utilizing priors from diffusion models, which are trained or fine-tuned using a curated set of high-quality music tracks. The algorithm simultaneously performs two critical tasks: estimation of the filter degradation magnitude response and hallucination of the restored audio. The proposed method is objectively evaluated on historical piano recordings, showing a marked enhancement over the prior version. The method yields similarly impressive results in rejuvenating the works of renowned vocalists Enrico Caruso and Nellie Melba. 
This research represents an advancement in the practical restoration of historical music.

Listen to our [audio samples](http://research.spa.aalto.fi/publications/papers/dafx-babe2/)

Read the pre-print in [arXiv](https://arxiv.org/abs/2403.18636)

## Restore a recording with a pretrained model

The pretrained checkpoints used in the paper experiments are available  [here](http://research.spa.aalto.fi/publications/papers/dafx-babe2/checkpoints/)

Note: If you intend to train this version of the model, you'll need to modify the ```"conf_custom.yaml"``` file. Change the paths "/path/to/clean/dataset" and "/path/to/noisy/dataset" to the correct paths of your clean and noisy datasets that will be used for training.

```bash
python test.py  --config-name=conf_singing_voice.yaml tester=singer_evaluator_BABE2 tester.checkpoint="path/to/checkpoint.pt" id="BABE2_restored" tester.evaluation.single_recording="path/to/recording.wav"
```

## Test unconditional sampling

```bash
python test.py  --config-name=conf_piano.yaml tester=only_uncond_maestro tester.checkpoint="path/to/checkpoint.pt" id="BABE2" tester.modes=["unconditional"]
```


## Train or fine-tune your own conditional diffusion model

Train a model from scratch (conditional noisy and clean):

```bash
python train.py --config-name=conf_custom.yaml model_dir="experiments/model_dir" exp.batch=$batch_size"
```

Fine-tune from pre-trained model:

```bash
python train.py --config-name=conf_custom.yaml model_dir="experiments/finetuned_model_dir" exp.batch=$batch_size dset.path="/path/to/dataset" exp.finetuning=True exp.base_checkpoint="/path/to/pretrained/checkpoint.pt" 
```

