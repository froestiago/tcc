## Foundation models for vision
### Text
<p style='text-align: justify'>Foundation models for vision, similar to Large Language Models (LLMs) in language, are large transformer-based models that have undergone extensive training on vast amounts of data. These models, like their language counterparts, are designed to be generally applicable and accept prompts from users or training data.<br>
One example of such a vision model is CLIP, an image classification model that leverages text prompts. By encoding both text and images as vectors using transformer encoders, CLIP achieves image classification by correlating these vectors. CLIP's capabilities extend beyond image classification, as it is also used by generative models like DALL-E for text-to-image representation. This powerful model has been trained on a massive amount of data.<br>
Building on the concept of using prompts for image classification introduced by CLIP, a recent work introduces spatial prompts. These prompts involve drawing a red circle around a desired part of the image, which is then classified based on the enclosed content. The underlying architecture of this approach follows the same concept as CLIP, encoding both the image and the prompt.<br>
Taking the idea of prompts even further, Segment Anything (SAM) emerges as the largest foundation model for vision to date. SAM specializes in segmenting instances that correlate with a given prompt. To train SAM, its authors at FAIR have developed a novel dataset comprising 11 million images and 1 billion masks. Prompts serve as references for SAM's instance segmentation, including segmentation masks, bounding boxes, and even spatial prompts in the form of points. Although text prompts have been explored by the authors, they have not been released as part of SAM's current capabilities. SAM can be utilized as a zero-shot instance segmentation model out of the box or fine-tuned for specific datasets, providing a versatile tool for vision tasks.</p>

### References
* [(CLIP)Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020v1.pdf)
* [What does CLIP know about a red circle? Visual prompt engineering for VLMs](https://arxiv.org/pdf/2304.06712.pdf)
* [Segment Anything](https://paperswithcode.com/paper/segment-anything)
___
## Tools built on top of SAM
### Text
<p style='text-align: justify'>Upon reviewing recently published papers, it is evident that a similar phenomenon to what occurred with Language Models (LLMs) is now happening, particularly with Segment Anything (SAM). There is a growing amount of new research being conducted based on this versatile architecture. These studies primarily focus on two main areas: evaluating SAM's performance in specific domains such as medical imaging and object tracking, and exploring extensions and modifications to the model as an architectural reference for new models. One such extension, which I'll refer to as "perSAM," introduces a novel approach to utilize SAM as a few-shot learning model. On the other hand, "fastSAM" constructs a model based on Convolutional Neural Networks (CNNs) instead of Transformers, resulting in reduced parameters and improved inference time for real-time applications. Another paper, //paper on tracking//, extends SAM's capabilities to object tracking in sequential images. Additionally, "LangSAM" introduces text prompts for SAM, offering this functionality to the public before the original author. Considering the results and the increasing number of similar research endeavors, it is apparent that SAM can serve as a starting point for further segmentation research.</p>

### References
* [fast segment anything](https://paperswithcode.com/paper/fast-segment-anything)
* [personalize segment anything model with one shot](https://paperswithcode.com/paper/personalize-segment-anything-model-with-one)
* [segment anything meets point tracking](https://paperswithcode.com/paper/segment-anything-meets-point-tracking)
* [segment and track anything](https://paperswithcode.com/paper/segment-and-track-anything)
* [Lang Segment Anything](https://github.com/luca-medeiros/lang-segment-anything)
* [MedSam](https://github.com/bowang-lab/medsam)
___
## Instance Segmentation
* U-Net
* Mask RCNN -> Cascade Mask RCNN
* Backbones (CNN -> Transformers)
* Transformers -> ViT -> Swin
___
## Datasets
* [LIVECell](https://www.nature.com/articles/s41592-021-01249-6)
* [PanNuke](https://arxiv.org/pdf/2003.10778.pdf)
* A431 (?)
* [BBBC038v1 from Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC038) //Used as reference on SAM
___
## Live Cell Analysis
* Why? (Applications)
* Pre deep learning and why using deep learning
* Get some references from Nabeel's master thesis
___

