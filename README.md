# summer-2025-REU TXST
# Utilizing AI to Support Communication in Individuals with ASD.
# Image to Narrative with multi modal architecture.

This project implements an image captioning model using (mostly) BLIP-2 & MC-LLaVA. It is designed as an assistive tool to help individuals with Autism Spectrum Disorder (ASD) interpret and communicate visual scenes. The model generates descriptive captions for input images and is structured to allow future integration with a language model for narrative generation.

---

## Project Structure
```bash
├── README.md                    # Project description & architecture

├── web-prototype (folder)
  ├── app.py (run on conda + streamlit)                  
  ├── requirements.txt             # Python dependencies for app deployment on streamlit
  ├── config.toml                  # configuration file for web app

├── mc-llava_training.py           # fine tuning code for MC-LLaVA
  ├── data_processing.py           # data pre-processing code

├── MC_LLaVA_model.py              # fine tuning attempt of MC-LLaVA w/ flickr30k dataset. No preprocessing. Collate function difficulties.

├──mc-llava-metric-train.py        # failed attempt at receiving metrics for MC-LLaVA. issues with GPU + HPC.

├── Blip_2-Training_v3.py          # blip-2 fine tuning w/ flickr30k dataset with metrics implemented

├── unused-models
  ├── prompt_fading.ipynb          # initial implementation of MC-LLaVA with prompt fading feature (failed because widgets would not load on notebook)
  ├── blip2.ipynb                  # Basic implementation of BLIP-2 captioning pipeline in Jupyter Notebook
  ├── custom_CNN_1.ipynb           # custom CNN from scratch. Used https://www.kaggle.com/datasets/chetankv/dogs-cats-images?resource=download as the dataset. early implementation.
  ├── resnet_model.ipynb           # continuation of customCNN model. Used Resnet framework to improve results from previous model.Essentially practicing using CV models as well as AI                                       training (using dog & cat datset from kaggle).
  ├── blip_llava_model.ipynb       # failed blip & llava model combo
