# Story Generator for Rick and Morty
This project uses a [pre-trained GPT2 model](https://huggingface.co/gpt2), which was fine-tuned on [Rick and Morty transcripts](https://rickandmorty.fandom.com/wiki/Category:Transcripts), to generate new stories in the form of a dialog. The project uses Hugging Face's [Transformers library](https://github.com/huggingface/transformers) to do inference and [Streamlit](https://www.streamlit.io/) for the demo. 



### Fine-tuning a custom model

You can fine-tune your own model using Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1opXtwhZ02DjdyoVlafiF3Niec4GqPJvC?usp=sharing)

### Setup

This repository has only been test with Python 3.7. Install dependencies in a virtual environment:
```
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

Start the app (on the first run, the app will download the pre-trained model from Hugging Face's Model Hub or you can supply your custom model by adjusting the load_model() function to your local, standard PyTorch model directory path):
```
streamlit run app.py
```