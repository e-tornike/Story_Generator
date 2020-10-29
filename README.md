# Rick and Morty Story Generator
This project uses a [pre-trained GPT2 model](https://huggingface.co/gpt2), which was fine-tuned on [Rick and Morty transcripts](https://rickandmorty.fandom.com/wiki/Category:Transcripts), to generate new stories in the form of a dialog. The project uses Hugging Face's [Transformers library](https://github.com/huggingface/transformers) to do inference and [Streamlit](https://www.streamlit.io/) for the application. 

Try out the [demo](https://share.streamlit.io/e-tony/story_generator_rnm/main/app.py) to generate fun stories or read the blog [post](https://towardsdatascience.com/rick-and-morty-story-generation-with-gpt2-using-transformers-and-streamlit-in-57-lines-of-code-8f81a8f92692) on how to create your won story generator.

### Fine-tuning a custom model

You can fine-tune your own model using Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1opXtwhZ02DjdyoVlafiF3Niec4GqPJvC?usp=sharing)

### Setup

This repository has only been tested with Python 3.7. 

Install the dependencies in a virtual environment:
```
python3.7 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

On the first run, the app will download the pre-trained model from Hugging Face's Model Hub or you can supply your own custom model by in the `load_model()` function. To start the application, simply run:
```
streamlit run app.py
```
