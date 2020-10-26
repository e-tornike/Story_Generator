# Story Generator for Rick and Morty
This project uses a [pre-trained GPT2 model](https://huggingface.co/gpt2), which was fine-tuned on [Rick and Morty transcripts](https://rickandmorty.fandom.com/wiki/Category:Transcripts), to generate new stories in the form of a dialog. The project uses Hugging Face's [Transformers library](https://github.com/huggingface/transformers) to do inference and [Streamlit](https://www.streamlit.io/) for the demo. 



### Fine-tuning a custom model

You can fine-tune your own model using Google Colab [![Open In Colab](/home/titan/Coding/personal_website/images/colab-badge.svg)](https://colab.research.google.com/drive/1dEZL9YR-RuV6gZ2EtDbMWLS6RC8VZTlu?usp=sharing)

### Setup

Install dependencies in virtual environment:
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

Start app (on the first run, the app will download the model from Hugging Face's Model Hub):
```
streamlit run app.py
```