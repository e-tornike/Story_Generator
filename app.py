import urllib
from random import randint
import torch
from transformers import pipeline, set_seed
import streamlit as st
from SessionState import _SessionState, _get_session, _get_state

device = torch.device("cpu")

BAD_WORDS = []
file = urllib.request.urlopen("https://raw.githubusercontent.com/RobertJGabriel/Google-profanity-words/master/list.txt")
for line in file:
    dline = line.decode("utf-8")
    BAD_WORDS.append(dline.split("\n")[0])

STARTERS = {0: "Rick: Morty, quick! Get in the car!\nMorty: Oh no, I can't do it Rick! Please not this again.\nRick: You don't have a choice! The crystal demons are going to eat you if you don't get in!", 
            1: "Elon: Oh, you think you're all that Rick? Fight me in a game of space squash!\nRick: Let's go, you wanna-be crazy genius!\nElon: SpaceX fleet, line up!", 
            2: "Morty: I love Jessica, I want us to get married on Octopulon 300 and have octopus babies.\nRick: Shut up, Morty! You're not going to Octopulon 300!", 
            3: "Rick: Hey there, Jerry! What a nice day for taking these anti-gravity shoes for a spin!\nJerry: Wow, Rick! You would let me try out one of your crazy gadgets?\nRick: Of course, Jerry! That's how much I respect you."}

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    return pipeline("text-generation", model="../RickNMorty-GPT2/models/gpt2-entmax")


def filter_bad_words(text):
    explicit = False
    
    res_text = text.lower()
    for word in BAD_WORDS:
        if word in res_text:
            res_text = res_text.replace(word, word[0]+"*"*len(word[1:]))
            explicit = True

    if not explicit:
        return text

    output_text = ""
    for oword,rword in zip(text.split(" "), res_text.split(" ")):
        if oword.lower() == rword:
            output_text += oword+" "
        else:
            output_text += rword+" "

    return output_text


def main():
    state = _get_state()
    st.set_page_config(page_title="Story Generator", page_icon="🤗")

    model = load_model()
    # set_seed(42)  # for reproducibility

    load_page(state, model)

    state.sync()  # Mandatory to avoid rollbacks with widgets, must be called at the end of your app


def load_page(state, model):
    disclaimer_short = """
    __Disclaimer__: 
    This website uses a machine learning model to produce fictional stories.
    The model may produce hurtful, vulgar or discriminating text. 
    View the information in the sidebar for more details.
    """
    disclaimer_long = """
    __Disclaimer__:
    This website is meant for entertainment purposes only!

    __Ethical considerations__:
    We know it contains a lof of unfiltered content from the internet, which is far from neutral.

    __Model Card (by OpenAI)__:
    Because large-scale language models like GPT-2 do not distinguish fact from fiction, 
    we don’t support use-cases that require the generated text to be true. Additionally, 
    language models like GPT-2 reflect the biases inherent to the systems they were trained on, 
    so we do not recommend that they be deployed into systems that interact with humans unless 
    the deployers first carry out a study of biases relevant to the intended use-case. We found 
    no statistically significant difference in gender, race, and religious bias probes between 
    774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of 
    caution around use cases that are sensitive to biases around human attributes.

    __Tech stack__:
    This website was built using [Streamlit](https://www.streamlit.io/) and uses the awesome [Transformers](https://huggingface.co/transformers/) framework to generate text.
    """
    st.markdown(disclaimer_short)
    st.sidebar.markdown(disclaimer_long)
    st.write("---")

    state.input = st.text_area("Start your story:", state.input or STARTERS[randint(0,3)], height=200, max_chars=5000)

    state.slider = st.slider(
        "Max script length in characters (longer scripts will take more time to generate):",
        50,
        1000,
        state.slider,
    )

    if len(state.input) + state.slider > 5000:
        st.warning("Your story cannot be longer than 5000 characters!")
        st.stop()

    button_generate = st.button("Generate Story (burps)")
    if st.button("Random Reset"):
        state.clear()

    if button_generate:
        try:
            outputs = model(
                state.input,
                do_sample=True,
                max_length=len(state.input) + state.slider,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
            )

            output_text = filter_bad_words(outputs[0]["generated_text"])
            state.input = st.text_area(
                "Start your story:", output_text or "", height=50
            )
        except:
            pass

    st.markdown('<h1 style="font-family:Courier;text-align:center;">Your Story</h1>',
                unsafe_allow_html=True)
    
    for i, line in enumerate(state.input.split("\n")):
        if ":" in line:
            speaker, speech = line.split(":")

            st.markdown(
                f'<p style="font-family:Courier;text-align:center;"><b>{speaker}:</b><br>{speech}</br></p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<p style="font-family:Courier;text-align:center;">{line}</p>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
