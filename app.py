import urllib
from random import randint
import torch
from transformers import pipeline, set_seed
from transformers.pipelines import TextGenerationPipeline
import streamlit as st
from SessionState import _SessionState, _get_session, _get_state


device = torch.device("cpu")


def load_bad_words() -> list:
    res_list = []

    file = urllib.request.urlopen(
        "https://raw.githubusercontent.com/RobertJGabriel/Google-profanity-words/master/list.txt"
    )
    for line in file:
        dline = line.decode("utf-8")
        res_list.append(dline.split("\n")[0])

    return res_list


BAD_WORDS = load_bad_words()

STARTERS = {
    0: "Rick: Morty, quick! Get in the car!\nMorty: Oh no, I can't do it Rick! Please not this again.\nRick: You don't have a choice! The crystal demons are going to eat you if you don't get in!",
    1: "Elon: Oh, you think you're all that Rick? Fight me in a game of space squash!\nRick: Let's go, you wanna-be genius!\nElon: SpaceX fleet, line up!",
    2: "Morty: I love Jessica, I want us to get married on Octopulon 300 and have octopus babies.\nRick: Shut up, Morty! You're not going to Octopulon 300!",
    3: "Rick: Hey there, Jerry! What a nice day for taking these anti-gravity shoes for a spin!\nJerry: Wow, Rick! You would let me try out one of your crazy gadgets?\nRick: Of course, Jerry! That's how much I respect you.",
    4: "Rick: Come on, flip the pickle, Morty. You're not gonna regret it. The payoff is huge.",
    5: "Rick: I turned myself into a pickle, Morty! Boom! Big reveal - I'm a pickle. What do you think about that? I turned myself into a pickle!",
    6: "Rick: Come on, flip the pickle, Morty. You're not gonna regret it. The payoff is huge.\nMorty: What? Where are you?\nRick: Morty, just do it! [laughing] Just flip the pickle!",
}


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model() -> TextGenerationPipeline:
    return pipeline("text-generation", model="e-tony/gpt2-rnm")


def filter_bad_words(text: str) -> str:
    explicit = False

    res_text = text.lower()
    for word in BAD_WORDS:
        if word in res_text:
            print(word)
            res_text = res_text.replace(word, word[0] + "*" * len(word[1:]))
            explicit = True

    if explicit:
        output_text = ""
        for oword, rword in zip(text.split(" "), res_text.split(" ")):
            if oword.lower() == rword:
                output_text += oword + " "
            else:
                output_text += rword + " "
        text = output_text

    return text


def main():
    state = _get_state()
    st.set_page_config(page_title="Story Generator", page_icon="🛸")

    model = load_model()
    # set_seed(42)  # for reproducibility

    load_page(state, model)

    state.sync()  # Mandatory to avoid rollbacks with widgets, must be called at the end of your app


def load_page(state: _SessionState, model: TextGenerationPipeline):
    disclaimer_short = """
    __Disclaimer__: 

    _This website is for entertainment purposes only!_

    This website uses a machine learning model to produce fictional stories.
    Even though certain bad words get censored, the model may still produce hurtful, vulgar, violent or discriminating text. 
    Use at your own discretion.
    View the information in the sidebar for more details.
    """
    disclaimer_long = """
    __Description__:

    This project uses a [pre-trained GPT2 model](https://huggingface.co/gpt2), which was fine-tuned on [Rick and Morty transcripts](https://rickandmorty.fandom.com/wiki/Category:Transcripts), to generate new stories in the form of a dialog. 
    For a detailed explanation of GPT2 and its architecture see the [original paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), OpenAI’s [blog post](https://openai.com/blog/better-language-models/) or Jay Alammar’s [illustrated guide](http://jalammar.github.io/illustrated-gpt2/).

    __Ethical considerations__:

    The original GPT2 model was trained on WebText, which contains 45 million outbound links from Reddit (i.e. websites that comments reference).
    While certain domains were removed, the model was trained on largely unfiltered content from the Internet, which contains biased and discriminating language.
    
    __[Model Card](https://github.com/openai/gpt-2/blob/master/model_card.md) (by OpenAI)__:

    "_Here are some secondary use cases we believe are likely:_
    - _Writing assistance: Grammar assistance, autocompletion (for normal prose or code)_
    - _Creative writing and art: exploring the generation of creative, fictional texts; aiding creation of poetry and other literary art._
    - _Entertainment: Creation of games, chat bots, and amusing generations._

    _Out-of-scope use cases:_

    _Because large-scale language models like GPT-2 do not distinguish fact from fiction, 
    we don’t support use-cases that require the generated text to be true. Additionally, 
    language models like GPT-2 reflect the biases inherent to the systems they were trained on, 
    so we do not recommend that they be deployed into systems that interact with humans unless 
    the deployers first carry out a study of biases relevant to the intended use-case. We found 
    no statistically significant difference in gender, race, and religious bias probes between 
    774M and 1.5B, implying all versions of GPT-2 should be approached with similar levels of 
    caution around use cases that are sensitive to biases around human attributes._"

    __Tech stack__:

    This website was built using [Streamlit](https://www.streamlit.io/) and uses the [Transformers](https://huggingface.co/transformers/) framework to generate text.
    """
    st.markdown(disclaimer_short)
    st.sidebar.markdown(disclaimer_long)

    # st.write("---")

    st.title("Story Generator")

    state.input = st.text_area(
        "Start your story:",
        state.input or STARTERS[randint(0, 6)],
        height=200,
        max_chars=5000,
    )

    state.slider = st.slider(
        "Max story length in characters (longer scripts will take more time to generate):",
        50,
        1000,
        state.slider,
    )

    if len(state.input) + state.slider > 5000:
        st.warning("Your story cannot be longer than 5000 characters!")
        st.stop()

    button_generate = st.button("Generate Story (burps)")
    if st.button("Reset Prompt (Random)"):
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

    st.markdown(
        '<h2 style="font-family:Courier;text-align:center;">Your Story</h2>',
        unsafe_allow_html=True,
    )

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
    
    st.markdown("---")
    st.markdown(
        "_You can read about how to create your own story generator application [here](https://towardsdatascience.com/rick-and-morty-story-generation-with-gpt2-using-transformers-and-streamlit-in-57-lines-of-code-8f81a8f92692). The code for this project is on [Github](https://github.com/e-tony/Story_Generator_RnM)._"
    )


if __name__ == "__main__":
    main()
