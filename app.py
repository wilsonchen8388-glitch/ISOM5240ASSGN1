import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

# page title
st.title("AI Storytelling App for Kids")

# upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# run only if image uploaded
if uploaded_file is not None:

    # show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # image caption model
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    # generate caption
    inputs = processor(image, return_tensors="pt")

    output = model.generate(**inputs, max_new_tokens=30)

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    # display caption
    st.subheader("Image Caption")
    st.write(caption)

    # generate story
    story_text = f"""
    Once upon a time, there was {caption}.
    One sunny morning, a little child saw this and became very curious.
    The child smiled and imagined a kind adventure full of magic and friendship.
    Everyone learned that being brave, helpful, and kind can make the day special.
    From that moment, the world felt a little brighter and happier.
    """

    # display story
    st.subheader("Generated Story")
    st.write(story_text)

    # text to speech
    tts = gTTS(text=story_text, lang="en")

    audio_file = "story.mp3"

    tts.save(audio_file)

    # play audio
    st.audio(audio_file)
