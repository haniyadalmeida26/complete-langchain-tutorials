import streamlit as st
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from PIL import Image
import os
import base64

# Load environment variables
load_dotenv()


def image_to_data_url(uploaded_file):

    image_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{mime_type};base64,{base64_image}"


def get_cohere_response(input_text, image_data, prompt):

    llm = ChatCohere(
        model="command-a-vision-07-2025",
        temperature=0.3,
        cohere_api_key=os.getenv("CO_API_KEY")
    )

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt + "\n\nUser instruction: " + input_text
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data
                }
            }
        ]
    )

    response = llm.invoke([message])

    return response.content


# Streamlit configuration
st.set_page_config(
    page_title="Calories Health",
    page_icon="🥗",
    layout="centered"
)

st.header("Calories & Health 🥗")

input_text = st.text_input(
    "Input Prompt:",
    key="input"
)

uploaded_file = st.file_uploader(
    "Choose a food image...",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Food Image",
        use_container_width=True
    )


input_prompt = """
You are an expert nutritionist.

Analyze the food items visible in the uploaded image.

Identify each food item and provide:

1. Food item
2. Estimated portion
3. Estimated calories

Then calculate the total estimated calories.

Use this format:

1. Item 1 - portion - calories
2. Item 2 - portion - calories
3. Item 3 - portion - calories

--------------------------------

Total estimated calories: XXXX kcal

Important:
The calorie values are estimates because exact portion size,
ingredients, and cooking method cannot always be determined
from an image.
"""


submit = st.button("Calculate Calories")


if submit:

    if uploaded_file is None:

        st.warning("Please upload a food image first.")

    else:

        with st.spinner("Analyzing the food image..."):

            image_data = image_to_data_url(uploaded_file)

            response = get_cohere_response(
                input_text,
                image_data,
                input_prompt
            )

        st.subheader("Nutrition Analysis")

        st.write(response)