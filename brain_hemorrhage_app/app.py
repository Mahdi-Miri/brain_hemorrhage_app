import streamlit as st
import numpy as np
import io
from PIL import Image

from api_service import check_hemorrhage_api
from vlm_service import analyze_with_vlm


def process_npy_to_image(npy_bytes):
    """
    Converts .npy to:
    - middle slice image
    - JPEG bytes for VLM
    """
    img_array = np.load(io.BytesIO(npy_bytes))

    # take middle slice if 3D
    if len(img_array.shape) == 3:
        img_array = img_array[img_array.shape[0] // 2]

    img_array = img_array - np.min(img_array)
    max_val = np.max(img_array)

    if max_val > 0:
        img_array = (img_array / max_val) * 255

    img_array = img_array.astype(np.uint8)

    image = Image.fromarray(img_array)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    return image, buffer.getvalue()


def main():

    st.set_page_config(
        page_title="Brain Hemorrhage AI",
        page_icon="🧠",
        layout="centered"
    )

    st.title("🧠 Intracranial Hemorrhage Detection (DL + VLM Fusion)")

    file = st.file_uploader("Upload .npy CT Scan", type=["npy"])

    if file:

        npy_bytes = file.getvalue()

        image, image_bytes = process_npy_to_image(npy_bytes)

        st.image(image, caption="CT Middle Slice", width=400)

        if st.button("Run Diagnosis"):

            # Step 1: DL model
            with st.spinner("Running Deep Learning model..."):
                api_result, prob = check_hemorrhage_api(npy_bytes)

                if "hemorrhage detected" in api_result.lower():
                    st.error(f"{api_result} ({prob})")
                else:
                    st.success(f"{api_result} ({prob})")

            # Step 2: VLM
            if api_result != "Error":
                with st.spinner("Running VLM analysis..."):
                    report = analyze_with_vlm(image_bytes, api_result, prob)

                st.divider()
                st.subheader("🧠 Final VLM Report")
                st.markdown(report)


if __name__ == "__main__":
    main()