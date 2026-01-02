import streamlit as st
import numpy as np
import cv2
from PIL import Image

from streamlit_drawable_canvas import st_canvas
from segment_anything import sam_model_registry, SamPredictor

@st.cache_resource
def load_sam():
    sam = sam_model_registry["vit_h"](
        checkpoint="checkpoints/sam_vit_h_4b8939.pth"
    )
    sam.to(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    predictor = SamPredictor(sam)
    return predictor

predictor = load_sam()

st.title("Interactive Object Segmentation with SAM")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

st.subheader("Draw a bounding box around the object")

canvas = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=2,
    stroke_color="#FF0000",
    background_image=image,
    update_streamlit=True,
    height=image.height,
    width=image.width,
    drawing_mode="rect",
    key="canvas",
)

if canvas.json_data is not None:
    objects = canvas.json_data["objects"]

    if len(objects) > 0:
        rect = objects[0]

        x_min = int(rect["left"])
        y_min = int(rect["top"])
        x_max = int(x_min + rect["width"])
        y_max = int(y_min + rect["height"])

        bbox = np.array([x_min, y_min, x_max, y_max])
            
        predictor.set_image(image_np)

        masks, scores, _ = predictor.predict(
            box=bbox[None, :],
            multimask_output=False
        )

        mask = masks[0]
        overlay = image_np.copy()
        overlay[mask] = [0, 255, 0]  # green mask

        blended = cv2.addWeighted(image_np, 0.7, overlay, 0.3, 0)

        st.subheader("Segmentation Overlay")
        st.image(blended)
else:
    print("No object selected.")
