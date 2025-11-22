import streamlit as st
from model import TransformerNet
import torch
from torchvision import transforms
from PIL import Image
import io
import os

styles = [
    "American Steam Train Travel",
    "Farmhouse in Mahantango Valley",
    "Candy",
    "Mosaic",
    "Houses at Murnau",
    "Starry Night",
    "Terrace and Observation Deck at the Moulin de Blute-Fin Montmartre",
    "Under the Wave off Kanagawa"
]

content_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

model = TransformerNet()

def stylize(data):
    try:
        user_file_path = os.path.join("user_uploads", "1.jpg")
        os.makedirs(os.path.dirname(user_file_path), exist_ok=True)
        with open(user_file_path, "wb") as user_file:
            user_file.write(data.getbuffer())
        
        content_image = Image.open(user_file_path).convert('RGB')
        return content_image
    except Exception as e:
        st.error(f"Error loading the content image: {e}")
        return None

def main():
    st.title("Fast Neural Style Transfer")
    data = st.file_uploader("Choose an image", type=["jpg","png"], accept_multiple_files=False)
    style_choice = st.selectbox("Choose a style", styles)
    style_path = os.path.join("Style Images for Neural Style Transfer", style_choice + ".jpg")
    
    if os.path.exists(style_path):
        style_image = Image.open(style_path)
        style_image = style_image.resize((256, 256))
        col1, col2 = st.columns(2)
        with col1:
            st.header("Style Image")
            st.image(style_image, use_column_width=True)
    else:
        st.error(f"Style image not found: {style_path}")
        return

    if data is not None:
        content_image = stylize(data)
        if content_image:
            content_image_dis = content_image.resize((256, 256))
            with col2:
                st.header("Content Image")
                st.image(content_image_dis, use_column_width=True)

            content_image = content_image.resize((512,512))
            content_image = content_transform(content_image)
            content_image = content_image.unsqueeze(0)

            model_path = os.path.join("style_weights", style_choice + ".model")
            if os.path.exists(model_path):
                try:
                    state_dic = torch.load(model_path)
                    model.load_state_dict(state_dic)
                    model.eval()
                    gen_img = model(content_image)
                    gen_img = gen_img[0].clone().clamp(0, 255).detach().numpy()
                    gen_img = gen_img.transpose(1, 2, 0).astype("uint8")
                    gen_img = Image.fromarray(gen_img)
                    st.header("Generated Image")
                    st.image(gen_img, use_column_width=True)

                    img_byte_arr = io.BytesIO()
                    gen_img.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()

                    st.download_button(
                        label="Download Image",
                        data=img_byte_arr,
                        file_name="styled_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error applying style transfer: {e}")
            else:
                st.error(f"Model weights not found: {model_path}")

if __name__ == "__main__":
    main()
