import mahotas as mh
import streamlit as st
import numpy as np
import keras
from io import BytesIO
from keras.models import model_from_json

st.title('Final Project AB - Kelompok 9')

json_file = open('Covid19-model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into a new model
model.load_weights('Covid19-model/model.h5')

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""


class FileUpload(object):

    def __init__(self):
        self.fileTypes = ["png", "jpg", "jpeg"]

    def run(self):
        """
        Upload File on Streamlit Code
        :return:
        """
        st.subheader("Predict Covid-19 Image Using CNN")
        st.markdown(STYLE, unsafe_allow_html=True)
        file = st.file_uploader(
            "Upload your chest x-ray image:", type=self.fileTypes)
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a file of type: " +
                           ", ".join(["png", "jpg", "jpeg"]))
            return
        content = file.getvalue()
        if isinstance(file, BytesIO):
            lab = {'Viral Pneumonia': 0, 'Normal': 1, 'Covid': 2}
            IMM_SIZE = 224
            show_file.image(file)
            image = mh.imread(file)
            if len(image.shape) > 2:
                # resize of RGB and png images
                image = mh.resize_to(
                    image, [IMM_SIZE, IMM_SIZE, image.shape[2]])
            else:
                # resize of grey images
                image = mh.resize_to(image, [IMM_SIZE, IMM_SIZE])
            if len(image.shape) > 2:
                # change of colormap of images alpha chanel delete
                image = mh.colors.rgb2grey(image[:, :, :3], dtype=np.uint8)

            x_val = [image]
            x_val = np.array(x_val) / 255
            inputs = x_val.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

            prediction = model.predict(inputs)
            st.code(
                f"Prediction: {list(lab.keys())[list(lab.values()).index(np.argmax(prediction))]}")

        file.close()


if __name__ == "__main__":
    helper = FileUpload()
    helper.run()
