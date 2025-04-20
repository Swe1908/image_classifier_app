with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# App title
st.title("🖼️ Image Classification with MobileNetV2BySweZinWinLae")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    # Preprocess the image
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Prediction
    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]

    # Display predictions
    st.subheader("Predictions:")
    for i, pred in enumerate(top_preds):
        st.write(f"{i+1}. **{pred[1]}** — {round(pred[2]*100, 2)}%")
