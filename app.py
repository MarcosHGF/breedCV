from flask import Flask, render_template, request, redirect, url_for
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg16 import decode_predictions
import numpy as np

app = Flask(__name__)

# Carregar o modelo VGG16 pré-treinado com ImageNet
model = VGG16(weights='imagenet')

# Lista de palavras-chave associadas a cachorros em ImageNet (você pode adicionar outras se necessário)
dog_breeds_keywords = [
    'Labrador_retriever', 'golden_retriever', 'chihuahua', 'poodle', 'beagle', 
    'doberman', 'rottweiler', 'schnauzer', 'shih-tzu', 'pomeranian', 'husky'
    # Você pode expandir essa lista com mais raças de cães se necessário.
]

def classify_dog_breed(img_path):
    # Carregar a imagem e redimensioná-la para 224x224, tamanho compatível com a VGG16
    img = image.load_img(img_path, target_size=(224, 224))

    # Converter a imagem para um array numpy
    img_array = image.img_to_array(img)

    # Adicionar uma dimensão para representar o lote (batch size = 1)
    img_array = np.expand_dims(img_array, axis=0)

    # Pré-processar a imagem (necessário para o modelo VGG16)
    img_array = preprocess_input(img_array)

    # Fazer a previsão
    predictions = model.predict(img_array)

    # Decodificar a previsão (ImageNet retorna as 1000 classes de ImageNet)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Verificar se qualquer uma das previsões está relacionada a um cachorro
    is_dog = any(dog in decoded_predictions[i][1].lower() for i in range(3) for dog in dog_breeds_keywords)

    if is_dog:
        return f"This is a dog! Predicted breed: {decoded_predictions[0][1]}"
    else:
        return "This is not a dog!"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Obter a imagem do usuário
        file = request.files['image']

        # Verificar se o arquivo tem uma extensão válida
        if file and file.filename.endswith(('.jpg', '.jpeg', '.png')):
            # Salvar a imagem no diretório "uploads"
            img_path = os.path.join('static', 'uploads', file.filename)
            file.save(img_path)

            # Classificar a imagem
            result = classify_dog_breed(img_path)

            # Retornar o resultado para o usuário
            return render_template('index.html', result=result, img_path=img_path)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    # Certifique-se de que a pasta de uploads existe
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
    
    app.run(debug=True)
