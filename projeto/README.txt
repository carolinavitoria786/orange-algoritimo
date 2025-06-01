PROJETO: Classificador de Imagens (Gato x Cachorro e Laranjas) em TensorFlow.js

ESTRUTURA:
- index.html ............. Interface web para o classificador
- models/original/ ....... Modelos .h5 em Keras
- models/tfjs/ ............ Onde ficarão os modelos convertidos para TensorFlow.js

CONVERSÃO DOS MODELOS:
Para converter os .h5 para TF.js:

tensorflowjs_converter --input_format=keras models/original/final_model.h5 models/tfjs/catdog_model
tensorflowjs_converter --input_format=keras "models/original/final_model (1).h5" models/tfjs/orange_model

COMO EXECUTAR:
1. Inicie um servidor local:
   python -m http.server 8080
2. Acesse http://localhost:8080 no navegador
3. Use a interface para classificar imagens

REQUISITOS:
- Python 3 instalado
- TensorFlow.js (instalado via pip: pip install tensorflowjs)
