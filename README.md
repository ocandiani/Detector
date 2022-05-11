# README

Código de um detector que utiliza o backbone da RetinanetR101FPN3x

O __main__.py é o arquivo que realiza o treinamento. O conjunto de dados deve estar localizado em uma pasta __"Dataset"__ dentro de __src__ (ou alterar o path)

O __Predict.py__ realiza as predições em todos as imagens do conjunto de testes e printa alguns exemplos de detecção (quantidade de exemplos pode ser determinada)

O __test_predict.py__ realiza a predições em imagens do conjunto de teste, que não possuem anotações, portanto não é possivel calcular métricas.