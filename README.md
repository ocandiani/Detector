# README

Código do meu trabalho de conclusão de curso. Transfer learning do imagenet para o conjunto de dados ZJU-Leaper 
http://www.qaas.zju.edu.cn/zju-leaper/

Código de um detector que utiliza o backbone da RetinanetR101FPN3x

Experimento realizado em um container local utilizando a dockerfile aqui disponível.

O __main__.py é o arquivo que realiza o treinamento. O conjunto de dados deve estar localizado em uma pasta __"Dataset"__ dentro de __src__ (ou alterar o path)

As anotações devem estar no formato do Coco (common objects in context).

O __Predict.py__ realiza as predições em todos as imagens do conjunto de validação e printa alguns exemplos de detecção (quantidade de exemplos pode ser determinada)

O __test_predict.py__ realiza a predições em imagens do conjunto de teste, que não possuem anotações, portanto não é possivel calcular métricas.
