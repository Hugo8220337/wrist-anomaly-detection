# RawData Folder
- Dados extraídos dos ficheiros XML do Xsens, que contém as coordenadas cartesianas de quatro pontos do braço direito da pessoa
# Data Folder
- Dados da mão divididos por repetição e movimento (temos 5 repetições com 9 movimentos cada, 4 pick(movType=0), 4 place (movType=1) e 1 de move (movType=6))
A ordem das colunas é:
Sujeito
Número da repetição
Número do movimento
Tipo de movimento (pick=0,place=1,move=1)
Step da trajetória (1/60 (60hz) segundos de intervalo entre cada step)
Coordenada em X
Coordenada em Y
Coordenada em Z
Velocidade
Aceleração
Torção
Curvatura



## 
- sub - identifica a tarefa (Repor latas numa estante ou "montar um janela") (não parece bem, isto é o identificador do sujeito)
- repID - repetição da tarefa (ex: "primeira ou segunda vez que faz a tarefa")
- movID - identificação do movimento de uma tarefa (uma tarefa tem várias repetições, e uma repetição tem vários moviementos)
- movType - "pick", "place" (colocar o objeto"), "move" (transportar o objeto) (está numérico, LabelEncoder) (pick=0,place=1,move=2)
- step -  passo do movimento (iterações de registo de movimento)
- x - coordenadas do pulso
- y - coordenadas do pulso
- z - coordenadas do pulso
(variáveis calculadas a partirl do x, y e z)
- vel - velocidade linear do pulso 
- acc - aceleração linear do pulso 
- t - torção da trajetória da pulso
- k -curvatura da trajetória da pulso (se o movimento fosse em linha reta este valor seria 0)