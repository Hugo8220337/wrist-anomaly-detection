
# Eu simplesmente copiei e colei, pode nao ter exatamente estes atributos
Dicionário de dados (Data)
- sub - identifica a tarefa (Repor latas numa estante ou "montar um janela")
- repID - repetição da tarefa (ex: "primeira ou segunda vez que faz a tarefa")
- movID - identificação do movimento de uma tarefa (uma tarefa tem várias repetições, e uma repetição tem vários moviementos)
- movType - "pick", "place" (colocar o objeto"), "move" (Movimento da Mao para a posiçao inicial) (está numérico, LabelEncoder) (pick=0,place=1,move=6)
- step -  passo do movimento (iterações de registo de movimento)
- (Ombro: xs, ys, zs | Cotovelo: xe, ye, ze | Mão: xw, yw, zw)
- (variáveis calculadas a partirl do x, y e z)
- vel - velocidade linear do pulso 
- acc - aceleração linear do pulso 
- t - torção da trajetória da pulso
- k -curvatura da trajetória da pulso (se o movimento fosse em linha reta este valor seria 0)

```
Envio os dados dos movimentos contendo as posições e orientações do ombro, cotovelo e mão.

Os dados incluem 4 pontos do braço, sendo que os últimos 3 são os relevantes para a análise.
As colunas 5 a 13, xu, yu, zu, xf, yf, zf, xh, yh, zh, correspondem às coordenadas cartesianas do ombro, cotovelo e mão, respetivamente.

Nas colunas seguintes são apresentadas as orientações em quaternion e em RPY. Por exemplo:
- q0h, q1h, q2h, q3h representam o quaternion da orientação da mão;
- rxh, ryh, rzh representam a orientação da mão em RPY.

Relativamente aos ficheiros da tarefa de reposição em retalho (SmartRetail), os nomes seguem o formato 1_right.txt, 1_left.txt, etc. O número identifica o participante, enquanto right e left indicam o braço que os dados reportam.

Cumprimentos,
Daniel Rodrigues
```
