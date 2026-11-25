Os dados incluem 4 pontos do braço, sendo que os últimos 3 são os relevantes para a análise.
As colunas 5 a 13, xu, yu, zu, xf, yf, zf, xh, yh, zh, correspondem às coordenadas cartesianas do ombro, cotovelo e mão, respetivamente.

Nas colunas seguintes são apresentadas as orientações em quaternion e em RPY. Por exemplo:
- q0h, q1h, q2h, q3h representam o quaternion da orientação da mão;
- rxh, ryh, rzh representam a orientação da mão em RPY.

Relativamente aos ficheiros da tarefa de reposição em retalho (SmartRetail), os nomes seguem o formato 1_right.txt, 1_left.txt, etc. O número identifica o participante, enquanto right e left indicam o braço que os dados reportam.


Assumo: movID 1/3/5/7 = pick; 2/4/6/8 = place; 9/10/11/12 = park.


Email Daniel:
```
Olá, Hugo,

Ao analisar as trajetórias do SmartRetail, reparei numa questão que te devia ter dito e que pode ter afetado a tua análise .

Algumas trajetórias tinham 9 movimentos (4 pick, 4 place, 1 park), mas outras tinham 12 movimentos (4 pick, 4 place, 4 park). Os sujeitos que tinham 12 movimentos eram o: 12,15,16,17,18,19.

Esta diferença acontece porque nem todos os participantes executavam a tarefa da mesma forma. Alguns mexiam os braços ao mesmo tempo enquanto outros faziam o seguinte: 1º colocavam um produto com o braço direito, repousavam o braço direito e só depois colocavam um produto com o braço esquerdo e repousavam o braço esquerdo.  

Assim, por exemplo, o movimento 3 do sujeito 12 (park) não é igual ao movimento 3 do sujeito 11 (pick). Ou seja, se a análise foi feita tendo em conta o movID, pode ter afetado o processamento que fizeste.

Se isto aconteceu, envio novamente os dados, agora com movID diferentes. 
Desta vez, atribuí identificadores diferentes aos movimentos de park dos sujeitos que executavam a tarefa de forma assíncrona, passando esses movimentos a ter os movID de 9,10,11,12. Desta forma, fica garantida a correspondência entre os diferentes tipos de movimento em todos os sujeitos.

Obrigado e desculpa por só me ter apercebido disto agora.
```
