Dicas para o trabalho:

- Tem que ter otimização de bytes (talvez com 2 apenas não seja possível);
- Configuração tomada pelo máximo não funciona (tomar decisão pelo maior valor). Aqui o número de saídas tem que indicar a tomada de decisão;
- Quando o círculo identificar identificar uma área preenchida pelas bolas, com círculo deve estar dividido em quadrantes, isso deve indicar para qual lado vale a pena ir o próximo círculo identificador (se identificar no quadrante noroeste, por exemplo, isso indica aumentar x e diminuir y);
- Testar inicialmente com apenas uma bola para verificar como o aprendizado acontece;
- Basicamente o algoritmo genético vai treinar a rede neural;
- A rede deve ser apenas binária;
- Talvez cálculo de seno/cosseno ajudem para determinar qual quadrante se mover;
- Determinar quanto ele deve se movimentar na direção determina, algo entre 0 e 1 raio do círculo identificador. Talvez utilizar sin(2PI . 180/255) = 0,X. 180 é um número aleatório escolhido pelo algoritmo genético. 0,X é a resposta entre 0 e 1 raio do círculo identificador;
    - A movimentação de x e y tem que ser de no máximo 1r.
- Porém, antes disso, é necessário gerar um conjunto de dados para o teste bom em que haja bolas para serem identificadas de raios aleatórios e posições aleatórias. Talvez gerar dois conjuntos/amostras, um com uma bola e um com mais de uma bola;
- Gerar amostragem de dados com matriz, um mapa de 255 x 255;
- Adicionar um atributo também de verificar se a borda está cortando um pixel preto. O ideal é que toda a área do círculo identificador esteja preenchido sem nenhuma borda estiver cortando. Isso significa que o círculo completo foi encontrado com perfeição.
