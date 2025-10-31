Baixar as dependências:

```bash
pip install -r requirements.txt
```

Fluxo geral:

1. Coletar imagens
2. Pré processamento (redimensionar para ficar mais leve)
3. Definir rede neural
4. Definir função de perda
5. Implementar algotritmo de genético (vai ser usado para otimizar os pesos da rede neural)
6. Treinar a rede neural

Gerar amostras:

```bash
python generate_bolas_dataset.py --out ./dados --single 100 --size 24 --r-min 2 --r-max 8 --k-min 2 --k-max 6 --allow-overlap --seed 24
```
