# TinyGPT — Dense vs Mixture of Experts (MoE)

Trabajo práctico de la materia **PLN-II**: implementación y estudio de un modelo tipo GPT (decoder Transformer) entrenado **desde cero** sobre `tinyshakespeare.txt` con **tokenización carácter a carácter**. Se comparan dos variantes:
- **Vanilla (Dense FFN)**: bloque feed-forward clásico.
- **MoE**: reemplazo del FFN por **Mixture of Experts** (4 expertos, ruteo top-k=1).

## Contenido básico del trabajo
- Un **notebook** (`TinyGPT.ipynb`) que entrena y evalúa ambos modelos.
- **Estrategias de decodificación** en `generate()` y `generateV2()`: greedy, temperature, top-k/top-p.
- **Curvas de loss y perplexity** (train/val) por época.
- **Visualización de atención** (mapas por capa/cabeza) para entender qué mira el modelo.
- Implementación de **MoE** (experts, gate y capa MoE) y comparación con la versión Dense.

## Expectativas de los resultados
- Con tokenización a nivel **carácter**, ambos modelos generan **palabras válidas** y muestran **estructura sintáctica básica** (puntuación, mayúsculas, cambios de hablante).
- La variante **MoE** mejora **loss** y **perplexity** frente a **Dense** (más capacidad efectiva por token), pero la **calidad percibida del texto** es **similar** dada la escala del modelo y la granularidad carácter-a-carácter.
- En mapas de atención verás **diagonales** (atención causal al pasado) y cabezas con enfoques distintos (locales vs. más dispersos).

## Estructura de archivos abreviada
```
/imgs                   # Imágenes utilizadas en la Jupyter notebook principal
/moe_checkpoints        # Pesos MoE (si se guardan)
/vanilla_checkpoints    # Pesos Dense (si se guardan)
trainer.py              # Módulo con lógica de entrenamiento/evaluación y checkpoints
TinyGPT.ipynb           # Notebook con desarrollo del trabajo práctico
```

## Limitaciones y mejoras futuras
- La tokenización por **caracteres** limita la coherencia a largo plazo. Próximo paso: **BPE/WordPiece**.
- Escalar **n_embd/n_layer** y corpus para evidenciar mejor las ventajas cualitativas de MoE.
- Regularización/balanceo de ruteo en MoE para fomentar especialización estable.
