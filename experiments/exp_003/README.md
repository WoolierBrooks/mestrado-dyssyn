1. Visão geral rápida  
- Pequeno experimento exploratório para entender tamanhos dos datasets (instâncias × features) e testar comportamentos do regressor treinado com curvas sintéticas do MOSS em 29 datasets iniciais. A partir daqui vou expandir para todos os datasets (~100+)

2. O que eu executei (resumo do que foi feito)  
- Rodei um script simples para contar instâncias e features por dataset (curiosidade/diagnóstico)  
- Rodei testes do regressor treinado a partir de dados sintéticos gerados pelo MOSS sobre 29 datasets UCI (via QuaPy)  
- Não subi os arquivos gerados pelo MOSS para o repositório, eles excedem o limite do GitHub (100 MB)

3. Configurações testadas (MOSS ↔ Regressor)
- Testei o regressor usando combinações de parâmetros do MOSS:
    - ns (n_samples) = [10, 100]
    - np (n_prevalences) = [20, 30]
    - nm (n_merges) = [3, 6]
    - nc (n_curves) = [10, 100]
- cada combinação gerou um arquivo .pkl com curvas sintéticas para treinar um regressor


4. Observações preliminares (resultados curtos)
- Arquivos MOSS não versionados (peso > 100 MB)
- Hipótese inicial: algoritmos simples podem alcançar desempenho comparável a modelos treinados com datasets maiores (se as curvas sintéticas capturem bem as distribuições)
- Resultado até agora: a combinação de parâmetros do MOSS não mostrou diferenças grandes no desempenho do regressor (entre as combinações testadas)
- Batch size (variações testadas) também não evidenciou mudanças significativas no erro final, nas configurações atuais


5. Recomendação do orientador & plano de escala
- Orientador sugeriu testar treinamentos massivos: 100k – 500k curvas para checar ganho de performance
- Cenário extremo (estimo ser bons números):
    - ns = 1_000_000
    - np = 99
    - nm = 100
    - nc = 100
- → 990.000 curvas com 1M pontos cada (exercício de dimensionamento, custo computacional enorme principalmente me quesito armazenamento)

6. Observação sobre prevalências (avaliação)
- Ao avaliar erro por prevalência, recomenda-se testar várias prevalências para uma boa visão estatística
- Regra prática: ~30 prevalências costuma ser suficiente; eu pretendo testar ~50 para maior robustez

7. Próximos passos (curto prazo)
- Expandir a avaliação para todos os datasets disponíveis (~100+).
- Fazer experimentos com n_curves maiores (iniciar escala: 100k) e medir custo/benefício
- Analisar mais detalhadamente por prevalência (gráficos por dataset, médias por prevalência)
- Documentar quais combinações de MOSS foram salvas localmente (lista de arquivos) e espaço ocupado
- Se houver ganho mensurável com mais curvas, planejar armazenamento/backup