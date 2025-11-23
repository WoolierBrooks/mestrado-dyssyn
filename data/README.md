# Datasets

- `raw/` – untouched source data
- `processed/` – cleaned/transformed datasets
- `external/` – datasets from collaborators or public sources

A modo de referência, todos os datasets gerados pelo moss seguem a seguinte nomenclatura:  
moss\_ns[a]\_np[b]\_nm[c]\_nc\_[d]  

Onde:
- a = número de pontos na curva   
- b = número de prevalências divididas equidistantemente usando o np.line  
- c = número de merge divido equitativamente usando o np.line  
- d = número de curvas geradas por cada combinação de b e c