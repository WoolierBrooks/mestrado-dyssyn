# Dirichlet e Multinomial (MN) — notas rápidas

- **Dirichlet** é uma distribuição sobre vetores de probabilidades (componentes não negativas que somam 1). É usada como *prior* conjugada para parâmetros de distribuições multinomiais/categóricas.
- **Multinomial (MN)** modela contagens em várias categorias dadas probabilidades fixas; no caso binário, a multinomial com 2 classes é equivalente à binomial (ou à categórica com 2 classes para dados individuais).
- **Caso binário**: é comum usar **Beta-Binomial** (ou Bernoulli/Binomial com prior Beta) em vez de Dirichlet-Multinomial, pois Beta é o caso particular da Dirichlet em 2 dimensões.
- **Quando faz sentido usar Dirichlet + Multinomial**: quando você tem **múltiplas categorias** (>2) ou deseja manter a mesma formulação para o caso geral, mesmo que o binário seja um caso especial.
