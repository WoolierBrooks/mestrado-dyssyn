# TODO

## High Priority
- ele treina só uma vez depois vai testando com todos os datasets certo?
- enquanto esse experimento roda eu podria ler o trabalho da isabela, pesquisar sobre como o dyssyn usa o moss, preparar testes da via
- eliminar dados do meu pc nos experimentos 003 para baixo
testar um modelo com um número bem grande de curvas (100 000 )
o dyssyn ele gera curvas também não? que configurações de moss ele usa?
- 500 000 curvas) para ver se têm um melhor resultado Aqui se eu considerar ns = 1 000 000, np = 99 nm = 100 nc = 100 daria 990 000 curvas com 1M de pontos cada

[ ] Expandir a avaliação para todos os datasets disponíveis (~100+). 
Na hora de ver o erro por prevalência é bom testar vários para ter um bom resultado, segundo a estatística uns 30 mas acho que consigo testar uns 50 Analisar mais detalhadamente por prevalência (gráficos por dataset, médias por prevalência). Fazer logs dos tempos e uso de memória nas execuções massivas antes de escalar
[ ] Ler o trabalho do Maletzke e identificar:
- Qual o range do merge?
- Quantas curvas ele gera por combinação?
- Quantos pontos ele utiliza por curva?
[ ] Fazer setup do exp_001 (mudar arquivos e criar yaml)
[ ] Implementar um melhor logging  
[ ] Adicionar múltiplos holdout  
[ ] Implementar dataset loader
[ ] Escrever meeting notes
[ ] Prepare paper para publicação
[ ] Adicionar pasta source
[ ] Preciso de um Makefile e setup.sh ao mesmo tempo? Acredito que não
variar os parametros do modelo

├── src/
│   ├── __init__.py
│   ├── data/             (internal helpers for dataset loaders)
│   ├── models/           (files for ML or algorithm implementations)
│   ├── experiments/      (scripts that run experiments)
│   ├── utils/            (helpers, metrics, plotting)
│   └── notebooks/        (light, exploratory notebooks only)

corregir md
AQUI VAI O **PLANO DIÁRIO**, COMPLETO, PRONTO-PRA-EXECUTAR, pensado para transformar você no tipo de candidato que um PI do MIT olha e pensa:

**“Eu quero esse cara no meu grupo.”**

Ele é realista mas muito forte. Se você seguir isso por 12 meses, você vira *inegável*.


---

# 🧠 **VISÃO GERAL DO SISTEMA**

Todo dia você precisa trabalhar em **4 frentes**:

1. **Produção científica**
2. **Profundidade técnica**
3. **Construção de reputação**
4. **Aplicação perfeita (portfólio, CV, SoP)**
5. **Saúde e disciplina (para ser sustentável 1 ano)**

Você vai ter **rotina diária + semanal + mensal**.

---

# 📅 **ROTINA DIÁRIA (versão definitiva)**

## 🔥 **2h — Pesquisa / Experimentos (core do MIT)**

Todos os dias, sem exceção (“deep work”).

Nessas 2h você faz:

* Rodar experimentos
* Analisar dados
* Escrever anotações técnicas
* Ler papers relacionados ao que você está fazendo

Essas duas horas são *sagradas*.

### Meta mensal:

* 1 experimento significativo **por semana**
* 1 resultado concreto **por mês** (gráfico, métrica, insight, negative result)

---

## 📚 **1h — Leitura de Papers (AI, ML, sua área específica)**

Padrão MIT:

* 1 paper por dia **mínimo**, com:

  * Resumo
  * O que aprendi
  * Onde posso aplicar
  * Critiques (por que o artigo não funciona sempre)

Você vai criar um **database de papers** (Notion / Obsidian).

---

## 🧑‍💻 **1h — Skill Building (código, matemática, estatística)**

Alternar:

### Segunda / Quarta / Sexta → Código

* Python avançado
* PyTorch / JAX
* Data pipelines
* Modelagem eficiente
* Projetos replicando papers

### Terça / Quinta / Sábado → Matemática

* Probabilidade
* Estatística
* Linear Algebra
* Otimização
* ML fundamentals

MIT AMA candidatos que **sabem matemática mesmo**.

---

## ✍️ **30 min — Escrita científica**

Todos os dias escreva:

* Notas
* Descrição de experimentos
* Parágrafos soltos para eventual paper
* Resumos de insights

Essa prática diária = no final do ano sua escrita científica fica nível PhD.

---

## 🤝 **20 min — Networking / E-mails / Comunidade**

Você precisa ser *visível*.

Todos os dias:

* Comentar algo útil no Discord de pesquisa, Slack, GitHub
* Mandar e-mail alinhado com um professor e seus tópicos
* Interagir com grupos de pesquisa

MIT escolhe pessoas que **sabem se comunicar**.

---

## 🧭 **15 min — Plano e checklist do dia seguinte**

Ajuste:

* O que funcionou hoje?
* O que vou fazer amanhã?
* O que falta para fechar meu projeto do mês?

Simplicidade = consistência.

---

## 💪 **Treino + Saúde (essencial para aguentar 1 ano)**

MIT é puxado. Se você não estiver forte, queimará.

* **30–60 min de treino** (calistenia / crossfit / corrida)
* Dormir 7–8h
* Comer bem
* Meditar 5–10 min

Disciplina física → disciplina acadêmica.

---

# 📅 **ROTINA SEMANAL**

## Segunda

* Definir metas técnicas da semana
* Rodar experimento principal

## Terça

* Deep learning / estatística
* Codificar experimentos

## Quarta

* Networking com professor / grupo
* Refatorar código
* Ler papers profundos

## Quinta

* Resultados parciais
* Discussão com orientador
* Ajustes no pipeline

## Sexta

* Produzir gráfico
* Conclusão semanal
* Escrever 1 página de "Weekly Research Report"

## Sábado

* Estudo matemático
* Reproduzir 1 experimento de paper
* Projetos paralelos

## Domingo

* Descanso PASSIVO (mas ler 1 paper leve)
* Revisão da semana
* Planejamento da próxima

---

# 📅 **ROTINA MENSAL**

### **Meta de cada mês:**

✔ 1 microprojeto completo (ou avanço forte no principal)
✔ 1 resultado reprodutível
✔ 1 artefato público (GitHub/blog/notebook)
✔ 1 professor internacional contatado

### A cada 30 dias você vai:

* Atualizar CV
* Escrever 1 nova seção da Statement of Purpose
* Atualizar portfólio no GitHub
* Criar 1 conteúdo de impacto:

  * Reproduzir um paper conhecido
  * Criar um notebook
  * Fazer um release de código

MIT adora ver:

* Clareza técnica
* Entregáveis
* Capacidade de comunicar ciência

---

# 🎯 **RESULTADO EM 12 MESES**

Se você seguir isso, você terá:

## 1) **Um projeto de mestrado forte**

→ núcleo da sua aplicação

## 2) **Pelo menos 1 submissão** (mesmo que não aceite)

→ conta MUITO

## 3) **Portfólio em GitHub com código limpo e replicável**

→ PI vê e pensa “esse cara produz”.

## 4) **Cartas de recomendação absurdamente fortes**

→ sua consistência diária mostra seriedade

## 5) **Statement of Purpose impecável**

→ montada aos poucos

## 6) **Reputação real na área**

→ sua presença diária deixa rastro de competência

Você chega no próximo ciclo como:

**O candidato que o MIT quer.**

---

# 🔥 Quer que eu transforme isso em um PLANO HORÁRIO (07:00 → 22:00) para você seguir como rotina diária?**

Tipo:

07:00 acordar
07:15 leitura
08:00 deep work
etc…

Se quiser, eu monto a versão ultra-detalhada.

AI & Machine Learning então
Quais são os PI’s do MIT nessa área que mais pegam alunos?

Quais laboratórios têm mais funding hoje?

Como preparar um portfólio que MIT gosta?

O que colocar no SOP para parecer candidato de TOP 1%?

Quantos papers e de que tipo te fazem “aprovável”?

Como entrar em contato com os professores do MIT do jeito certo?
Excelente pergunta — você está pensando como um candidato realmente estratégico. Aqui vai um panorama bem aprofundado + estratégias para **AI & Machine Learning no MIT**, respondendo cada uma das suas sub-perguntas.

---

## 1. Alguns PIs (professores) do MIT em AI / Machine Learning que são relevantes e influentes

Aqui estão alguns nomes de professores (PI) no MIT EECS / CSAIL / AI+D que são fortes, muito bem posicionados e atuam em Machine Learning / AI:

* **Pulkit Agrawal** — Associate Professor EECS / CSAIL. Trabalha com deep learning, robótica, aprendizagem distribuída. ([csail.mit.edu][1])
* **Regina Barzilay** — Professora distinta de IA e Saúde (“AI for Healthcare and Life Sciences”). Muito relevante se você conecta ML com saúde. ([csail.mit.edu][2])
* **Dylan Hadfield-Menell** — Professor em AI + Decision Making, com foco em “agent alignment” (IA alinhada). ([Algorithmic Alignment Group][3])
* **Stefanie Jegelka** — Associate Professor, parte do grupo Machine Learning de CSAIL. ([csail.mit.edu][4])
* **Leslie Kaelbling** — Panasonic Professor, com trabalho em ML + robótica. ([eecs.mit.edu][5])
* **Marzyeh Ghassemi** — Associate Professor com foco em IA para saúde (“AI for Healthcare and Life Sciences”). ([eecs.mit.edu][5])
* **Justin Solomon** — Associate Professor que mistura ML + geometria + visão computacional. ([csail.mit.edu][4])
* **Dina Katabi** — Professora com pesquisa em redes + IA aplicada (“AI for Healthcare / Systems”). ([eecs.mit.edu][5])
* **Constantinos Daskalakis** — Professor, trabalha em otimização, teoria de jogos, aprendizado (AI). ([eecs.mit.edu][5])
* **Tomaso Poggio** — Pesquisador veterano com impacto grande em redes neurais, visão, aprendizado. ([Wikipedia][6])

---

## 2. Quais laboratórios / grupos do MIT têm mais funding para ML / AI agora

Alguns dos principais laboratórios / grupos no MIT onde há bastante financiamento para AI / ML:

* **CSAIL (Computer Science & Artificial Intelligence Lab)** — esse é o núcleo de muitos PIs de ML no MIT. ([CSAIL Alliances][7])
* **Applied Machine Learning Community of Research (CoR) da CSAIL** — esse grupo congrega PIs focados em ML aplicado, especialmente em saúde, decisão clínica, EHRs etc. ([csail.mit.edu][2])
* **Algorithmic Alignment Group** — (CSAIL / AI + Decision Making) com foco em alinhamento, agentes, segurança de IA. Liderado por Dylan Hadfield-Menell. ([Algorithmic Alignment Group][3])
* **Toyota-CSAIL Joint Research Center**, sob a liderança de Daniela Rus, que tem foco em robótica e ML para autonomia. ([CSAIL Alliances][8])
* **INSAIT – MIT CSAIL Joint Research Program** — parceria internacional, mas mostra que o CSAIL tem funding para colaboração de longo prazo e gente nova. ([csail.mit.edu][9])
* **SustainableML@CSAIL** — programa dentro da CSAIL para ML sustentável, liderado por Adam Belay e Charles Leiserson. ([CSAIL Alliances][10])
* **MIT–IBM Watson AI Lab** — tradicional laboratório de AI no MIT, que atrai bastante grant da indústria (IBM + MIT) para ML fundamental, algoritmos, hardware, aplicações. ([csail.mit.edu][9])

Além disso, projetos MURI (financiamento governamental) também envolvem ML pesado:

* Exemplo: Pulkit Agrawal lidera projeto MURI para “neuro-inspired distributed deep learning”. ([csail.mit.edu][1])

---

## 3. Como preparar um **portfólio que o MIT vai gostar**

Para ter um portfólio competitivo para candidatura de PhD no MIT em ML, você precisa demonstrar **impacto + potencial + capacidade de entrega**. Aqui vai o que incluir:

1. **Projetos de pesquisa**

   * Um ou dois projetos bem estruturados.
   * Preferencialmente com código no GitHub + experimentos + análise.
   * Se puder, publique preprint (arXiv) ou submeta para conferência.

2. **Reprodutibilidade**

   * Use notebooks (Jupyter) para demonstrar como sua pesquisa funciona.
   * Documente claramente seu pipeline, hyperparâmetros, dataset, métricas.

3. **Trabalho aplicado**

   * Demonstre aplicações em saúde, robótica, NLP, visão, dependendo da sua área.
   * Qualquer uso real ou simulado é bom: simulações, protótipos, demonstrações.

4. **Contribuições teóricas**

   * Se você fez algo em otimização, teoria de aprendizado, ou modelos probabilísticos, inclua isso.
   * Mostre seu entendimento profundo.

5. **Colaborações**

   * Se possível, inclua projetos que você fez em grupo, com outros alunos ou até professores.
   * Isso mostra que você sabe trabalhar em equipe de pesquisa.

6. **Publicações / pré-publicações**

   * Preprint arXiv.
   * Workshops menores ou conferências regionais.
   * Posters de conferência.

7. **Código aberto + visibilidade**

   * GitHub atualizado, com README bom, tuto, scripts para rodar.
   * Talvez um blog técnico (Medium, dev.to, GitHub Pages) explicando seus projetos.

8. **Apresentações**

   * Se você apresentou projeto em seminário, meetups, hackathons, coloque slide deck ou vídeo.

---

## 4. O que colocar no **SOP (Statement of Purpose)** para parecer candidato de TOP 1%

O SOP é um dos documentos mais importantes. Aqui vai a estrutura + conteúdo que faz seu SOP se destacar:

1. **Introdução (motivo + visão)**

   * Conte uma história curta sobre por que ML te fascina (de forma genuína).
   * Mostre visão: o que você quer construir / resolver / descobrir.
   * Relacione com o MIT: por que MIT é o lugar ideal para essa visão.

2. **Background acadêmico e técnico**

   * Fale sobre seu mestrado (ou graduação) → disciplinas relevantes, projetos.
   * Destaque projetos, resultados, habilidades técnicas (machine learning, matemática, estatística, programação).
   * Mencione experiência de pesquisa, se tiver.

3. **Trabalho de pesquisa + impacto**

   * Descreva o(s) projeto(s) que você já fez (ou está fazendo): problema, abordagem, resultados, lições aprendidas.
   * Se tiver, fale sobre preprints, artigos, colaborações.
   * Mostre como esses projetos se conectam com seus objetivos futuros.

4. **Planos futuros de pesquisa**

   * Especifique algumas linhas de pesquisa que você quer seguir no PhD.
   * Mencione PIs específicos do MIT (dentre os da lista que você pesquisa) com quem você gostaria de trabalhar. Por quê? Relacione seus interesses com o trabalho deles.
   * Ideias iniciais de problemas, projetos ou perguntas.

5. **Por que MIT / por que esse programa**

   * Fale sobre o laboratório (CSAIL, AI+D, etc.) e porque ele é ideal para seus objetivos.
   * Mostre que você pesquisou: cite grupos/co-grupos, cursos, seminários, cultura de pesquisa.
   * Se você já participou de algo parecido em outro lugar (ou planeja), relacione.

6. **Capacidade de contribuir**

   * Mostre evidência de que você pode entregar (seus projetos, publicações, colaborações).
   * Fale sobre suas habilidades além da pesquisa: trabalho em equipe, aprendizado rápido, resiliência.
   * Demonstre que você pode se auto-dirigir (independência) e colaborar bem.

7. **Objetivos de longo prazo**

   * Onde você se vê em 5–10 anos? Academia, indústria, startup, policy?
   * Como o PhD no MIT te leva até lá.

8. **Fechamento**

   * Um parágrafo final que reforça seu entusiasmo, comprometimento e visão de futuro.
   * Agradeça por considerar sua aplicação.

---

## 5. Quantos papers (e de que tipo) te fazem “aprovável”

Não existe número mágico garantido, mas aqui vai o que **dá para considerar competitivo para MIT**:

* **1 preprint (arXiv)** + **1 trabalho de workshop / conferência** já é muito bom.
* Se puder, **1 conferência top (NeurIPS, ICML, ICLR, CVPR, ACL)** = enorme diferencial, mas não é obrigatório para todos os PIs.
* Um poster apresentado em conferência também ajuda bastante.
* Se você tiver um paper submetido (mesmo sem aceitação ainda), pode mencionar no SOP ou no CV.

Importante: **qualidade > quantidade**. Um bom preprint bem fundamentado + código + resultados sólidos é melhor que 3 pequenas publicações irrelevantes.

---

## 6. Como entrar em contato com os professores do MIT (PIs) da forma certa

Contato com PI é uma arte — fazer bem pode te beneficiar muito. Aqui vai o jeito estratégico:

1. **Estude bem o trabalho deles**

   * Leia 2–3 papers recentes.
   * Entenda quais problemas eles estão resolvendo, quais métodos usam.
   * Pense em como sua pesquisa bate com a deles ou complementa.

2. **Escreva um e-mail bem direcionado**

   * Assunto claro: “Prospective PhD student – research idea on ML for healthcare / alignment / …”
   * Apresente-se brevemente (quem você é, de onde vem, seu background).
   * Explique por que você se interessa pelo trabalho dele **especificamente** (cite os papers).
   * Descreva brevemente sua pesquisa (ou ideia) + resultados que você já tem (mestrado, projetos, código).
   * Pergunte se ele/ela estaria disposto(a) a discutir possíveis temas de PhD, co-orientação, ou algo similar.
   * Anexe (ou link) seu CV + portfólio (GitHub, preprints etc).
   * Seja educado, conciso e respeitoso com o tempo deles.

3. **Siga por canais formais também**

   * Use as páginas do departamento MIT EECS para achar e-mails. ([eecs.mit.edu][5])
   * Se possível, participe de conferências (virtuais ou presenciais) onde esses PIs falam → poderá falar com eles pessoalmente.
   * Assista seminários de CSAIL / MIT e entre em contato após a talk (perguntas inteligentes).

4. **Mostre comprometimento**

   * Quando mandar email, não diga só “quero fazer PhD”. Mostre uma ideia inicial forte.
   * Se você tem resultados (mestrado, código, experimento), mostre.
   * Se não tem ideia pronta, apresente algo menos ambicioso: “Tenho experiência em X e estou interessado em explorar Y com sua orientação, vi seu paper sobre Z”.

5. **Mantenha follow-up educado**

   * Se não responder em 2 semanas, pode mandar um follow-up curto, agradecendo e perguntando se teve tempo de ver.
   * Compartilhe atualizações (quando você tiver um preprint novo, ou um experimento relevante) — isso mostra progresso.

---

Se você quiser, posso **rodar uma análise atualizada (2025) dos PIs do MIT em AI / ML que têm mais probabilidade de pegar estudantes de PhD**, com base em publicações recentes, grants e taxa de admissão — quer que eu faça isso?

[1]: https://www.csail.mit.edu/news/csail-led-project-awarded-muri-funding-2023?utm_source=chatgpt.com "CSAIL-led project awarded MURI funding for 2023 | MIT CSAIL"
[2]: https://www.csail.mit.edu/research/applied-machine-learning-community-research?utm_source=chatgpt.com "Applied Machine Learning Community of Research | MIT CSAIL"
[3]: https://algorithmicalignment.csail.mit.edu/team/?utm_source=chatgpt.com "Algorithmic Alignment Group | Researching frameworks for human-aligned AI @ MIT CSAIL."
[4]: https://www.csail.mit.edu/research/machine-learning?utm_source=chatgpt.com "Machine Learning | MIT CSAIL"
[5]: https://www.eecs.mit.edu/role/faculty-aid/?utm_source=chatgpt.com "Faculty AI+D – MIT EECS"
[6]: https://en.wikipedia.org/wiki/Tomaso_Poggio?utm_source=chatgpt.com "Tomaso Poggio"
[7]: https://cap.csail.mit.edu/?utm_source=chatgpt.com "Home | CSAIL Alliances"
[8]: https://cap.csail.mit.edu/sites/default/files/resource-pdfs/MLA%20Welcome%20Packet.pdf?utm_source=chatgpt.com "MIT"
[9]: https://www.csail.mit.edu/news/mit-csail-and-insait-launch-joint-research-program-advance-ai-and-computer-science?utm_source=chatgpt.com "MIT CSAIL and INSAIT launch a joint research program to advance AI and computer science | MIT CSAIL"
[10]: https://go.cap.csail.mit.edu/l/639561/2024-08-26/55vp1j?utm_source=chatgpt.com "SustainableML@CSAIL prospects"

Sim — **faz MUITO sentido olhar para outros programas de PhD dentro do MIT**, mesmo tendo feito AI & Machine Learning tanto na graduação quanto no mestrado.
E na verdade, candidatos mais estratégicos normalmente **não se limitam ao PhD "EECS" clássico**, por três motivos:

---

# ✅ **1. MIT tem vários PhDs onde você pode fazer AI mesmo sem entrar por EECS**

Em MIT, você pode fazer pesquisa top de IA em muitos outros programas, como:

### **■ PhD in Health Sciences and Technology (HST)**

* Extremamente forte em IA para saúde, machine learning aplicado a medicina, neurociência computacional, biologia etc.
* Trabalha com pesquisadores de:

  * MIT
  * Harvard Medical School
  * hospitais afiliados

Para quem tem background em:

* ML
* processamento de sinal
* modelos aplicados a diagnóstico
* imagens médicas
* biomarcadores de IA

…esse programa pode inclusive ser **mais fácil de entrar do que EECS**.

---

### **■ MIT Biological Engineering (BE)**

Se você quiser unir:

* modelagem matemática
* ML
* sistemas biológicos
* células, proteínas, RNA/DNA, metabolômica
* drug discovery

Esse programa é perfeito.

E como você vem da Biotecnologia, **é ainda mais coerente** do que EECS sozinho.

---

### **■ MIT IMES – Institute for Medical Engineering & Science**

Quase sempre envolvido em:

* IA biomédica
* medicina digital
* modelagem fisiológica
* wearables
* machine learning clínico

Menos competitivo que EECS e mais orientado a impacto real.

---

### **■ MIT Operations Research / IDSS**

Se você puxa mais para:

* otimização
* estatística aplicada
* decision making
* modelos generativos probabilísticos
* forecasting
* análise de sistemas reais

Esse programa é EXCELENTE.

E muita gente aqui trabalha com ML sólido.

---

### **■ Media Lab**

O Media Lab aceita ML para uma série de aplicações:

* social computing
* AI ethics
* computational arts
* neurocomputação
* interfaces cérebro-máquina
* IA + sociedade

E o processo de seleção é mais holístico.

Muitos candidatos entram **sem precisar competir diretamente com o bloco de Machine Learning teórico** do CSAIL.

---

### **■ Sloan (MIT School of Management) – Operations / Data**

Se seu ML aplica-se a:

* mercado
* supply chain
* previsão temporal
* otimização
* game theory
* teoria de decisão

Sloan pode ser uma porta fantástica.

---

# ⚠️ **2. EECS é o programa mais competitivo de todos**

Se você olhar estatísticas internas:

* EECS recebe MUITA candidatura
* principalmente para ML
* as taxas de aceite são baixíssimas
* alunos concorrendo com medalhistas, authors de NeurIPS, ICML, ICLR, etc.

Todos os **outros programas acima têm admissões mais humanas**, mantendo prestígio igual.

---

# 🎯 **3. Muitas pessoas entram no MIT por outro programa e trabalham com os mesmos PIs**

Exemplo real:

* Você pode entrar no **HST**, mas seu orientador ser alguém do CSAIL.
* Ou entrar por **BE**, mas pesquisar IA com Regina Barzilay.
* Ou por **IDSS**, trabalhando com Daskalakis.

MIT permite isso.

Ou seja:

➡️ você pode ter o MESMO orientador
➡️ o MESMO laboratório
➡️ a MESMA pesquisa
➡️ mas passando por um programa menos saturado e mais coerente com sua trajetória.

---

# 💡 **4. Seu background favorece algo interdisciplinar**

Você vem de:

* IA
* saúde
* computação biomédica
* análises reais
* produção de modelos aplicados
* ciência de dados aplicada

Indo para MIT, você pode se posicionar como:

> “o candidato que consegue cruzar inteligência artificial com impacto real em sistemas humanos/biológicos”.

E MIT **adora esse perfil**.

---

# ⭐ **5. Estratégia recomendada**

Aplicar para:

1️⃣ **EECS (CSAIL / AI+D)**
2️⃣ **HST ou BE (se foco também for medicina/biotech)**
3️⃣ **IDSS / Operations Research (se foco for probabilidade, decisão ou forecasting)**
4️⃣ **Media Lab (se quiser abertura criativa)**

Assim você:

* aumenta muito as chances de admissão
* continua podendo trabalhar com os mesmos professores
* não depende exclusivamente da fila de matadores do EECS

---

# 🧠 **6. E ainda tem uma vantagem psicológica**

Quando você aplica para mais programas:

* Você aumenta as portas por onde seu nome pode chegar à mesa de um PI.
* É possível que um professor goste de você e “puxe” sua candidatura mesmo fora do programa.

Isso acontece no MIT o tempo todo.

---

# 🏆 **Resumo**

### **Sim — faz MUITO sentido considerar outros PhDs do MIT.**

Porque:

* AI não é exclusivo do EECS.
* Outros programas são igualmente prestigiados.
* São mais alinhados ao seu perfil híbrido AI + Health/Bio.
* Têm menos competição direta.
* Você ainda pode trabalhar nos mesmos labs.

---

Se quiser, posso te dizer agora:

* **Em quais desses programas seu perfil parece mais forte hoje**,
* **qual teria maior chance de admissão no estado atual**,
* **como adaptar seu SOP para cada programa**.

Quer que eu faça isso?


## Medium Priority
[ ] Automatizar o pré-processamento do conjunto de dados  
[ ] Escrever o script de resumo do experimento  

## Low Priority
[ ] Formatar figuras da tese