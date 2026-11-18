library(tidymodels)
library(workflowsets)

dados <- read.csv("titanic.csv") |>
  transform(
    vivo = relevel(factor(vivo), ref = "nao"),
    classe = relevel(factor(classe), ref = "III"),
    sexo = relevel(factor(sexo), ref = "masculino"),
    sozinho = relevel(factor(sozinho), ref = "nao")
  )

set.seed(6)
dobras <- vfold_cv(dados, v = 5, strata = vivo)

preditores <- list(
  idade = vivo ~ idade,
  aditivo = vivo ~ classe + idade + sexo + sozinho,
  interacao_sexo_sozinho = vivo ~ classe + idade + sexo * sozinho,
  interacao_sexo_classe = vivo ~ idade + sozinho + sexo * classe,
  interacao_sexo_idade = vivo ~ classe + sozinho + sexo * idade,
  interacao_classe_idade = vivo ~ sexo + sozinho + classe * idade,
  interacoes_sexo = vivo ~ sozinho + sexo * classe + sexo * idade,
  interacao_sozinho_classe = vivo ~ classe + idade + sexo + sozinho + sozinho:classe,
  interacao_sozinho_idade = vivo ~ classe + idade + sexo + sozinho + sozinho:idade
)

# Mesma regressão logística, com links diferentes
modelos <- list(
  logit = logistic_reg() |> set_engine("glm", family = binomial("logit")),
  probit = logistic_reg() |> set_engine("glm", family = binomial("probit")),
  cloglog = logistic_reg() |> set_engine("glm", family = binomial("cloglog"))
)

resultados <- workflow_set(preditores, modelos) |>
  workflow_map(
    "fit_resamples",
    resamples = dobras,
    metrics = metric_set(roc_auc, accuracy, brier_class)
  )

# Comparação: AUC e acurácia maiores; Brier menor
comparacao <- collect_metrics(resultados) |>
  select(wflow_id, .metric, mean) |>
  pivot_wider(names_from = .metric, values_from = mean) |>
  arrange(desc(roc_auc), desc(accuracy), brier_class)

comparacao
