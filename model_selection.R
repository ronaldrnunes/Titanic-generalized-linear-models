library(tidymodels)
library(workflowsets)

# Dados e categorias de referência
dados <- read.csv("titanic.csv") |>
  transform(
    vivo = relevel(factor(vivo), ref = "nao"),
    classe = relevel(factor(classe), ref = "III"),
    sexo = relevel(factor(sexo), ref = "masculino"),
    sozinho = relevel(factor(sozinho), ref = "nao")
  )

set.seed(123)
dobras <- vfold_cv(dados, v = 5, strata = vivo)

# Preditores lineares a comparar
preditores <- list(
  idade = vivo ~ idade,
  simples = vivo ~ classe + idade + sexo + sozinho,
  interacao = vivo ~ classe + idade + sexo * sozinho
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
