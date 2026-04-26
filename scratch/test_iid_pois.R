library(rustyINLA)
library(CASdatasets)
data("freMTPL2freq")
df <- freMTPL2freq[1:5000, ]
train_idx <- sample(1:nrow(df), 0.8 * nrow(df))
train_data <- df[train_idx, ]
fit <- rusty_inla(
  formula = ClaimNb ~ 1 + f(VehBrand, model="iid") + f(Region, model="iid"),
  data = train_data, 
  family = "poisson"
)
