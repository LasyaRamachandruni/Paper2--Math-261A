## 00_clean-data.R
library(tidyverse)
places <- read_csv("../data/places_local_data_2025.csv")
glimpse(places)

# Top of unique measures
unique(places$MeasureId)[1:30]

# Diabetes-related entries
unique(places$Short_Question_Text[grepl("Diabetes", places$Short_Question_Text)])
unique(places$MeasureId[grepl("DIAB", places$MeasureId)])



target_ids <- c(
  "DIABETES",   # response
  "OBESITY",    # obesity
  "LPA",        # physical inactivity
  "BPHIGH",     # high blood pressure
  "CSMOKING",   # current smoking
  "CHECKUP"     # had a checkup
)

analysis_long <- places %>%
  filter(
    MeasureId %in% target_ids,
    Data_Value_Type == "Crude prevalence"
  ) %>%
  select(
    Year,
    StateAbbr,
    StateDesc,
    CountyName = LocationName,
    CountyFIPS = LocationID,
    MeasureId,
    Data_Value
  )

analysis_wide <- analysis_long %>%
  pivot_wider(
    names_from = MeasureId,
    values_from = Data_Value
  ) %>%
  drop_na(DIABETES, OBESITY, LPA, BPHIGH, CSMOKING, CHECKUP)

glimpse(analysis_wide)
summary(analysis_wide$DIABETES)

# Baseline multiple linear regression
baseline_mod <- lm(
  DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
  data = analysis_wide
)

summary(baseline_mod)

# Basic diagnostics
par(mfrow = c(2, 2))
plot(baseline_mod)
par(mfrow = c(1, 1))

set.seed(261)

n <- nrow(analysis_wide)
idx <- sample(seq_len(n))

train_end <- floor(0.7 * n)
valid_end <- floor(0.85 * n)

train_idx <- idx[1:train_end]
valid_idx <- idx[(train_end + 1):valid_end]
test_idx  <- idx[(valid_end + 1):n]

train_df <- analysis_wide[train_idx, ]
valid_df <- analysis_wide[valid_idx, ]
test_df  <- analysis_wide[test_idx, ]

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

baseline_train <- lm(
  DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
  data = train_df
)

valid_rmse_baseline <- rmse(
  valid_df$DIABETES,
  predict(baseline_train, newdata = valid_df)
)
valid_rmse_baseline

library(splines)

obesity_knots <- quantile(train_df$OBESITY, probs = c(0.25, 0.5, 0.75))

spline_train <- lm(
  DIABETES ~ bs(OBESITY, knots = obesity_knots, degree = 3) +
    LPA + BPHIGH + CSMOKING + CHECKUP,
  data = train_df
)

valid_rmse_spline <- rmse(
  valid_df$DIABETES,
  predict(spline_train, newdata = valid_df)
)

c(baseline = valid_rmse_baseline,
  spline   = valid_rmse_spline)

library(glmnet)

# Build model matrix on training data (drop intercept column)
x_train <- model.matrix(
  DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
  data = train_df
)[, -1]

y_train <- train_df$DIABETES

set.seed(261)
lasso_cv <- cv.glmnet(
  x_train,
  y_train,
  alpha = 1,
  standardize = TRUE
)

plot(lasso_cv)

# Optimal lambdas
lasso_cv$lambda.min
lasso_cv$lambda.1se

# Coefficients at lambda.min
coef(lasso_cv, s = "lambda.min")

x_valid <- model.matrix(
  DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
  data = valid_df
)[, -1]

x_test <- model.matrix(
  DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
  data = test_df
)[, -1]

valid_rmse_lasso <- rmse(
  valid_df$DIABETES,
  predict(lasso_cv, newx = x_valid, s = "lambda.min")
)

test_rmse_lasso <- rmse(
  test_df$DIABETES,
  predict(lasso_cv, newx = x_test, s = "lambda.min")
)

c(
  baseline_valid = valid_rmse_baseline,
  spline_valid   = valid_rmse_spline,
  lasso_valid    = valid_rmse_lasso,
  lasso_test     = test_rmse_lasso
)

# Since all five predictors are selected, refit OLS on train data
post_sel_mod <- lm(
  DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
  data = train_df
)

summary(post_sel_mod)

# Refit spline model on train + validation, evaluate on test
train_valid_df <- bind_rows(train_df, valid_df)

obesity_knots_tv <- quantile(train_valid_df$OBESITY,
                             probs = c(0.25, 0.5, 0.75))

spline_tv <- lm(
  DIABETES ~ bs(OBESITY, knots = obesity_knots_tv, degree = 3) +
    LPA + BPHIGH + CSMOKING + CHECKUP,
  data = train_valid_df
)

test_rmse_spline <- rmse(
  test_df$DIABETES,
  predict(spline_tv, newdata = test_df)
)

test_rmse_spline

library(car)

# For baseline model on full data (or on train_valid_df)
vif_baseline <- vif(
  lm(DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
     data = analysis_wide)
)

vif_baseline

library(splines)

# Make sure knots and model exist
obesity_knots_tv <- quantile(analysis_wide$OBESITY,
                             probs = c(0.25, 0.5, 0.75))

spline_full <- lm(
  DIABETES ~ bs(OBESITY, knots = obesity_knots_tv, degree = 3) +
    LPA + BPHIGH + CSMOKING + CHECKUP,
  data = analysis_wide
)

# Cook's distance
cooks <- cooks.distance(spline_full)

plot(cooks, type = "h",
     ylab = "Cook's distance", main = "Influential counties")
abline(h = 4 / nrow(analysis_wide), col = "red", lty = 2)

# Define infl_idx here
infl_idx <- which(cooks > 4 / nrow(analysis_wide))

length(infl_idx)
head(infl_idx)

# Drop influential points
analysis_no_infl <- analysis_wide[-infl_idx, ]

spline_no_infl <- lm(
  DIABETES ~ bs(OBESITY, knots = obesity_knots_tv, degree = 3) +
    LPA + BPHIGH + CSMOKING + CHECKUP,
  data = analysis_no_infl
)

# Compare key coefficients
coef_full    <- summary(spline_full)$coefficients
coef_no_infl <- summary(spline_no_infl)$coefficients

coef_full
coef_no_infl

# Model matrices for validation and test sets
x_valid <- model.matrix(
  DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
  data = valid_df
)[, -1]

x_test <- model.matrix(
  DIABETES ~ OBESITY + LPA + BPHIGH + CSMOKING + CHECKUP,
  data = test_df
)[, -1]

# Validation RMSE (you already had something like this)
lasso_valid <- rmse(
  valid_df$DIABETES,
  predict(lasso_cv, newx = x_valid, s = "lambda.min")
)

# Test RMSE: this creates lasso_test
lasso_test <- rmse(
  test_df$DIABETES,
  predict(lasso_cv, newx = x_test, s = "lambda.min")
)

c(lasso_valid = lasso_valid,
  lasso_test  = lasso_test)

analysis_no_infl <- analysis_wide[-infl_idx, ]

spline_no_infl <- lm(
  DIABETES ~ bs(OBESITY, knots = obesity_knots_tv, degree = 3) +
    LPA + BPHIGH + CSMOKING + CHECKUP,
  data = analysis_no_infl
)

summary(spline_full)$coefficients
summary(spline_no_infl)$coefficients

rmse_summary <- tibble::tibble(
  model = c("Baseline linear", "Spline (validation)", "Lasso (validation)",
            "Spline (test)", "Lasso (test)"),
  RMSE  = c(
    valid_rmse_baseline,
    valid_rmse_spline,
    valid_rmse_lasso,
    test_rmse_spline,
    lasso_test
  )
)

rmse_summary

anova(baseline_mod, spline_full)



