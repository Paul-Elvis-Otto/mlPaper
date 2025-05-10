import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import math  # Needed for floor/ceil if using percentage split index precisely

# --- Add tqdm import ---
from tqdm.auto import tqdm  # Auto-detects notebook or console environment
# ---

# MLX Imports
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Scikit-learn Imports
# from sklearn.model_selection import train_test_split # Correctly removed
from sklearn.preprocessing import StandardScaler

# Linear Models
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Non-Linear Models
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

# Metrics & Version
from sklearn.metrics import r2_score, mean_squared_error
import sklearn

# Optional: Scipy for QQ plot

# Ignore warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Check Scikit-learn version
print(f"Scikit-learn version: {sklearn.__version__}")


# 1. ── Data Loading and Initial Setup ───────────────────────────────────────
try:
    df = pd.read_csv("../temp/df_imputed.csv")
    print(f"Data shape: {df.shape}")
    print(f"Any missing values? {df.isna().sum().sum()} (should be 0)")
except FileNotFoundError:
    print("Error: df_imputed.csv not found. Creating dummy data.")
    n_samples = 1000
    n_features = 50
    feature_names = [f"feature_{i}" for i in range(n_features)] + [
        "some_correlation_measure"
    ]
    extra_cols = [
        "country_name",
        "country_text_id",
        "country_id",
        "year",
        "v2x_corr",
        "e_gdppc_imputed_rf",
        "e_v2x_corr_4C",
        "v2x_polyarchy_imputed_rf",
        "v2x_delibdem_imputed",
        "v2x_pubcorr",
        "v2x_execorr",
        "v2lgcrrpt",
        "v2jucorrdc",
        "v2exbribe",
        "v2exembez",
        "v2excrptps",
        "v2exthftps",
    ]
    all_cols = list(set(feature_names + extra_cols))
    df = pd.DataFrame(np.random.rand(n_samples, len(all_cols)), columns=all_cols)
    df["v2x_corr"] = (
        np.sin(df["feature_0"] * np.pi) * 0.5
        + df["feature_1"] * -0.3
        + np.square(df["feature_2"]) * 0.2
        + np.random.randn(n_samples) * 0.5
    )
    if "e_gdppc_imputed_rf" in df.columns:
        df["e_gdppc_imputed_rf"] = np.exp(df["feature_2"] * 3 + 10)
    for col in [
        "v2x_pubcorr",
        "v2x_execorr",
        "v2lgcrrpt",
        "v2jucorrdc",
        "v2exbribe",
        "v2exembez",
        "v2excrptps",
        "v2exthftps",
        "e_v2x_corr_4C",
        "v2x_polyarchy_imputed_rf",
        "v2x_delibdem_imputed",
        "some_correlation_measure",
    ]:
        if col in df.columns:
            df[col] = (
                df["feature_0"] * (0.2 + 0.6 * np.random.rand(n_samples))
                + np.random.rand(n_samples) * 0.1
            )
    df["country_name"] = [f"Country_{i % 10}" for i in range(n_samples)]
    df["country_text_id"] = [f"C{i % 10}" for i in range(n_samples)]
    df["country_id"] = [i % 10 for i in range(n_samples)]
    df["year"] = [1950 + i // 20 for i in range(n_samples)]


# 2. ── Define Target & Predictors, Drop Problematic Columns ────────────────
target = "v2x_corr"
base_drop_cols = [
    "country_name",
    "country_text_id",
    "country_id",
    "year",
    target,
]  # Keep 'year' initially
base_drop_cols = [col for col in base_drop_cols if col in df.columns and col != "year"]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
predictors = [
    c for c in numeric_cols if c not in base_drop_cols and c != target and c != "year"
]
print(
    f"Identified {len(predictors)} potential numeric predictors initially (excluding year)."
)
manual_drop_cols = [
    "e_v2x_corr_3C",
    "e_v2x_corr_4C",
    "e_v2x_corr_5C",
    "v2x_delibdem_imputed",
    "v2xel_frefair_imputed",
    "v2xdl_delib_imputed",
    "e_v2xdl_delib_4C_imputed",
    "e_v2xdl_delib_5C_imputed",
    "e_v2x_delibdem_3C_imputed",
    "e_v2x_delibdem_4C_imputed",
    "e_v2x_delibdem_5C_imputed",
    "e_v2xel_frefair_3C_imputed",
    "e_v2xel_frefair_4C_imputed",
    "e_v2xel_frefair_5C_imputed",
    "v2clpolcl_imputed_rf",
    "v2peasjsoecon_imputed_rf",
    "v2peapsgen_imputed_rf",
    "v2peasjgen_imputed_rf",
    "v2xpe_exlecon_imputed_rf",
    "v2x_polyarchy_imputed_rf",
    "v2x_libdem_imputed_rf",
    "v2x_partipdem_imputed_rf",
    "v2x_egaldem_imputed_rf",
    "v2x_mpi_imputed_rf",
    "v2x_liberal_imputed_rf",
    "v2x_regime_imputed_rf",
    "v2x_regime_amb_imputed_rf",
    "v2x_EDcomp_thick_imputed_rf",
    "e_v2x_egaldem_3C_imputed_rf",
    "e_v2x_egaldem_4C_imputed_rf",
    "e_v2x_egaldem_5C_imputed_rf",
    "e_v2x_libdem_3C_imputed_rf",
    "e_v2x_libdem_4C_imputed_rf",
    "e_v2x_libdem_5C_imputed_rf",
    "e_v2x_mpi_5C_imputed_rf",
    "e_v2x_EDcomp_thick_5C_imputed_rf",
    "e_v2x_liberal_4C_imputed_rf",
    "e_v2x_liberal_5C_imputed_rf",
    "e_v2x_polyarchy_5C_imputed_rf",
    "v2x_pubcorr",
    "v2x_execorr",
    "v2lgcrrpt",
    "v2jucorrdc",
    "v2exbribe",
    "v2exembez",
    "v2excrptps",
    "v2exthftps",
]
corr_containing_cols = [p for p in predictors if "corr" in p.lower()]
print(f"Found {len(corr_containing_cols)} predictors containing 'corr'.")
cols_to_drop = list(set(manual_drop_cols + corr_containing_cols))
cols_to_drop_existing = [col for col in cols_to_drop if col in predictors]
predictors = [p for p in predictors if p not in cols_to_drop_existing]
print(
    f"Dropped {len(cols_to_drop_existing)} columns (manual list + 'corr' containing)."
)


# 3. ── Log Transform GDP ────────────────────────────────────────────────────
gdp_var = "e_gdppc_imputed_rf"
if gdp_var in predictors and gdp_var in df.columns:
    min_val = df[gdp_var].min()
    if min_val <= 0:
        offset = -min_val + 1e-6
        print(f"Warning: '{gdp_var}' non-positive. Adding {offset:.2e} before log.")
        df["log_gdppc"] = np.log(df[gdp_var] + offset)
    else:
        df["log_gdppc"] = np.log(df[gdp_var])
    predictors = [p for p in predictors if p != gdp_var] + ["log_gdppc"]
    print(f"Replaced {gdp_var} with log_gdppc.")
elif gdp_var in cols_to_drop_existing:
    print(f"'{gdp_var}' was previously dropped.")
else:
    print(f"'{gdp_var}' not found or already processed.")
predictors = [p for p in predictors if p in df.columns]
if not predictors:
    raise ValueError("No valid predictor columns remain.")
print(f"Using {len(predictors)} final predictors for the models.")


# 4. ── Prepare Data: Select final columns & Sort by Year ───────────────────
time_col = "year"
if time_col not in df.columns:
    raise ValueError(f"Time column '{time_col}' not found.")
print(f"\n--- Sorting Data by '{time_col}' for Time-Based Split ---")
df_cleaned = (
    df[[time_col] + predictors + [target]]
    .dropna()
    .sort_values(by=time_col)
    .reset_index(drop=True)
)
print(f"Shape after dropping NAs and sorting: {df_cleaned.shape}")
X_sorted = df_cleaned[predictors]
y_sorted = df_cleaned[target]


# 5. ── Time-Based Train/Test Split ──────────────────────────────────────────
train_ratio = 0.80
split_index = math.floor(len(df_cleaned) * train_ratio)
if split_index == 0 or split_index == len(df_cleaned):
    raise ValueError(f"Train ratio {train_ratio} results in empty train or test set.")
X_train_pd = X_sorted.iloc[:split_index]
X_test_pd = X_sorted.iloc[split_index:]
y_train_pd = y_sorted.iloc[:split_index]
y_test_pd = y_sorted.iloc[split_index:]
train_years = df_cleaned.iloc[:split_index][time_col]
test_years = df_cleaned.iloc[split_index:][time_col]
print("\n--- Performing Time-Based Split ---")
print(
    f"Splitting at index {split_index} ({(train_ratio * 100):.0f}% train / {((1 - train_ratio) * 100):.0f}% test)"
)
print(f"Training data time range: {train_years.min()} - {train_years.max()}")
print(f"Test data time range:     {test_years.min()} - {test_years.max()}")
print(f"Train set shape: {X_train_pd.shape}, Test set shape: {X_test_pd.shape}")
y_train_np = y_train_pd.values
y_test_np = y_test_pd.values


# 6. ── Standardize Features (Fit on Train, Transform Train & Test) ──────────
print("\n--- Standardizing Features (Fit on Train, Transform Train & Test) ---")
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_pd)
X_test_np = scaler.transform(X_test_pd)
print("Standardization complete using training set parameters.")


# 7. ── Prepare Data for MLX (Convert NumPy to MLX) ─────────────────────────
X_train_mx = mx.array(X_train_np, dtype=mx.float32)
X_test_mx = mx.array(X_test_np, dtype=mx.float32)
y_train_mx = mx.array(y_train_np, dtype=mx.float32).reshape(-1, 1)


# 8. ── Define MLX Linear Regression Model & Training Function (Corrected) ──
class MlxLinearRegression(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"Num features must be > 0, got {num_features}")
        self.linear = nn.Linear(num_features, 1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(x)


def train_mlx_model(
    X_train, y_train, num_features, epochs=500, batch_size=128, lr=0.01
):
    print(f"  Training MLX Linear (Epochs: {epochs}, Batch: {batch_size}, LR: {lr})...")
    model = MlxLinearRegression(num_features)
    mx.eval(model.parameters())  # Initialize before defining loss/grad

    def mse_loss(model, x, y):  # Define local loss function on a new line
        return nn.losses.mse_loss(model(x), y, reduction="mean")

    loss_and_grad_fn = nn.value_and_grad(model, mse_loss)
    optimizer = optim.Adam(learning_rate=lr)
    start_time = time.time()
    model.train()
    for epoch in tqdm(
        range(epochs), desc="    MLX Epochs", leave=False, ncols=100, mininterval=0.5
    ):
        permutation = mx.array(np.random.permutation(X_train.shape[0]))
        for i in range(0, X_train.shape[0], batch_size):
            batch_indices = permutation[i : i + batch_size]
            X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]
            loss, grads = loss_and_grad_fn(model, X_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)  # Ensure updates happen
    end_time = time.time()
    training_time = end_time - start_time
    print(f"  MLX Training finished in {training_time:.2f} seconds.")
    model.eval()
    return model, training_time


# 9. ── Define Models to Benchmark ───────────────────────────────────────────
models_to_benchmark = {
    "MLX Linear": None,
    "Linear Regression (SKL)": SkLinearRegression(),
    "RidgeCV (SKL)": RidgeCV(alphas=np.logspace(-6, 6, 13), cv=5),
    "LassoCV (SKL)": LassoCV(cv=5, random_state=42, max_iter=5000, tol=0.001),
    "ElasticNetCV (SKL)": ElasticNetCV(cv=5, random_state=42, max_iter=5000, tol=0.001),
    "Random Forest (SKL)": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Hist Gradient Boosting (SKL)": HistGradientBoostingRegressor(random_state=42),
    "SVR (RBF Kernel - SKL)": SVR(),
}
num_features = X_train_np.shape[1]
if num_features <= 0:
    raise ValueError("No features left after dropping.")


# 10. ── Run Benchmark Loop & Store Trained Models (with tqdm) ──────────────
print("\n--- Starting Model Benchmarking (Time-Based Split) ---")
benchmark_results = []
trained_models = {}

for name, model_instance in tqdm(
    models_to_benchmark.items(), desc="Benchmarking Models", ncols=100
):
    preds_np = None
    training_time = None
    current_trained_model = None
    try:
        if name == "MLX Linear":
            current_trained_model, training_time = train_mlx_model(
                X_train_mx, y_train_mx, num_features, epochs=500
            )
            print(f"  Predicting with {name}...")
            start_pred_time = time.time()
            preds_mx = current_trained_model(X_test_mx)
            mx.eval(preds_mx)
            preds_np = np.array(preds_mx).flatten()
            print(
                f"  {name} prediction finished in {time.time() - start_pred_time:.2f} seconds."
            )
        else:
            current_trained_model = model_instance
            print(f"  Training {name}...")
            start_time = time.time()
            current_trained_model.fit(X_train_np, y_train_np)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"  {name} Training finished in {training_time:.2f} seconds.")
            print(f"  Predicting with {name}...")
            start_pred_time = time.time()
            preds_np = current_trained_model.predict(X_test_np)
            print(
                f"  {name} prediction finished in {time.time() - start_pred_time:.2f} seconds."
            )
        print(f"  Evaluating {name}...")
        r2 = r2_score(y_test_np, preds_np)
        mse = mean_squared_error(y_test_np, preds_np)
        rmse = np.sqrt(mse)
        print(f"  {name} Results -> Test R²: {r2:.4f}, Test RMSE: {rmse:.4f}")
        benchmark_results.append(
            {
                "Model": name,
                "Test R²": r2,
                "Test RMSE": rmse,
                "Training Time (s)": training_time,
            }
        )
        if name in [
            "RidgeCV (SKL)",
            "LassoCV (SKL)",
            "MLX Linear",
            "Random Forest (SKL)",
            "Hist Gradient Boosting (SKL)",
        ]:
            trained_models[name] = current_trained_model
    except Exception as e:
        print(f"\n  ERROR processing {name}: {e}")
        import traceback

        traceback.print_exc()
        benchmark_results.append(
            {
                "Model": name,
                "Test R²": np.nan,
                "Test RMSE": np.nan,
                "Training Time (s)": training_time
                if training_time is not None
                else np.nan,
            }
        )


# 11. ── Display Benchmark Results ───────────────────────────────────────────
print("\n--- Benchmark Results (Time-Based Split) ---")
results_df = pd.DataFrame(benchmark_results)
if not results_df.empty:
    results_df = results_df.sort_values(by="Test R²", ascending=False).reset_index(
        drop=True
    )
    print(results_df.to_string(index=False, float_format="%.4f"))
else:
    print("No benchmark results were generated.")


# 12. ── Plot Benchmark Results ───────────────────────────────────────────────
if not results_df.empty and not results_df["Test R²"].isnull().all():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Model Performance Benchmark (Time-Based Split - Excl. 'corr' Vars)")
    plot_df_r2 = results_df.dropna(subset=["Test R²"]).sort_values("Test R²")
    plot_df_rmse = results_df.dropna(subset=["Test RMSE"]).sort_values(
        "Test RMSE", ascending=False
    )
    if not plot_df_r2.empty:
        axes[0].barh(plot_df_r2["Model"], plot_df_r2["Test R²"], color="skyblue")
        axes[0].set_xlabel("R² Score")
        axes[0].set_title("R² Comparison")
        min_r2 = plot_df_r2["Test R²"].min()
        axes[0].set_xlim(left=max(-0.1, min_r2 - 0.05), right=1.0)
        axes[0].grid(axis="x", linestyle="--", alpha=0.6)
    else:
        axes[0].set_title("R² Comparison (No Data)")
    if not plot_df_rmse.empty:
        axes[1].barh(
            plot_df_rmse["Model"], plot_df_rmse["Test RMSE"], color="lightcoral"
        )
        axes[1].set_xlabel("RMSE")
        axes[1].set_title("RMSE Comparison")
        axes[1].grid(axis="x", linestyle="--", alpha=0.6)
    else:
        axes[1].set_title("RMSE Comparison (No Data)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
elif not results_df.empty:
    print("\nPlotting skipped: All models failed or produced NaN results.")
else:
    print("\nPlotting skipped: No results available.")


# 13. ── Extract and Analyze Linear Model Coefficients ──────────────────────
print("\n--- Analyzing Coefficients from Linear Models (Time-Based Split) ---")

N_TOP_FEATURES_LINEAR = 15
linear_coef_dfs = {}
model_name = "RidgeCV (SKL)"

if model_name in trained_models:
    print(f"\nProcessing Coefficients: {model_name}")
    ridge_model = trained_models[model_name]
    ridge_coefs = pd.DataFrame(
        {"feature": predictors, "coefficient": ridge_model.coef_}
    )
    ridge_coefs["abs_coef"] = ridge_coefs["coefficient"].abs()
    ridge_coefs = ridge_coefs.sort_values("abs_coef", ascending=False).reset_index(
        drop=True
    )
    linear_coef_dfs[model_name] = ridge_coefs

    print(f"--- Top {N_TOP_FEATURES_LINEAR} Predictors for {model_name} ---")
    print(
        ridge_coefs[["feature", "coefficient"]]
        .head(N_TOP_FEATURES_LINEAR)
        .to_string(index=False, float_format="%.4f")
    )
else:
    print(f"\n{model_name} not found for coefficient analysis.")


# --- Corrected LassoCV Indentation ---
model_name = "LassoCV (SKL)"
if model_name in trained_models:
    print(f"\nProcessing Coefficients: {model_name}")
    lasso_model = trained_models[model_name]
    lasso_coefs = pd.DataFrame(
        {"feature": predictors, "coefficient": lasso_model.coef_}
    )
    lasso_coefs["abs_coef"] = lasso_coefs["coefficient"].abs()
    lasso_coefs_nonzero = lasso_coefs[lasso_coefs["coefficient"] != 0].copy()
    lasso_coefs_nonzero = lasso_coefs_nonzero.sort_values(
        "abs_coef", ascending=False
    ).reset_index(drop=True)
    linear_coef_dfs[model_name] = lasso_coefs_nonzero
    print(f"  LassoCV selected {len(lasso_coefs_nonzero)} non-zero features.")
    print(f"--- Top {N_TOP_FEATURES_LINEAR} (Non-Zero) Predictors for {model_name} ---")
    if (
        not lasso_coefs_nonzero.empty
    ):  # This line needs to be at the same level as the print above it
        print(
            lasso_coefs_nonzero[["feature", "coefficient"]]
            .head(N_TOP_FEATURES_LINEAR)
            .to_string(index=False, float_format="%.4f")
        )
    else:  # This line needs to be at the same level as the if above it
        print("  LassoCV selected 0 features.")
else:
    print(f"\n{model_name} not found for coefficient analysis.")
# --- End Correction ---

# 13b. ── Extract and Analyze MLX Linear Model Coefficients ────────────
model_name = "MLX Linear"

if model_name in trained_models:
    print(f"\nProcessing Coefficients: {model_name}")
    mlx_model = trained_models[model_name]

    # make sure the weights tensor is materialized
    mx.eval(mlx_model.linear.weight)

    # pull out a flat numpy array of coefficients
    mlx_weights = np.array(mlx_model.linear.weight.squeeze())

    # sanity check length
    if len(predictors) == len(mlx_weights):
        mlx_coefs = pd.DataFrame({"feature": predictors, "coefficient": mlx_weights})
        mlx_coefs["abs_coef"] = mlx_coefs["coefficient"].abs()
        mlx_coefs = mlx_coefs.sort_values("abs_coef", ascending=False).reset_index(
            drop=True
        )
        linear_coef_dfs[model_name] = mlx_coefs

        print(f"--- Top {N_TOP_FEATURES_LINEAR} Predictors for {model_name} ---")
        print(
            mlx_coefs[["feature", "coefficient"]]
            .head(N_TOP_FEATURES_LINEAR)
            .to_string(index=False, float_format="%.4f")
        )
    else:
        msg = (
            f"  ERROR: Mismatch predictor count "
            f"({len(predictors)}) vs MLX weights ({len(mlx_weights)})."
        )
        print(msg)
else:
    print(f"\n{model_name} not found for coefficient analysis.")


# 14. ── Extract and Analyze Tree Model Feature Importances ─────────────
print(
    "\n--- Analyzing Feature Importances from Tree-Based Models (Time-Based Split) ---"
)

N_TOP_FEATURES_TREE = 20
tree_importance_dfs = {}

# Random Forest
model_name = "Random Forest (SKL)"
if model_name in trained_models:
    print(f"\nProcessing Feature Importances: {model_name}")
    rf_model = trained_models[model_name]

    rf_importances = pd.DataFrame(
        {"feature": predictors, "importance": rf_model.feature_importances_}
    )
    rf_importances = rf_importances.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)
    tree_importance_dfs[model_name] = rf_importances

    print(f"--- Top {N_TOP_FEATURES_TREE} Features for {model_name} ---")
    print(
        rf_importances.head(N_TOP_FEATURES_TREE).to_string(
            index=False, float_format="%.4f"
        )
    )
else:
    print(f"\n{model_name} not found for importance analysis.")

# HistGradientBoosting (skip detailed example)
model_name = "Hist Gradient Boosting (SKL)"
if model_name in trained_models:
    print(f"\nProcessing Feature Importances: {model_name}")
    print(
        f"  Feature importance extraction for {model_name} "
        "(requires permutation_importance; skipped in this example)."
    )
else:
    print(f"\n{model_name} not found for importance analysis.")


# 15. ── Visualize Coefficients & Importances ────────────────────────────
# ---- 15a: Linear Coefficients ----
print(
    "\n--- Visualizing Top Linear Predictor Coefficients "
    "(Example: RidgeCV - Time-Based Split) ---"
)

model_to_plot_lin = "RidgeCV (SKL)"
if (
    model_to_plot_lin in linear_coef_dfs
    and not linear_coef_dfs[model_to_plot_lin].empty
):
    plot_data = (
        linear_coef_dfs[model_to_plot_lin]
        .head(N_TOP_FEATURES_LINEAR)
        .sort_values("abs_coef")
    )

    plt.figure(figsize=(10, max(6, N_TOP_FEATURES_LINEAR / 2.5)))
    colors = ["lightcoral" if c < 0 else "skyblue" for c in plot_data["coefficient"]]
    plt.barh(plot_data["feature"], plot_data["coefficient"], color=colors)
    plt.xlabel("Coefficient Value (Standardized)")
    plt.ylabel("Feature")
    plt.title(
        f"Top {N_TOP_FEATURES_LINEAR} Predictor Coefficients "
        f"({model_to_plot_lin} - Time Split - No 'corr' Vars)"
    )
    plt.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print(f"Skipping linear coefficient plot: No data found for {model_to_plot_lin}.")


# ---- 15b: Tree Importances ----
print(
    "\n--- Visualizing Top Tree Feature Importances "
    "(Example: Random Forest - Time-Based Split) ---"
)

model_to_plot_tree = "Random Forest (SKL)"
if (
    model_to_plot_tree in tree_importance_dfs
    and not tree_importance_dfs[model_to_plot_tree].empty
):
    plot_data = (
        tree_importance_dfs[model_to_plot_tree]
        .head(N_TOP_FEATURES_TREE)
        .sort_values("importance")
    )

    plt.figure(figsize=(10, max(6, N_TOP_FEATURES_TREE / 2.5)))
    plt.barh(plot_data["feature"], plot_data["importance"])
    plt.xlabel("Feature Importance Score (Gini importance or similar)")
    plt.ylabel("Feature")
    plt.title(
        f"Top {N_TOP_FEATURES_TREE} Feature Importances "
        f"({model_to_plot_tree} - Time Split - No 'corr' Vars)"
    )
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
else:
    print(f"Skipping tree importance plot: No data found for {model_to_plot_tree}.")


print("\n--- Script Finished ---")
