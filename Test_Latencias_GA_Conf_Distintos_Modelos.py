"""
Benchmark de tiempos de entrenamiento por generación de Algoritmo Genético
para selección de características, usando distintos modelos (XGBoost, MLP de sklearn,
MLP-Torch) y distintas configuraciones hardware (CPU multi-hilo, GPU).

- El objetivo es comparar el tiempo que tarda en entrenar una generación completa
  del GA (es decir, entrenar una población de modelos con diferentes subconjuntos
  de features) bajo distintos escenarios de hardware y tamaños de dataset.
- El código es autocontenible, seguro para modificar y NO afecta a tu pipeline real de TFM.

"""

import os
import pandas as pd
import numpy as np
import time
import torch
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from datetime import datetime

# =========== CLASE GA MODIFICADA SOLO PARA BENCHMARK ============

class GA_Feature_Selection:
    """
    Implementación minimalista y modificada del Algoritmo Genético para selección de features,
    con soporte para medir el tiempo de entrenamiento en una o varias generaciones, y parametrizar
    explícitamente el hardware usado por XGBoost y MLP-Torch.

    Esta clase NO es la que usas en tu TFM; aquí puedes tunear lo que quieras.
    """
    def __init__(self, X_train, y_train, X_test, y_test, feature_names,
                 fitness_model="XGBRegressor", fitness_metric="rmse",
                 n_pop=25, n_gen=1, elite=5, mut_prob=0.5, random_state=42,
                 max_active=25, min_active=25, tournament_size=3,
                 # Parámetros explícitos para XGB
                 xgb_n_jobs=4, xgb_tree_method="auto", xgb_n_estimators=60, xgb_max_depth=4,
                 # Parámetro explícito para MLP-Torch
                 torch_device=None):
        """
        Inicialización de la clase. Aquí se reciben los parámetros de hardware y GA.
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.feature_names = list(feature_names)
        self.n_features = len(self.feature_names)
        self.fitness_model = fitness_model
        self.fitness_metric = fitness_metric
        self.metric_index = {"rmse": 0, "mae": 1, "mse": 2, "r2": 3}[self.fitness_metric]
        self.n_pop = n_pop          # Tamaño de la población (número de individuos por generación)
        self.n_gen = n_gen          # Número de generaciones a entrenar (para el benchmark normalmente 1)
        self.elite = elite          # Número de mejores individuos que pasan directo a la siguiente generación
        self.mut_prob = mut_prob    # Probabilidad de mutación de cada gen
        self.random_state = random_state
        self.max_active = max_active
        self.min_active = min_active
        self.tournament_size = tournament_size
        # Parámetros modelo XGBoost
        self.xgb_n_jobs = xgb_n_jobs
        self.xgb_tree_method = xgb_tree_method
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        # Dispositivo para MLP-Torch ("cpu" o "cuda")
        self.torch_device = torch_device
        np.random.seed(random_state)
        self.population = self.init_population()
        self.metrics_history = []
        self.best_individual = None
        self.best_score = np.inf
        self.torch_mlp = None
        self.torch_input_dim = self.n_features

    def init_population(self):
        """
        Inicializa la población de individuos. Cada uno es un vector binario que representa las
        features seleccionadas.
        """
        posibles_activas = np.full(self.n_pop, self.max_active)
        pop = []
        for n_active in posibles_activas:
            ind = np.zeros(self.n_features, dtype=int)
            selected = np.random.choice(self.n_features, n_active, replace=False)
            ind[selected] = 1
            pop.append(ind)
        return pop

    def calcular_metricas(self, y_true, y_pred):
        """
        Calcula RMSE, MAE, MSE y R2 para evaluar el modelo.
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        return rmse, mae, mse, r2

    def randomize_torch_weights(self):
        """
        Reinicializa los pesos del MLP-Torch para que cada individuo empiece desde cero.
        """
        if self.torch_mlp is not None:
            for layer in self.torch_mlp:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def fitness(self, individual, return_metrics=False):
        """
        Evalúa el individuo (vector binario de features) entrenando el modelo correspondiente
        y devolviendo la métrica deseada. 
        - XGBoost: usa los parámetros explícitos.
        - MLPRegressor: scikit-learn, sin soporte GPU/multihilo.
        - MLP-Torch: permite forzar CPU/GPU.
        """
        if np.sum(individual) == 0:
            resultado = (np.inf, np.inf, np.inf, -np.inf)
            return resultado if return_metrics else resultado[self.metric_index]
        idx = np.where(individual == 1)[0]
        X = self.X_train[:, idx]
        X_test = self.X_test[:, idx]
        y = self.y_train
        y_test = self.y_test
        try:
            if self.fitness_model == "XGBRegressor":
                # Aquí se fuerza el n_jobs y tree_method
                model = XGBRegressor(
                    n_estimators=self.xgb_n_estimators,
                    max_depth=self.xgb_max_depth,
                    n_jobs=self.xgb_n_jobs,
                    tree_method=self.xgb_tree_method,
                    random_state=self.random_state,
                    verbosity=0
                )
                model.fit(X, y)
                y_pred = model.predict(X_test)
            elif self.fitness_model == "MLP":
                # sklearn MLP - no permite GPU ni multihilo real
                model = model = MLPRegressor(
                    hidden_layer_sizes=(32,),
                    activation='relu',
                    solver='adam',
                    max_iter=250,
                    early_stopping=True,
                    n_iter_no_change=10,   # igual que paciencia de PyTorch
                    tol=1e-4,              # tolerancia razonable
                    random_state=self.random_state,
                    verbose=False
                )
                model.fit(X, y)
                y_pred = model.predict(X_test)
            elif self.fitness_model == "MLP-Torch":
                # Entrenamiento con PyTorch: aquí se fuerza CPU/GPU
                device = self.torch_device or ("cuda" if torch.cuda.is_available() else "cpu")
                if device == "cpu":
                    torch.set_num_threads(os.cpu_count())
                else:
                    torch.set_num_threads(1)
                torch.manual_seed(self.random_state)
                np.random.seed(self.random_state)
                if (self.torch_mlp is None) or (X.shape[1] != self.torch_input_dim):
                    self.torch_input_dim = X.shape[1]
                    self.torch_mlp = torch.nn.Sequential(
                        torch.nn.Linear(self.torch_input_dim, 32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32, 1)
                    ).to(device)

                self.randomize_torch_weights()
                
                X_torch = torch.tensor(X, dtype=torch.float32).to(device)
                y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
                X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
                optimizer = torch.optim.Adam(self.torch_mlp.parameters(), lr=0.001, weight_decay=0.0003)
                
                loss_fn = torch.nn.MSELoss()
                
                best_loss = float('inf')
                epochs_no_improve = 10
                n_iter_no_change = 10
                best_model_state = None
                
                for epoch in range(250):
                    self.torch_mlp.train()
                    optimizer.zero_grad()
                    output = self.torch_mlp(X_torch)
                    loss = loss_fn(output, y_torch)
                    loss.backward()
                    optimizer.step()
                    curr_loss = loss.item()
                    if curr_loss < best_loss - 1e-4:
                        best_loss = curr_loss
                        epochs_no_improve = 0
                        best_model_state = self.torch_mlp.state_dict()
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= n_iter_no_change:
                            break
                if best_model_state is not None:
                    self.torch_mlp.load_state_dict(best_model_state)
                self.torch_mlp.eval()
                with torch.no_grad():
                    y_pred = self.torch_mlp(X_test_torch).cpu().numpy().flatten()
            else:
                raise ValueError(f"Modelo '{self.fitness_model}' no soportado en este benchmark.")
            resultado = self.calcular_metricas(y_test, y_pred)
            return resultado if return_metrics else resultado[self.metric_index]
        except Exception as e:
            print(f"Error evaluando individuo: {e}")
            resultado = (np.inf, np.inf, np.inf, -np.inf)
            return resultado if return_metrics else resultado[self.metric_index]

    def select(self, scores):
        """
        Selección de individuos por torneo para el GA (aquí es funcional pero no relevante
        para el benchmark si solo mides una generación).
        """
        indices_ordenados = np.argsort(scores)
        elite = [self.population[i] for i in indices_ordenados[:self.elite]]
        rest = []
        while len(rest) < self.n_pop - self.elite:
            candidates = np.random.choice(self.n_pop, self.tournament_size, replace=False)
            best_idx = candidates[np.argmin(scores[candidates])]
            rest.append(self.population[best_idx].copy())
        return elite + rest

    def crossover(self, parent1, parent2):
        """
        Crossover genético simple con máscara aleatoria.
        """
        mask = np.random.randint(0, 2, self.n_features)
        child = (parent1 & mask) | (parent2 & ~mask)
        return child

    def mutate(self, individual):
        """
        Mutación simple de genes.
        """
        mutated = individual.copy()
        for i in range(self.n_features):
            if np.random.random() < self.mut_prob:
                mutated[i] = 1 - mutated[i]
        return mutated

    def fit(self, verbose=False):
        """
        Entrenamiento del GA. Aquí se controla el número de generaciones a medir.
        Si n_gen=1, solo se mide la generación inicial (población random).
        Si n_gen>1, el algoritmo evoluciona y mide el tiempo de todas las generaciones.
        """
        for gen in range(self.n_gen):
            t0 = time.time()
            metrics_all = []
            for ind in self.population:
                metrics_all.append(self.fitness(ind, True))
            metrics_all = np.array(metrics_all)
            scores = metrics_all[:, self.metric_index]
            best_idx = np.argmin(scores)
            best_metrics = metrics_all[best_idx]
            improved = best_metrics[self.metric_index] < getattr(self, 'best_score', np.inf)
            if improved:
                self.best_score = best_metrics[self.metric_index]
                self.best_individual = self.population[best_idx].copy()
            if verbose:
                print(f"Gen {gen+1}/{self.n_gen} | Score: {scores[best_idx]:.4f} | Tiempo gen: {time.time()-t0:.2f}s")
            self.metrics_history.append(best_metrics)
            if self.n_gen == 1:
                break
            selected = self.select(scores)
            next_pop = selected[:self.elite]
            while len(next_pop) < self.n_pop:
                p1, p2 = np.random.choice(selected, 2, replace=False)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_pop.append(child)
            self.population = next_pop

# =========== CONFIGURACIONES DEL BENCHMARK =====================

# Lista de tamaños de dataset a probar en el benchmark
DATASET_SIZES = [20000, 30000, 40000, 50000, 60000, 70000, 80000]
N_FEATURES = 25           # Número de features por individuo
N_GEN = 1                 # ¡Aquí definimos cuántas generaciones medir!
N_POP = 25                # Población por generación (número de modelos a entrenar por vez)

# Ruta al dataset (debe estar preparado y limpio)
DATA_PATH = "DATA/Dataset_All_Features_Transformado.csv"
TARGET_COL = 'TARGET_TREND_ANG_15_5'

# Definición de modelos y hardware a comparar
CONFIGS = [
    # ('XGB_CPU_4',   dict(model='XGBRegressor', xgb_n_jobs=4, xgb_tree_method='auto')),
    # ('XGB_CPU_8',   dict(model='XGBRegressor', xgb_n_jobs=8, xgb_tree_method='auto')),
    # ('XGB_CPU_16',  dict(model='XGBRegressor', xgb_n_jobs=16, xgb_tree_method='auto')),
    ('XGB_CPU_32',  dict(model='XGBRegressor', xgb_n_jobs=32, xgb_tree_method='auto')),
    ('XGB_GPU',     dict(model='XGBRegressor', xgb_n_jobs=4, xgb_tree_method='gpu_hist')),
    # ('MLP_SKLEARN', dict(model='MLPRegressor')),
    # ('MLP_TORCH_CPU', dict(model='MLP-Torch', torch_device='cpu')),
    ('MLP_TORCH_GPU', dict(model='MLP-Torch', torch_device='cuda')),
]

def load_and_prepare_data(dataset_size, n_features, seed=42):
    """
    Prepara el dataset mezclando varios stocks hasta completar el tamaño deseado,
    manteniendo la separación temporal (train = todos los años menos el último, test = último año).
    Devuelve X_train, y_train, X_test, y_test, features.
    """
    df = pd.read_csv(DATA_PATH, parse_dates=['Fecha'], sep=';')
    df = df.sort_values(['Symbol', 'Fecha']).reset_index(drop=True)
    # Excluye columnas no numéricas ni de features técnicas
    cols_excluir = ['Fecha', 'Symbol', 'Open', 'Close', 'High', 'Low', 'AdjClose', 'Volume']
    cols_excluir += [c for c in df.columns if c.startswith('EMA_')]
    all_features = [c for c in df.columns if c not in cols_excluir and not c.startswith('TARGET_')]
    features = all_features[:n_features]
    years = sorted(df['Fecha'].dt.year.unique())
    train_years = years[:-1]
    test_year = years[-1]
    df_train = df[df['Fecha'].dt.year.isin(train_years)].copy()
    df_test = df[df['Fecha'].dt.year == test_year].copy()
    # Baraja y coge las primeras filas necesarias para cada tamaño de train
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_train = df_train.iloc[:dataset_size]
    X_train = df_train[features]
    y_train = df_train[TARGET_COL]
    # Test = todo el último año
    X_test = df_test[features]
    y_test = df_test[TARGET_COL]
    return X_train, y_train, X_test, y_test, features

def run_benchmark(dataset_size, n_features, config, n_gen=N_GEN, n_pop=N_POP, elite=5, seed=42):
    """
    Ejecuta una ronda del benchmark para una configuración de modelo y tamaño de dataset.
    Cronometra el tiempo de entrenamiento completo del GA (1 o más generaciones).
    """
    X_train, y_train, X_test, y_test, features = load_and_prepare_data(dataset_size, n_features, seed=seed)
    model_name = config['model']
    # Extrae parámetros relevantes según modelo
    xgb_n_jobs = config.get('xgb_n_jobs', 4)
    xgb_tree_method = config.get('xgb_tree_method', 'auto')
    torch_device = config.get('torch_device', None)
    if model_name == 'XGBRegressor':
        fitness_model = 'XGBRegressor'
    elif model_name == 'MLPRegressor':
        fitness_model = 'MLP'
    elif model_name == 'MLP-Torch':
        fitness_model = 'MLP-Torch'
        if torch_device == 'cuda' and not torch.cuda.is_available():
            print("¡Advertencia! GPU no disponible, saltando MLP-Torch GPU.")
            return None
    else:
        raise ValueError("Modelo no soportado en este benchmark.")
    # Instancia la clase GA con los parámetros de hardware/modelo elegidos
    ga = GA_Feature_Selection(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        feature_names=features,
        fitness_model=fitness_model,
        fitness_metric='rmse',
        n_pop=n_pop, n_gen=n_gen, elite=elite, mut_prob=0.5, random_state=seed,
        max_active=n_features, min_active=n_features, tournament_size=3,
        xgb_n_jobs=xgb_n_jobs, xgb_tree_method=xgb_tree_method,
        torch_device=torch_device
    )
    # Cronometra el entrenamiento completo
    t0 = time.time()
    ga.fit(verbose=False)
    t1 = time.time()
    tiempo_total = t1 - t0
    print(f"Modelo: {model_name} | Size: {dataset_size} | Tiempo total: {tiempo_total:.2f}s")
    return tiempo_total

if __name__ == "__main__":
    resultados = []
    
    # Itera sobre todas las configuraciones y tamaños para comparar todos los escenarios
    for config_name, config in CONFIGS:
        for size in DATASET_SIZES:
            print(f"\n{'='*60}")
            print(f"Configuración: {config_name} | Tamaño dataset: {size}")
            try:
                tiempo = run_benchmark(size, N_FEATURES, config)
                if tiempo is not None:
                    resultados.append({
                        'Configuración': config_name,
                        'Tamaño_dataset': size,
                        'N_features': N_FEATURES,
                        'N_generaciones': N_GEN,
                        'Tiempo_segundos': tiempo
                    })
            except Exception as e:
                print(f"ERROR: {e}")
                resultados.append({
                    'Configuración': config_name,
                    'Tamaño_dataset': size,
                    'N_features': N_FEATURES,
                    'N_generaciones': N_GEN,
                    'Tiempo_segundos': None,
                    'Error': str(e)
                })
    
    # Guarda los resultados en CSV y muestra resumen por pantalla
    df_resultados = pd.DataFrame(resultados)
    fecha_hora = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = f"Resultados/Benchmark_GA_Tiempos_{fecha_hora}.csv"
    
    
    df_resultados.to_csv(output_path, index=False)
    print(f"\nResultados guardados en {output_path}")
    print(df_resultados)
