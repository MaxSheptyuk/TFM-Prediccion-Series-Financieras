import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from time import perf_counter
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import ElasticNet, Ridge, Lasso, LinearRegression
from sklearn.svm import SVR


class GA_Feature_Selection:
    """
    Algoritmo Genético para Selección de Variables (Features) en problemas de regresión.

    Permite seleccionar de forma inteligente subconjuntos de variables para modelos tipo XGBoost, ElasticNet o Ridge.
    Permite ajustar el número mínimo y máximo de variables activas, el tamaño de la población, el número de generaciones,
    la presión de torneo, y otros hiperparámetros.

    Parámetros del constructor:
    --------------------------
    X_train, y_train: array-like
        Matriz de variables predictoras y vector objetivo para entrenamiento.

    X_test, y_test: array-like
        Matriz de variables predictoras y vector objetivo para test.

    feature_names: list of str
        Lista con los nombres de todas las features posibles (en el mismo orden que las columnas de X).

    fitness_model: str (por defecto 'XGBRegressor')
        Modelo a utilizar en la evaluación de cada individuo. Opciones: 'XGBRegressor', 'ElasticNet', 'Ridge', 'SVR', 'Lasso', 'LinearRegression'.

    fitness_metric: str (por defecto 'r2')
        Métrica de evaluación del fitness. Opciones: 'rmse', 'mae', 'mse', 'r2'.

    scale_target: bool (por defecto False)
        Si es True, el objetivo y las predicciones se escalan con MinMaxScaler (solo ElasticNet).

    n_pop: int (por defecto 25)
        Número de individuos en cada generación.

    n_gen: int (por defecto 10)
        Número de generaciones.

    elite: int (por defecto 5)
        Número de individuos élite que pasan automáticamente a la siguiente generación.

    mut_prob: float (por defecto 0.25)
        Probabilidad de mutación por bit (feature) en cada individuo.

    random_state: int (por defecto 42)
        Semilla de aleatoriedad para reproducibilidad.

    max_active: int (por defecto 25)
        Número máximo de features activas por individuo.

    min_active: int (por defecto 18)
        Número mínimo de features activas por individuo.

    tournament_size: int (por defecto 3)
        Tamaño de torneo para la selección de padres (más grande = más presión selectiva).
    """

    def __init__(self, X_train, y_train, X_test, y_test, feature_names,
                 fitness_model="XGBRegressor", fitness_metric="r2", scale_target=False,
                 n_pop=25, n_gen=10, elite=5, mut_prob=0.3, random_state=42,
                 max_active=25, min_active=18, tournament_size=3):
        """
        Inicializamos la clase con todos los hiperparámetros necesarios para el algoritmo genético.
        """

        # Aquí asignamos la matriz de entrenamiento (X_train) y el vector objetivo (y_train).
        self.X_train = np.array(X_train)  # Matriz de variables predictoras para entrenamiento
        self.y_train = np.array(y_train)  # Vector objetivo para entrenamiento

        # Aquí asignamos la matriz de test (X_test) y el vector objetivo (y_test).
        self.X_test = np.array(X_test)    # Matriz de variables predictoras para test
        self.y_test = np.array(y_test)    # Vector objetivo para test

        # Aquí guardamos los nombres de las features, importantísimo para luego interpretarlas.
        self.feature_names = list(feature_names)   # Lista con los nombres de todas las features posibles
        self.n_features = len(self.feature_names)  # Número total de features

        # Elegimos el modelo para evaluar la aptitud (fitness) de cada individuo.
        self.fitness_model = fitness_model  # Modelo a usar en la evaluación (por defecto XGBRegressor)
        self.fitness_metric = fitness_metric.lower()  # Métrica de fitness ('rmse', 'mae', 'mse', 'r2')
        self.metric_index = {"rmse": 0, "mae": 1, "mse": 2, "r2": 3}[self.fitness_metric]
        self.scale_target = scale_target  # Si True, escalamos el target con MinMaxScaler (solo ElasticNet)

        # Definimos el tamaño de la población y el número de generaciones a simular.
        self.n_pop = n_pop            # Número de individuos en cada generación
        self.n_gen = n_gen            # Número de generaciones

        # Aquí configuramos cuántos individuos pasan automáticamente (elitismo) y la probabilidad de mutación.
        self.elite = elite            # Número de individuos élite que pasan automáticamente
        self.mut_prob = mut_prob      # Probabilidad de mutación por feature
        self.original_mut_prob = mut_prob  # Guardamos el valor original para resetearlo si hace falta

        # Aquí se fija la semilla para reproducibilidad de resultados.
        self.random_state = random_state   # Semilla de aleatoriedad

        # Aquí fijamos los límites mínimo y máximo de features activas permitidas por individuo.
        self.max_active = max_active   # Número máximo de features activas por individuo
        self.min_active = min_active   # Número mínimo de features activas por individuo
        
        # Definimos el tamaño del torneo para la selección de padres.
        self.tournament_size = tournament_size  # Tamaño de torneo (presión selectiva)

        # Fijamos la semilla de numpy y random para que todo sea reproducible.
        np.random.seed(random_state)
        random.seed(random_state)

        # Aquí creamos la población inicial aleatoria de individuos (cada uno con un subconjunto de features).
        self.population = self.init_population()
        self.metrics_history = []  # Historial de métricas (por generación)
        self.best_individual = None  # Mejor individuo encontrado hasta ahora
        self.best_score = -np.inf if self.fitness_metric == "r2" else np.inf
        self.no_improvement_counter = 0

        # Si elegimos 'MLP-Torch' como modelo, configuramos la red neuronal y la GPU (si existe).
        if self.fitness_model == "MLP-Torch":
            print(f"torch.cuda.is_available = {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print("Dispositivos CUDA disponibles:")
                print(torch.cuda.device_count())
                print(torch.cuda.current_device())
                print(torch.cuda.get_device_name())
            
            # Elegimos automáticamente GPU si hay disponible
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_mlp = None
            self.torch_input_dim = self.n_features
            
            # Definimos la arquitectura del MLP con PyTorch (capa oculta de 32, dropout, etc.)
            self.torch_mlp = nn.Sequential(
                nn.Linear(self.torch_input_dim, 32),
                nn.Sigmoid(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ).to(self.torch_device)

    def init_population(self):
        """
        Aquí estamos inicializando la población, asegurándonos de que haya variedad
        en el número de features activas por individuo.
        """
        posibles_activas = np.linspace(self.min_active, self.max_active, self.n_pop, dtype=int)
        np.random.shuffle(posibles_activas)
        pop = []
        for n_active in posibles_activas:
            ind = np.zeros(self.n_features, dtype=int)
            selected = np.random.choice(self.n_features, n_active, replace=False)
            ind[selected] = 1
            pop.append(ind)
        return pop

    def enforce_active_limits(self, ind):
        """
        Aquí nos aseguramos de que el individuo tenga un número de features activas dentro de los límites.
        Si se pasa de activas, apagamos algunas. Si le faltan, encendemos otras al azar.
        """
        n_active = np.sum(ind)
        # Si el número de features activas está fuera de los límites, lo ajustamos.
        if n_active > self.max_active:
            idx_active = np.where(ind == 1)[0]
            idx_to_turn_off = np.random.choice(idx_active, n_active - self.max_active, replace=False)
            ind[idx_to_turn_off] = 0
        elif n_active < self.min_active:
            idx_inactive = np.where(ind == 0)[0]
            idx_to_turn_on = np.random.choice(idx_inactive, self.min_active - n_active, replace=False)
            ind[idx_to_turn_on] = 1
        return ind

    def calcular_metricas(self, y_true, y_pred):
        """
        Aquí calculamos todas las métricas clásicas de regresión para comparar modelos.
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return rmse, mae, mse, r2

    def randomize_torch_weights(self):
        """
        Aquí simplemente reinicializamos los pesos del MLP de PyTorch,
        para evitar que arrastre información de generaciones previas.
        """
        if self.torch_mlp is not None:
            for layer in self.torch_mlp:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def fitness(self, individual, return_metrics=False):
        """
        Aquí estamos evaluando un individuo: probamos su subconjunto de features
        en el modelo correspondiente y devolvemos la métrica principal.

        El fitness mide la calidad del subconjunto de variables elegido por este individuo,
        usando el modelo y métrica definidos (ElasticNet, Ridge, XGBRegressor, MLP...).
        """
        # Si el individuo no tiene ninguna feature activa, devolvemos la peor métrica posible.
        # Esto penaliza soluciones que "no seleccionan nada" y fuerza a usar algún input.
        if np.sum(individual) == 0:
            resultado = (np.inf, np.inf, np.inf, -np.inf)
            return resultado if return_metrics else resultado[self.metric_index]

        # Elegimos solo las columnas (features) activas según el cromosoma del individuo.
        # idx es un array (máscara) de índices donde el individuo tiene 1 (features activas).
        # Esto nos permite seleccionar dinámicamente las columnas de X_train y X_test.
        idx = np.where(individual == 1)[0]
        X = self.X_train[:, idx]
        X_test = self.X_test[:, idx]
        y = self.y_train
        y_test = self.y_test

        try:
            # Entrenamos y predecimos con el modelo solicitado. Cada uno se maneja diferente:
            if self.fitness_model == "ElasticNet":
                
                # Si queremos escalar el target (solo útil en algunos casos, sobre todo para ElasticNet)
                if self.scale_target:
                    scaler = MinMaxScaler()
                    # Aquí añadimos una dimensión a y/y_test, escalamos y luego devolvemos a vector 1D.
                    y_train_s = scaler.fit_transform(y[:, None]).flatten()
                    y_test_s = scaler.transform(y_test[:, None]).flatten()
                else:
                    y_train_s = y
                    y_test_s = y_test
                
                # Entrenamos el modelo ElasticNet con parámetros fijos (se puede tunear)
                model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000, random_state=self.random_state)
                model.fit(X, y_train_s)
                
                y_pred = model.predict(X_test)
                
                # Si escalamos, deshacemos el escalado para poder comparar contra el y original.
                if self.scale_target:
                    y_pred = scaler.inverse_transform(y_pred[:, None]).flatten()
                    y_test = scaler.inverse_transform(y_test_s[:, None]).flatten()

            elif self.fitness_model == "Ridge":
                # Modelo Ridge: similar, pero sin escalado del target por defecto.
                model = Ridge(alpha=1.0, random_state=self.random_state)
                model.fit(X, y)
                y_pred = model.predict(X_test)
            
            elif self.fitness_model == "SVR":
                # Modelo SVR: útil para relaciones no lineales. 
                model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                model.fit(X, y)
                y_pred = model.predict(X_test)

            elif self.fitness_model == "Lasso":
                # Lasso, evita overfitting si tenemos muchas features ruidosas.
                model = Lasso(alpha=0.1)
                model.fit(X, y)
                y_pred = model.predict(X_test)
            
            elif self.fitness_model == "LinearRegression":
                # Regresión lineal simple, para comparar con otros modelos.
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X_test)

            elif self.fitness_model == "XGBRegressor":
                # Modelo XGBoost: potente para relaciones no lineales y selección de features.
                model = XGBRegressor(n_estimators=60, max_depth=4, n_jobs=32,
                                     random_state=self.random_state, verbosity=0)
                model.fit(X, y)
                y_pred = model.predict(X_test)

            elif self.fitness_model == "MLP":
                # MLP de sklearn: red neuronal simple para regresión.
                model = MLPRegressor(
                    hidden_layer_sizes=(16,),
                    activation='relu',
                    solver='adam',
                    max_iter=250,
                    early_stopping=True,
                    n_iter_no_change=5,
                    tol=1e-3,
                    random_state=self.random_state,
                    verbose=False
                )
                model.fit(X, y)
                y_pred = model.predict(X_test)

            elif self.fitness_model == "MLP-Torch":
                # Aquí usamos una red neuronal propia hecha en PyTorch, compatible con GPU si hay.
                torch.manual_seed(self.random_state)
                np.random.seed(self.random_state)

                # Si el número de features cambia, reconstruimos la arquitectura (dinámica).
                if (self.torch_mlp is None) or (X.shape[1] != self.torch_input_dim):
                    self.torch_input_dim = X.shape[1]
                    self.torch_mlp = nn.Sequential(
                        nn.Linear(self.torch_input_dim, 32),
                        nn.Sigmoid(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 1),
                        nn.Sigmoid()
                    ).to(self.torch_device)

                # Aquí reseteamos los pesos para que no arrastre información de generaciones anteriores.
                self.randomize_torch_weights()

                # Convertimos los datos a tensores y los subimos al dispositivo correcto.
                X_torch = torch.tensor(X, dtype=torch.float32).to(self.torch_device)
                y_torch = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.torch_device)
                X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(self.torch_device)

                # Optimizador Adam y función de pérdida MSE.
                optimizer = torch.optim.Adam(self.torch_mlp.parameters(), lr=0.001, weight_decay=0.0003)
                loss_fn = torch.nn.MSELoss()

                # Early stopping manual, igual que en sklearn.
                best_loss = float('inf')
                epochs_no_improve = 10
                n_iter_no_change = 10
                best_model_state = None

                # Entrenamiento por épocas (máximo 500).
                for epoch in range(500):
                    self.torch_mlp.train()
                    
                    # PyTorch suma los gradientes por defecto! Lo limpiamos al inicio de cada época.
                    optimizer.zero_grad()
                    # Aquí hacemos la pasada hacia adelante y calculamos la pérdida.
                    output = self.torch_mlp(X_torch)
                    # Calculamos la pérdida y hacemos backpropagation.
                    loss = loss_fn(output, y_torch)
                    # Aquí hacemos el paso de optimización.
                    loss.backward()
                    # Actualizamos los pesos del modelo.
                    optimizer.step()

                    # Aquí guardamos la pérdida actual para early stopping.
                    curr_loss = loss.item()
                    
                    # Guardamos el mejor estado del modelo (early stopping)
                    # si la pérdida mejora, actualizamos el mejor estado.
                    # Si no mejora, contamos cuántas épocas llevamos sin mejora.
                    if curr_loss < best_loss - 1e-4:
                        best_loss = curr_loss
                        epochs_no_improve = 0
                        best_model_state = self.torch_mlp.state_dict()
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= n_iter_no_change:
                            break

                # Recuperamos el mejor estado (si lo hay)
                if best_model_state is not None:
                    self.torch_mlp.load_state_dict(best_model_state)

                # Predicción final sobre el set de test.
                self.torch_mlp.eval()
                
                # Aquí hacemos la predicción y la pasamos a CPU para convertirla a numpy.
                with torch.no_grad():
                    y_pred = self.torch_mlp(X_test_torch).cpu().numpy().flatten()

            else:
                # Si alguien mete un modelo no soportado, lanzamos error explícito.
                raise ValueError(f"Modelo '{self.fitness_model}' no soportado")

            # Calculamos todas las métricas de rendimiento (RMSE, MAE, MSE, R2).
            resultado = self.calcular_metricas(y_test, y_pred)
            # Según lo que nos pidan, devolvemos solo la métrica principal o el paquete entero de métricas.
            return resultado if return_metrics else resultado[self.metric_index]

        except Exception as e:
            print(f"Error evaluando individuo: {e}")
            # Si algo falla (error numérico, shape, lo que sea), penalizamos el individuo con el peor fitness posible.
            resultado = (np.inf, np.inf, np.inf, -np.inf)
            return resultado if return_metrics else resultado[self.metric_index]

    def select(self, scores):
        """
        Aquí elegimos los individuos que pasan a la próxima generación:
        Primero, guardamos los élite y luego rellenamos el resto con torneos.
        """
        if self.fitness_metric == "r2":
            indices_ordenados = np.argsort(scores)[::-1]  # De mayor a menor
        else:
            indices_ordenados = np.argsort(scores)  # De menor a mayor

        elite = [self.population[i] for i in indices_ordenados[:self.elite]]
        rest = []
        
        # Aquí jugamos torneos para llenar el resto de la población
        while len(rest) < self.n_pop - self.elite:
            candidates = np.random.choice(self.n_pop, self.tournament_size, replace=False)
            if self.fitness_metric == "r2":
                best_idx = candidates[np.argmax(scores[candidates])]
            else:
                best_idx = candidates[np.argmin(scores[candidates])]
            rest.append(self.population[best_idx].copy())
        return elite + rest

    def crossover(self, parent1, parent2):
        """
        Aquí estamos mezclando los genes de dos padres, eligiendo aleatoriamente
        cada feature de uno u otro. Así sale el hijo.
        """
        # Máscara binaria aleatoria para mezclar genes de los padres (crossover)
        mask = np.random.randint(0, 2, self.n_features)
        child = (parent1 & mask) | (parent2 & ~mask)
        return self.enforce_active_limits(child)

    def mutate(self, individual):
        """
        Aquí revisamos cada feature y, con cierta probabilidad, la cambiamos de estado.
        Es la chispa de variabilidad genética.
        """
        mutated = individual.copy()
        # Aquí recorremos cada feature y aplicamos la mutación si toca
        for i in range(self.n_features):
            if random.random() < self.mut_prob:
                # Convertimos el bit: si estaba activo, lo apagamos; si no, lo encendemos.
                if mutated[i] == 1:
                    mutated[i] = 0
                else:
                    mutated[i] = 1
        # Nos aseguramos de que el individuo sigue cumpliendo los límites de features activas.
        # Si no, lo ajustamos para que tenga el número correcto de features activas.
        return self.enforce_active_limits(mutated)

    def fit(self, verbose=True):
        """
        Aquí ejecutamos todo el ciclo evolutivo del algoritmo genético, generación a generación.

        En cada generación, evaluamos a toda la población, seleccionamos los mejores individuos,
        aplicamos cruces y mutaciones, y adaptamos la población para intentar mejorar el fitness.
        """
        print(f"Filas de train: {self.X_train.shape[0]}, Filas de test: {self.X_test.shape[0]}, Features a seleccionar: {self.max_active}\n")

        # Bucle principal: una vuelta por cada generación evolutiva.
        for gen in range(self.n_gen):
            tiempo_inicio = perf_counter()  # Tomamos el tiempo inicial (para saber cuánto tarda cada gen).

            # 1. Evaluación de la población: calculamos fitness de todos los individuos (en paralelo podría ir más rápido).
            metrics_all = []
            for ind in self.population:
                # Aquí almacenamos las 4 métricas para cada individuo (rmse, mae, mse, r2)
                metrics_all.append(self.fitness(ind, True))
            metrics_all = np.array(metrics_all)
            # Extraemos solo la métrica principal para selección (puede ser rmse o r2, según el objetivo).
            scores = metrics_all[:, self.metric_index]

            # 2. Identificamos el mejor individuo de la generación (según métrica)
            if self.fitness_metric == "r2":
                best_idx = np.argmax(scores)   # Buscamos el mayor r2
            elif self.fitness_metric == "rmse":
                best_idx = np.argmin(scores)   # Buscamos el menor rmse
            else:
                # Si hay una métrica no soportada, lo paramos aquí.
                raise ValueError(f"Métrica '{self.fitness_metric}' no soportada")

            best_metrics = metrics_all[best_idx]  # Guardamos todas las métricas del mejor
            n_active = np.sum(self.population[best_idx])  # Contamos cuántas features tiene activas el mejor

            # 3. ¿Hemos mejorado el mejor global? Si sí, lo actualizamos.
            improved = (
                (self.fitness_metric == "r2" and best_metrics[self.metric_index] > self.best_score) or
                (self.fitness_metric != "r2" and best_metrics[self.metric_index] < self.best_score)
            )
            if improved:
                self.best_score = best_metrics[self.metric_index]
                self.best_individual = self.population[best_idx].copy()
                self.no_improvement_counter = 0
                self.mut_prob = self.original_mut_prob  # Volvemos a la mutación base
            else:
                # Si no mejora, subimos el contador de estancamiento.
                self.no_improvement_counter += 1

            # 4. Mostramos información detallada de la generación si se solicita (útil para debug)
            if verbose:
                tiempo = perf_counter() - tiempo_inicio
                print(f"Gen {gen+1}/{self.n_gen} | Score: {scores[best_idx]:.4f} | R2: {best_metrics[3]:.4f} | "
                      f"RMSE: {best_metrics[0]:.4f} | MAE: {best_metrics[1]:.4f} | MSE: {best_metrics[2]:.4f} | "
                      f"Tiempo: {tiempo:.2f}s | Features activas: {n_active}")

            # Guardamos el histórico de las mejores métricas por generación (para análisis posterior o gráficas)
            self.metrics_history.append(best_metrics)

            # 5. Estrategia de adaptación automática: si llevamos varias generaciones atascados,
            # aumentamos la mutación o forzamos diversidad.
            if self.no_improvement_counter == 3:
                self.mut_prob *= 1.5  # Mutamos más fuerte para intentar salir del mínimo local.
            if self.no_improvement_counter == 5:
                # Si ni aun así mejora, reemplazamos a los 3 peores por individuos completamente nuevos.
                self.population[-3:] = self.init_population()[:3]
                self.mut_prob = self.original_mut_prob

            # 6. Selección de la nueva población:
            # a) Pasan los élite directamente.
            # b) El resto se genera cruzando y mutando padres seleccionados por torneo.
            selected = self.select(scores)
            next_pop = selected[:self.elite]
            
            while len(next_pop) < self.n_pop:
                # Elegimos dos padres aleatorios entre los seleccionados y generamos un hijo por crossover + mutación.
                p1, p2 = random.sample(selected, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_pop.append(child)
            
            # Actualizamos la población para la próxima generación.
            self.population = next_pop

        
        # Preparamos los datos para guardar como CSV
        resultados = []
        for gen, metrics in enumerate(self.metrics_history, 1):
            resultados.append({
                "Generacion": gen,
                "Score": metrics[self.metric_index],
                "R2": metrics[3],
                "RMSE": metrics[0],
                "MAE": metrics[1],
                "MSE": metrics[2],
                "Features_activas": np.sum(self.population[0]),  # Mejor individuo de la última generación (puedes ajustar si quieres guardar todos)
            })

        # Lo convertimos a DataFrame y volvamos a CSV
        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv("Resultados/Resultado_Ejecucion_GA_Feature_Selection.csv", index=False)


    def summary(self):
        """
        Aquí simplemente mostramos las mejores features seleccionadas por el algoritmo.
        """
        idx = np.where(self.best_individual == 1)[0]
        selected = [self.feature_names[i] for i in idx]
        print(f"Mejores features seleccionadas ({len(selected)}): {selected}")

    def get_best_features(self):
        """
        Aquí devolvemos la lista de features activas (seleccionadas) del mejor individuo.
        """
        idx = np.where(self.best_individual == 1)[0]
        return [self.feature_names[i] for i in idx]
