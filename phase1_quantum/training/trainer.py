"""
TensorFlow & PyGAD Trainer for Quantum CNN.
Uses a Genetic Algorithm to evolve the HybridQCNN weights instead of standard Backpropagation.
"""

import pygad
import pygad.kerasga
import tensorflow as tf
import numpy as np
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class GATrainer:
    def __init__(self, model, cfg, run_tag, output_dir):
        self.model = model
        self.cfg = cfg
        self.run_tag = run_tag
        self.output_dir = Path(output_dir)
        self.history = {"val_acc": [], "best_fitness": []}
        
        # Use a small subset of data to rapidly compute fitness per generation.
        # Computing on 800k samples * 15 population takes way too long per generation.
        self.fitness_batch_size = cfg.get("ga_fitness_batch", 5000)
        self.best_model_path = self.output_dir / "checkpoints" / self.run_tag / "final_model_weights.weights.h5"
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.keras_ga = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.best_val_acc = 0.0

    def fit(self, train_ds, val_ds):
        """
        train_ds and val_ds are tf.data.Dataset objects.
        """
        logger.info(f"Extracting fitness subset ({self.fitness_batch_size} samples)...")
        tx_list, ty_list = [], []
        samples = 0
        for x, y in train_ds:
            tx_list.append(x)
            ty_list.append(y)
            samples += x.shape[0]
            if samples >= self.fitness_batch_size:
                break
                
        self.train_x = tf.concat(tx_list, axis=0)[:self.fitness_batch_size]
        self.train_y = tf.concat(ty_list, axis=0)[:self.fitness_batch_size]
        
        logger.info("Extracting validation subset...")
        vx_list, vy_list = [], []
        # limit validation subset to 10k to keep it fast
        val_samples = 0
        for x, y in val_ds:
            vx_list.append(x)
            vy_list.append(y)
            val_samples += x.shape[0]
            if val_samples > 10000:
                break
                
        self.val_x = tf.concat(vx_list, axis=0)
        self.val_y = tf.concat(vy_list, axis=0)

        # 1. Initialize PyGAD model weights
        # We must call the model once to build its weights
        _ = self.model(self.train_x[:1])
        num_solutions = self.cfg.get("ga_population_size", 15)
        self.keras_ga = pygad.kerasga.KerasGA(model=self.model, num_solutions=num_solutions)

        # 2. Fitness Function
        def fitness_func(ga_instance, solution, sol_idx):
            model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
                model=self.model, weights_vector=solution
            )
            self.model.set_weights(weights=model_weights_matrix)
            
            logits = self.model(self.train_x, training=False)
            preds = tf.cast(logits > 0, tf.float32)
            
            correct = tf.reduce_sum(tf.cast(tf.equal(preds, tf.expand_dims(tf.cast(self.train_y, tf.float32), 1)), tf.float32))
            accuracy = float(correct / self.train_x.shape[0])
            return accuracy

        # 3. Generation Callback
        def on_generation(ga_instance):
            solution, solution_fitness, _ = ga_instance.best_solution()
            model_weights_matrix = pygad.kerasga.model_weights_as_matrix(
                model=self.model, weights_vector=solution
            )
            self.model.set_weights(weights=model_weights_matrix)
            
            val_preds = []
            batch_sz = 1024
            for i in range(0, self.val_x.shape[0], batch_sz):
                logits = self.model(self.val_x[i:i+batch_sz], training=False)
                val_preds.append(tf.cast(logits > 0, tf.float32))
                
            val_preds = tf.concat(val_preds, axis=0)
            correct = tf.reduce_sum(tf.cast(tf.equal(val_preds, tf.expand_dims(tf.cast(self.val_y, tf.float32), 1)), tf.float32))
            val_acc = float(correct / self.val_x.shape[0])
            
            self.history["val_acc"].append(val_acc)
            self.history["best_fitness"].append(solution_fitness)
            
            logger.info(f"[Generation {ga_instance.generations_completed}] Fitness (Train Acc): {solution_fitness:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.model.save_weights(self.best_model_path)
                logger.info(f"   => New best model saved! (Val Acc: {val_acc:.4f})")

        num_generations = self.cfg.get("ga_generations", 20)
        
        ga_instance = pygad.GA(
            num_generations=num_generations,
            num_parents_mating=max(2, num_solutions // 3),
            initial_population=self.keras_ga.population_weights,
            fitness_func=fitness_func,
            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
            on_generation=on_generation,
            suppress_warnings=True
        )

        logger.info(f"Starting Genetic Algorithm Training for {num_generations} generations (Population: {num_solutions})...")
        ga_instance.run()
        
        if self.best_model_path.exists():
            self.model.load_weights(self.best_model_path)
            
        return self.best_model_path, self.history
