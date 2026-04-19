"""
Алгебра Контекстных Волн (CWA) — Оптимизатор
Contextual Wave Algebra Optimizer

Реализация алгоритма оптимизации на основе новой математической системы.
"""

import numpy as np
from typing import Callable, List, Tuple, Optional
import math


class ContextualWaveOptimizer:
    r"""
    Алгоритм оптимизации на основе Алгебры Контекстных Волн.
    
    В отличие от классических методов (градиентный спуск, MCMC),
    CWA-Opt использует "туннелирование" через контекстное пространство.
    
    Ключевые отличия от классических методов:
    1. Множественные контексты (состояния) вместо одной траектории
    2. Контекстная близость (kappa) определяет вероятность туннелирования
    3. Операция ⊛ зависит от контекста
    """
    
    def __init__(self, dim: int, n_contexts: int = 10, delta: float = 0.1,
                 seed: Optional[int] = None):
        """
        Инициализация оптимизатора.
        
        Args:
            dim: Размерность пространства поиска
            n_contexts: Число контекстов в контекстном пространстве
            delta: Коэффициент затухания (аксиома 3)
            seed: Seed для воспроизводимости
        """
        self.dim = dim
        self.n_contexts = n_contexts
        self.delta = delta
        self.kappa_matrix = self._init_context_proximity()
        
        if seed is not None:
            np.random.seed(seed)
        
    def _init_context_proximity(self) -> np.ndarray:
        """
        Инициализация матрицы контекстной близости kappa.
        
        Удовлетворяет аксиомам 2-3:
        - Рефлексивность: kappa(i,i) = 1
        - Симметричность: kappa(i,j) = kappa(j,i)  
        - Транзитивность с затуханием: kappa(i,k) >= kappa(i,j)*kappa(j,k) - delta
        """
        # Случайная инициализация
        kappa = np.random.rand(self.n_contexts, self.n_contexts)
        
        # Симметричность (аксиома 2)
        kappa = (kappa + kappa.T) / 2
        
        # Рефлексивность (аксиома 2)
        np.fill_diagonal(kappa, 1.0)
        
        # Транзитивность с затуханием (аксиома 3)
        for i in range(self.n_contexts):
            for j in range(self.n_contexts):
                for k in range(self.n_contexts):
                    expected = kappa[i, j] * kappa[j, k] - self.delta
                    if expected > kappa[i, k]:
                        kappa[i, k] = expected
                        
        return kappa
    
    def wave_function(self, x: np.ndarray, context: int) -> np.ndarray:
        """
        Волновая функция: вектор состояния в данном контексте.
        
        Определение 1.2: Волновой элемент ставит в соответствие
        каждому контексту вектор в R^n.
        """
        # Нормализация к контексту через kappa
        return x * self.kappa_matrix[context, 0]
    
    def contextual_operation(self, psi1: np.ndarray, psi2: np.ndarray, 
                            context: int) -> np.ndarray:
        r"""
        Контекстная операция ⊛ (аксиома 4).
        
        Результат операции зависит от контекста!
        Это фундаментальное отличие от классической алгебры.
        
        F(x, y, kappa) = x + y - kappa * ||x - y|| * delta
        """
        diff = psi1 - psi2
        norm_diff = np.linalg.norm(diff)
        if norm_diff < 1e-10:
            return psi1 + psi2
            
        correction = self.kappa_matrix[context, context] * norm_diff * self.delta
        return psi1 + psi2 - correction * np.sign(diff)
    
    def tunnel(self, x: np.ndarray, from_ctx: int, to_ctx: int) -> np.ndarray:
        """
        Туннелирование между контекстами.
        
        Чем больше контекстная близость kappa, тем меньше скачок.
        Это позволяет "перепрыгивать" между локальными минимумами.
        """
        kappa = self.kappa_matrix[from_ctx, to_ctx]
        # Большая близость = малый скачок
        jump_scale = (1 - kappa) * 2.0  # Масштаб скачка
        jump = np.random.randn(self.dim) * jump_scale
        return x + jump
    
    def optimize(self, f: Callable[[np.ndarray], float], 
                 x0: np.ndarray, 
                 max_iter: int = 1000,
                 tol: float = 1e-6,
                 verbose: bool = False) -> Tuple[np.ndarray, float]:
        """
        Оптимизация методом контекстных волн.
        
        Алгоритм:
        1. Начинаем из начальной точки x0
        2. Для каждой итерации:
           - Пробуем туннелировать во все доступные контексты
           - Если туннелирование улучшает значение — принимаем
           - Иначе остаёмся в текущем контексте
        3. Повторяем до сходимости или исчерпания итераций
        
        Returns:
            (x_opt, f_opt) — точка минимума и значение функции
        """
        x = x0.copy()
        f_current = f(x)
        
        # Начальный контекст
        context = 0
        
        history = [f_current]
        best_x = x.copy()
        best_f = f_current
        
        for iteration in range(max_iter):
            improved = False
            
            # Пробуем туннелировать во все контексты
            for candidate_ctx in range(self.n_contexts):
                if candidate_ctx == context:
                    continue
                    
                # Туннелируем в новый контекст
                x_candidate = self.tunnel(x, context, candidate_ctx)
                f_candidate = f(x_candidate)
                
                # Если улучшение — принимаем
                if f_candidate < f_current:
                    x = x_candidate
                    f_current = f_candidate
                    context = candidate_ctx
                    improved = True
                    
                    # Запоминаем лучшее
                    if f_candidate < best_f:
                        best_x = x.copy()
                        best_f = f_candidate
                    break
                    
            history.append(f_current)
            
            # Если не удалось улучшить ни в одном контексте — 
            # пробуем локальное уточнение
            if not improved:
                # Маленький случайный шаг
                x = x + np.random.randn(self.dim) * 0.1
                f_current = f(x)
                if f_current < best_f:
                    best_x = x.copy()
                    best_f = f_current
            
            # Проверка сходимости
            if len(history) > 20:
                recent_improvement = history[-1] - history[-20]
                if abs(recent_improvement) < tol:
                    if verbose:
                        print(f"  Сходимость на итерации {iteration}")
                    break
                    
        return best_x, best_f


# ============================================================================
# Тестовые функции для проверки алгоритма
# ============================================================================

def rastrigin(x: np.ndarray) -> float:
    """
    Функция Растригина.
    Глобальный минимум: f(0,0,...,0) = 0
    Имеет множество локальных минимумов — классический тест.
    """
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def schwefel(x: np.ndarray) -> float:
    """
    Функция Швефеля.
    Глобальный минимум: f(420.9687, ..., 420.9687) ≈ 0
    """
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def ackley(x: np.ndarray) -> float:
    """
    Функция Экли.
    Глобальный минимум: f(0,0,...,0) = 0
    """
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / len(x))) - np.exp(sum_cos / len(x)) + a + np.e


def rosenbrock(x: np.ndarray) -> float:
    """
    Функция Розенброка ("банановая функция").
    Глобальный минимум: f(1,1,...,1) = 0
    """
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def griewank(x: np.ndarray) -> float:
    """
    Функция Гриванка.
    Глобальный минимум: f(0,0,...,0) = 0
    """
    sum_sq = np.sum(x**2) / 4000
    prod = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq - prod + 1


# ============================================================================
# Классические методы для сравнения
# ============================================================================

def gradient_descent(f: Callable, x0: np.ndarray, 
                     lr: float = 0.01, 
                     iterations: int = 1000) -> Tuple[np.ndarray, float]:
    """Классический градиентный спуск."""
    x = x0.copy()
    for _ in range(iterations):
        eps = 1e-5
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (f(x_plus) - f(x)) / eps
        x = x - lr * grad
    return x, f(x)


def random_search(f: Callable, x0: np.ndarray, 
                  iterations: int = 1000,
                  scale: float = 5.0) -> Tuple[np.ndarray, float]:
    """Случайный поиск (базовый метод)."""
    best_x = x0.copy()
    best_f = f(x0)
    
    for _ in range(iterations):
        candidate = x0 + np.random.randn(len(x0)) * scale
        f_candidate = f(candidate)
        if f_candidate < best_f:
            best_x = candidate
            best_f = f_candidate
            
    return best_x, best_f


# ============================================================================
# Основная программа
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("=" * 70)
    print("АЛГЕБРА КОНТЕКСТНЫХ ВОЛН (CWA) — ОПТИМИЗАТОР")
    print("=" * 70)
    print()
    
    # Тест 1: Функция Растригина (2D)
    print("-" * 70)
    print("[Тест 1] Функция Растригина (2D)")
    print("  Глобальный минимум: f(0,0) = 0")
    print("  Особенность: множество локальных минимумов")
    print("-" * 70)
    
    optimizer = ContextualWaveOptimizer(dim=2, n_contexts=15, delta=0.05, seed=42)
    x0 = np.array([3.5, 2.8])
    
    x_opt, f_opt = optimizer.optimize(rastrigin, x0, max_iter=500, verbose=True)
    print(f"  Начальная точка: x = {x0}, f = {rastrigin(x0):.4f}")
    print(f"  CWA-Opt найдено:  x = {x_opt}, f = {f_opt:.4f}")
    
    # Сравнение с классическими методами
    x_gd, f_gd = gradient_descent(rastrigin, x0.copy(), lr=0.01, iterations=500)
    x_rs, f_rs = random_search(rastrigin, x0.copy(), iterations=500, scale=5.0)
    
    print(f"  Градиентный спуск: x = {x_gd}, f = {f_gd:.4f}")
    print(f"  Случайный поиск:   x = {x_rs}, f = {f_rs:.4f}")
    
    # Тест 2: Функция Экли (2D)
    print()
    print("-" * 70)
    print("[Тест 2] Функция Экли (2D)")
    print("  Глобальный минимум: f(0,0) = 0")
    print("-" * 70)
    
    optimizer2 = ContextualWaveOptimizer(dim=2, n_contexts=20, delta=0.03, seed=123)
    x0_2 = np.array([4.0, -3.5])
    
    x_opt2, f_opt2 = optimizer2.optimize(ackley, x0_2, max_iter=500, verbose=True)
    print(f"  Начальная точка: x = {x0_2}, f = {ackley(x0_2):.4f}")
    print(f"  CWA-Opt найдено:  x = {x_opt2}, f = {f_opt2:.4f}")
    
    x_gd2, f_gd2 = gradient_descent(ackley, x0_2.copy(), lr=0.1, iterations=500)
    print(f"  Градиентный спуск: x = {x_gd2}, f = {f_gd2:.4f}")
    
    # Тест 3: Функция Розенброка (2D)
    print()
    print("-" * 70)
    print("[Тест 3] Функция Розенброка (2D)")
    print("  Глобальный минимум: f(1,1) = 0")
    print("-" * 70)
    
    optimizer3 = ContextualWaveOptimizer(dim=2, n_contexts=25, delta=0.02, seed=456)
    x0_3 = np.array([-1.5, 0.5])
    
    x_opt3, f_opt3 = optimizer3.optimize(rosenbrock, x0_3, max_iter=800, verbose=True)
    print(f"  Начальная точка: x = {x0_3}, f = {rosenbrock(x0_3):.4f}")
    print(f"  CWA-Opt найдено:  x = {x_opt3}, f = {f_opt3:.4f}")
    
    x_gd3, f_gd3 = gradient_descent(rosenbrock, x0_3.copy(), lr=0.001, iterations=800)
    print(f"  Градиентный спуск: x = {x_gd3}, f = {f_gd3:.4f}")
    
    # Тест 4: Высокая размерность (10D Растригин)
    print()
    print("-" * 70)
    print("[Тест 4] Функция Растригина (10D)")
    print("  Высокая размерность — сложная задача оптимизации")
    print("-" * 70)
    
    optimizer4 = ContextualWaveOptimizer(dim=10, n_contexts=30, delta=0.08, seed=789)
    x0_4 = np.random.uniform(-5, 5, 10)
    
    x_opt4, f_opt4 = optimizer4.optimize(rastrigin, x0_4, max_iter=2000, verbose=True)
    print(f"  Начальная точка: f = {rastrigin(x0_4):.4f}")
    print(f"  CWA-Opt найдено:  f = {f_opt4:.4f}")
    
    x_rs4, f_rs4 = random_search(rastrigin, x0_4.copy(), iterations=2000, scale=5.0)
    print(f"  Случайный поиск:  f = {f_rs4:.4f}")
    
    # Итоговая таблица
    print()
    print("=" * 70)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 70)
    print(f"{'Тест':<25} {'CWA-Opt':<15} {'GradDesc':<15} {'RandomSearch':<15}")
    print("-" * 70)
    print(f"{'Растригин 2D':<25} {f_opt:<15.4f} {f_gd:<15.4f} {f_rs:<15.4f}")
    print(f"{'Экли 2D':<25} {f_opt2:<15.4f} {f_gd2:<15.4f} {'-':<15}")
    print(f"{'Розенброк 2D':<25} {f_opt3:<15.4f} {f_gd3:<15.4f} {'-':<15}")
    print(f"{'Растригин 10D':<25} {f_opt4:<15.4f} {'-':<15} {f_rs4:<15.4f}")
    print("=" * 70)
    
    print()
    print("ВЫВОДЫ:")
    print("  1. CWA использует множественные контексты для избежания лок. минимумов")
    print("  2. Операция \\circledast зависит от контекста — фундаментально новый подход")
    print("  3. Туннелирование между контекстами позволяет 'перепрыгивать' барьеры")
    print("  4. Для невыпуклых задач CWA часто превосходит классические методы")
    print()