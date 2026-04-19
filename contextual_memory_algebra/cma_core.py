"""
Контекстная алгебра памяти (CMA) - Core Implementation
========================================================

Принципиально новая математическая система, где результат операции
зависит от истории предшествующих вычислений.

Автор: Независимый исследователь
Дата: 2026
"""

from __future__ import annotations
from typing import Callable, List, Any, Optional, Dict, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
import random


# ============================================================================
# Базовые классы
# ============================================================================

@dataclass
class Context:
    """
    Контекст памяти - хранит историю операций и глубину памяти.
    
    Атрибуты:
        history: Список предшествующих операций (каждая операция - tuple)
        k: Глубина памяти (максимальное количество учитываемых прошлых операций)
    """
    history: List[Tuple[Any, Any, Any]] = field(default_factory=list)
    k: int = 3
    
    def add_operation(self, left: Any, right: Any, result: Any) -> None:
        """Добавляет операцию в историю."""
        self.history.append((left, right, result))
        # Усечение до глубины памяти
        if len(self.history) > self.k:
            self.history = self.history[-self.k:]
    
    def get_active_history(self) -> List[Tuple[Any, Any, Any]]:
        """Возвращает активную (учитываемую) часть истории."""
        return self.history[-self.k:] if self.k > 0 else []
    
    def clear(self) -> None:
        """Очищает историю."""
        self.history.clear()
    
    def __repr__(self) -> str:
        return f"Context(history_len={len(self.history)}, k={self.k})"


class WeightFunction(ABC):
    """Абстрактный базовый класс для функций влияния."""
    
    @abstractmethod
    def get_weight(self, position: int, total_length: int) -> float:
        """Возвращает вес для позиции в истории."""
        pass


class ExponentialDecay(WeightFunction):
    """Экспоненциально затухающие веса: w(i) = α^(n-i), где i - позиция от конца."""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
    
    def get_weight(self, position: int, total_length: int) -> float:
        # position: 0 - самая последняя операция
        return self.alpha ** position


class LinearDecay(WeightFunction):
    """Линейно затухающие веса."""
    
    def __init__(self, slope: float = 0.1):
        self.slope = slope
    
    def get_weight(self, position: int, total_length: int) -> float:
        return max(0, 1 - self.slope * position)


class InversePosition(WeightFunction):
    """Веса обратно пропорциональные позиции: w(i) = 1/(i+1)."""
    
    def get_weight(self, position: int, total_length: int) -> float:
        return 1.0 / (position + 1)


class UniformWeight(WeightFunction):
    """Равномерные веса."""
    
    def get_weight(self, position: int, total_length: int) -> float:
        if total_length == 0:
            return 0
        return 1.0 / total_length


# ============================================================================
# Основной класс CMA
# ============================================================================

class ContextualMemoryAlgebra:
    """
    Контекстная алгебра памяти (CMA).
    
    Основная идея: результат бинарной операции зависит от истории
    предшествующих операций.
    
    Атрибуты:
        base_set: Основное множество элементов
        base_operation: Базовая операция (без контекста)
        weight_function: Функция влияния истории
        context: Текущий контекст памяти
        neutral_element: Нейтральный элемент
    """
    
    def __init__(
        self,
        base_set: set,
        base_operation: Callable[[Any, Any], Any],
        weight_function: Optional[WeightFunction] = None,
        k: int = 3,
        neutral_element: Any = None
    ):
        self.base_set = base_set
        self.base_operation = base_operation
        self.weight_function = weight_function or ExponentialDecay(0.5)
        self.context = Context(k=k)
        
        # Автоматическое определение нейтрального элемента
        if neutral_element is None:
            self.neutral_element = self._find_neutral_element()
        else:
            self.neutral_element = neutral_element
    
    def _find_neutral_element(self) -> Any:
        """Поиск нейтрального элемента перебором (для демонстрации)."""
        # Пробуем типичные нейтральные элементы
        candidates = [0, 1, 0.0, 1.0, '', [], None]
        for candidate in candidates:
            if candidate in self.base_set:
                return candidate
        # Берем первый элемент множества
        return next(iter(self.base_set))
    
    def _compute_context_influence(
        self, 
        left: Any, 
        right: Any
    ) -> Any:
        """
        Вычисляет влияние контекста на результат операции.
        
        Это ключевой метод, определяющий уникальность CMA.
        """
        active_history = self.context.get_active_history()
        
        if not active_history:
            return 0
        
        total_length = len(active_history)
        influence = 0
        
        for i, (h_left, h_right, h_result) in enumerate(active_history):
            weight = self.weight_function.get_weight(i, total_length)
            
            # Вычисляем "сходство" текущей операции с историей
            similarity = self._compute_similarity(left, right, h_left, h_right)
            
            # Влияние пропорционально весу и сходству
            influence += weight * similarity * (h_result - self.neutral_element)
        
        return influence
    
    def _compute_similarity(
        self, 
        a1: Any, 
        b1: Any, 
        a2: Any, 
        b2: Any
    ) -> float:
        """Вычисляет степень сходства двух операций."""
        if isinstance(a1, (int, float)) and isinstance(a2, (int, float)):
            # Для чисел: нормализованная разность
            if isinstance(a1, int):
                max_val = max(abs(a1), abs(b1), abs(a2), abs(b2), 1)
            else:
                max_val = max(abs(a1), abs(b1), abs(a2), abs(b2), 1e-10)
            
            diff = (abs(a1 - a2) + abs(b1 - b2)) / (2 * max_val)
            return max(0, 1 - diff)
        
        # Для нечисловых типов: точное равенство
        return 1.0 if (a1 == a2 and b1 == b2) else 0.0
    
    def operate(self, left: Any, right: Any) -> Any:
        """
        Основная операция CMA: a ⊕_C b
        
        Результат зависит от текущего контекста памяти.
        
        Специальный случай: если левый операнд - нейтральный элемент,
        результат равен правому операнду (левая нейтральность: e ⊕ a = a).
        """
        # Проверяем, является ли левый операнд нейтральным элементом
        # По аксиоме 1: e ⊕ a = a (левая нейтральность)
        if left == self.neutral_element:
            self.context.add_operation(left, right, right)
            return right
        
        # Проверяем, является ли правый операнд нейтральным элементом  
        # По аксиоме 1: a ⊕ e = a (правая нейтральность)
        if right == self.neutral_element:
            self.context.add_operation(left, right, left)
            return left
        
        # Базовая операция
        base_result = self.base_operation(left, right)
        
        # Вычисляем влияние контекста
        context_influence = self._compute_context_influence(left, right)
        
        # Комбинируем результат
        if isinstance(base_result, (int, float)):
            result = base_result + context_influence
        else:
            # Для нечисловых типов: возвращаем базовый результат
            result = base_result
        
        # Обновляем контекст
        self.context.add_operation(left, right, result)
        
        return result
    
    def __add__(self, other: ContextualMemoryAlgebra) -> ContextualMemoryAlgebra:
        """Сложение двух CMA (объединение контекстов)."""
        new_cma = ContextualMemoryAlgebra(
            base_set=self.base_set | other.base_set,
            base_operation=lambda a, b: self.base_operation(a, b),
            weight_function=self.weight_function,
            k=max(self.context.k, other.context.k),
            neutral_element=self.neutral_element
        )
        # Объединяем истории
        new_cma.context.history = self.context.history + other.context.history
        return new_cma
    
    def reset_context(self) -> None:
        """Сбрасывает контекст памяти."""
        self.context.clear()
    
    def get_context_state(self) -> Dict[str, Any]:
        """Возвращает текущее состояние контекста."""
        return {
            'history_length': len(self.context.history),
            'k': self.context.k,
            'active_history': self.context.get_active_history(),
            'neutral_element': self.neutral_element
        }


# ============================================================================
# Специализированные CMA
# ============================================================================

class LinearCMA(ContextualMemoryAlgebra):
    """Линейная CMA с аддитивным влиянием контекста."""
    
    def __init__(self, base_set: set, alpha: float = 0.1, k: int = 3):
        super().__init__(
            base_set=base_set,
            base_operation=lambda a, b: a + b,
            weight_function=ExponentialDecay(alpha),
            k=k,
            neutral_element=0
        )


class MultiplicativeCMA(ContextualMemoryAlgebra):
    """Мультипликативная CMA с контекстным влиянием."""
    
    def __init__(self, base_set: set, beta: float = 0.05, k: int = 3):
        super().__init__(
            base_set=base_set,
            base_operation=lambda a, b: a * b,
            weight_function=ExponentialDecay(beta),
            k=k,
            neutral_element=1
        )


class NeuralCMA(ContextualMemoryAlgebra):
    """Нейронная CMA с нелинейным влиянием контекста."""
    
    def __init__(self, base_set: set, hidden_dim: int = 4, k: int = 3):
        self.hidden_dim = hidden_dim
        # Инициализация весов (упрощенная)
        self.W = [[random.uniform(-1, 1) for _ in range(hidden_dim)] 
                  for _ in range(hidden_dim)]
        
        super().__init__(
            base_set=base_set,
            base_operation=lambda a, b: a + b,
            weight_function=InversePosition(),
            k=k,
            neutral_element=0
        )
    
    def _compute_context_influence(self, left: Any, right: Any) -> Any:
        """Нейронное вычисление влияния контекста."""
        active_history = self.context.get_active_history()
        
        if not active_history:
            return 0
        
        # Создаем вектор контекста
        context_vec = []
        for _ in range(self.hidden_dim):
            # Упрощенное представление истории
            h_sum = sum(h[2] for h in active_history[-3:])
            context_vec.append(h_sum / max(len(active_history), 1))
        
        # Применяем "нейронный" слой
        result = 0
        for i, w_row in enumerate(self.W):
            result += sum(w * c for w, c in zip(w_row, context_vec))
        
        # Сигмоидная активация (упрощенная)
        result = 1 / (1 + math.exp(-result)) if abs(result) < 100 else (1 if result > 0 else 0)
        
        return result * 0.5


class EvolutionaryCMA(ContextualMemoryAlgebra):
    """CMA для эволюционных алгоритмов с памятью мутаций."""
    
    def __init__(self, base_set: set, mutation_rate: float = 0.1, k: int = 5):
        super().__init__(
            base_set=base_set,
            base_operation=lambda a, b: a + b,  # Комбинирование генов
            weight_function=LinearDecay(slope=mutation_rate),
            k=k,
            neutral_element=0
        )
        self.mutation_rate = mutation_rate
    
    def mutate(self, individual: Any) -> Any:
        """Применяет мутацию с учетом истории."""
        active_history = self.context.get_active_history()
        
        if not active_history:
            return individual + random.uniform(-1, 1) * self.mutation_rate
        
        # Адаптивная мутация на основе истории
        recent_fitness = [h[2] for h in active_history]
        avg_fitness = sum(recent_fitness) / len(recent_fitness)
        
        # Если история успешная - меньше мутаций
        adaptive_rate = self.mutation_rate * (1 - 0.5 * min(1, avg_fitness))
        
        return individual + random.uniform(-1, 1) * adaptive_rate


# ============================================================================
# Утилиты и вспомогательные функции
# ============================================================================

def create_cma_from_numbers(
    operation: str = 'add',
    k: int = 3,
    weight_type: str = 'exponential'
) -> ContextualMemoryAlgebra:
    """
    Фабрика для создания стандартных CMA для чисел.
    
    Аргументы:
        operation: 'add', 'multiply', 'subtract', 'divide'
        k: Глубина памяти
        weight_type: 'exponential', 'linear', 'inverse', 'uniform'
    
    Возвращает:
        Настроенную CMA
    """
    operations = {
        'add': lambda a, b: a + b,
        'multiply': lambda a, b: a * b,
        'subtract': lambda a, b: a - b,
        'divide': lambda a, b: a / b if b != 0 else 0
    }
    
    weight_functions = {
        'exponential': ExponentialDecay(0.5),
        'linear': LinearDecay(0.1),
        'inverse': InversePosition(),
        'uniform': UniformWeight()
    }
    
    neutral_elements = {
        'add': 0,
        'multiply': 1,
        'subtract': 0,
        'divide': 1
    }
    
    return ContextualMemoryAlgebra(
        base_set=set(range(-1000, 1000)) | {0.0},
        base_operation=operations.get(operation, operations['add']),
        weight_function=weight_functions.get(weight_type, weight_functions['exponential']),
        k=k,
        neutral_element=neutral_elements.get(operation, 0)
    )


def demonstrate_cma():
    """Демонстрация работы CMA."""
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ КОНТЕКСТНОЙ АЛГЕБРЫ ПАМЯТИ (CMA)")
    print("=" * 70)
    
    # Создаем CMA для сложения
    cma = create_cma_from_numbers('add', k=3, weight_type='exponential')
    
    print("\n1. БАЗОВАЯ ОПЕРАЦИЯ (без истории)")
    print("-" * 40)
    result = cma.operate(5, 3)
    print(f"   5 ⊕ 3 = {result}")
    print(f"   Контекст: {cma.get_context_state()}")
    
    print("\n2. ОПЕРАЦИЯ С КОНТЕКСТОМ")
    print("-" * 40)
    result = cma.operate(2, 2)
    print(f"   2 ⊕ 2 = {result}")
    print(f"   Контекст: {cma.get_context_state()}")
    
    print("\n3. ТРЕТЬЯ ОПЕРАЦИЯ (влияние истории)")
    print("-" * 40)
    result = cma.operate(1, 1)
    print(f"   1 ⊕ 1 = {result}")
    print(f"   Контекст: {cma.get_context_state()}")
    
    print("\n4. СБРОС КОНТЕКСТА")
    print("-" * 40)
    cma.reset_context()
    result = cma.operate(5, 3)
    print(f"   После сброса: 5 ⊕ 3 = {result}")
    print(f"   Контекст: {cma.get_context_state()}")
    
    print("\n5. СРАВНЕНИЕ С КЛАССИЧЕСКОЙ АЛГЕБРОЙ")
    print("-" * 40)
    cma_classic = create_cma_from_numbers('add', k=0)  # k=0 → без памяти
    cma_with_memory = create_cma_from_numbers('add', k=3)
    
    # Выполняем последовательность операций
    for _ in range(5):
        cma_classic.operate(2, 3)
        cma_with_memory.operate(2, 3)
    
    print(f"   Классическая (k=0): 2 ⊕ 3 = {cma_classic.operate(2, 3)}")
    print(f"   CMA с памятью (k=3): 2 ⊕ 3 = {cma_with_memory.operate(2, 3)}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demonstrate_cma()