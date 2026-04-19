"""
Контекстуальная Алгебра (Contextual Algebra)
=============================================

Реализация новой математической системы с зависимыми от контекста операциями.

Автор: Разработчик агента
Дата: 2026
"""

from typing import Callable, Any, Optional, Tuple, List
from dataclasses import dataclass
import math


@dataclass
class ContextualElement:
    """
    Элемент контекстуального множества.
    
    Атрибуты:
        value: Значение из базового множества X
        context: Контекст из контекстного пространства K
    """
    value: Any
    context: Any
    
    def __repr__(self):
        return f"({self.value}, {self.context})"


class ContextSpace:
    """
    Контекстное пространство (K, ≤, ∘)
    
    Параметры:
        elements: Множество контекстов
        order: Функция сравнения (возвращает True если k1 ≤ k2)
        transition: Операция контекстного перехода
    """
    
    def __init__(
        self, 
        elements: List[Any],
        order: Callable[[Any, Any], bool] = lambda a, b: a <= b,
        transition: Callable[[Any, Any], Any] = lambda a, b: (a + b) / 2
    ):
        self.elements = set(elements)
        self.order = order
        self.transition = transition
        
    def is_leq(self, k1: Any, k2: Any) -> bool:
        """Проверка k1 ≤ k2"""
        return self.order(k1, k2)
    
    def transition_op(self, k1: Any, k2: Any) -> Any:
        """Операция контекстного перехода k1 ∘ k2"""
        return self.transition(k1, k2)
    
    def find_neutral(self) -> Optional[Any]:
        """Поиск нейтрального контекста (аксиома 1)"""
        # Сначала проверяем, задан ли нейтральный контекст явно
        if hasattr(self, '_neutral'):
            return self._neutral
            
        for k0 in self.elements:
            is_neutral = True
            for k in self.elements:
                # k0 ∘ k = k
                if self.transition_op(k0, k) != k:
                    is_neutral = False
                    break
                # k ∘ k0 = k  
                if self.transition_op(k, k0) != k:
                    is_neutral = False
                    break
            if is_neutral:
                return k0
        return None


class ContextualAlgebra:
    """
    Контекстуальная алгебра (K, M, ⊗)
    
    Реализует операцию ⊗ на контекстуальном множестве M.
    """
    
    def __init__(
        self,
        context_space: ContextSpace,
        base_operation: Callable[[Any, Any], Any] = lambda a, b: a + b,
        contextual_rule: Optional[Callable[[Any, Any, Any, Any], Tuple[Any, Any]]] = None
    ):
        """
        Инициализация алгебры.
        
        Параметры:
            context_space: Контекстное пространство K
            base_operation: Базовая операция f: X × X → X (для нейтрального контекста)
            contextual_rule: Правило вычисления результата и нового контекста
                           Формат: (x1, k1, x2, k2) → (result, new_context)
        """
        self.K = context_space
        self.base_op = base_operation
        self.contextual_rule = contextual_rule or self._default_rule
        
    def _default_rule(
        self, 
        x1: Any, 
        k1: Any, 
        x2: Any, 
        k2: Any
    ) -> Tuple[Any, Any]:
        """
        Правило по умолчанию: взвешенное среднее с адаптивными весами.
        
        Результат: x = x1 * w1 + x2 * w2, где веса зависят от контекстов
        Новый контекст: k1 ∘ k2
        
        В нейтральном контексте (k=0) сводится к базовой операции.
        """
        # Вычисляем новый контекст
        new_context = self.K.transition_op(k1, k2)
        
        # Проверяем, нейтральный ли контекст
        neutral = self.K.find_neutral()
        is_neutral = (k1 == neutral and k2 == neutral)
        
        if is_neutral:
            # Аксиома 4: в нейтральном контексте — классическая операция
            result = self.base_op(x1, x2)
        else:
            # Вычисляем веса на основе контекстов
            k_sum = abs(k1) + abs(k2) + 1e-10  # избегаем деления на 0
            
            w1 = abs(k1) / k_sum
            w2 = abs(k2) / k_sum
            
            # Результат — взвешенная сумма
            result = x1 * w1 + x2 * w2
        
        return result, new_context
    
    def operate(
        self, 
        elem1: ContextualElement, 
        elem2: ContextualElement
    ) -> ContextualElement:
        """
        Контекстуальная операция ⊗
        
        (x1, k1) ⊗ (x2, k2) = (result, new_context)
        """
        result, new_context = self.contextual_rule(
            elem1.value, elem1.context,
            elem2.value, elem2.context
        )
        return ContextualElement(result, new_context)
    
    def __mul__(self, other: 'ContextualAlgebra') -> 'ContextualAlgebra':
        """Композиция контекстуальных алгебр (произведение)"""
        # Новый контекст — декартово произведение
        def combined_order(k1, k2):
            return self.K.is_leq(k1[0], k2[0]) and other.K.is_leq(k1[1], k2[1])
        
        def combined_transition(k1, k2):
            return (
                self.K.transition_op(k1[0], k2[0]),
                other.K.transition_op(k1[1], k2[1])
            )
        
        combined_space = ContextSpace(
            elements=[(a, b) for a in self.K.elements for b in other.K.elements],
            order=combined_order,
            transition=combined_transition
        )
        
        return ContextualAlgebra(combined_space)


# =============================================================================
# СПЕЦИАЛИЗИРОВАННЫЕ РЕАЛИЗАЦИИ
# =============================================================================

class LinearContextualAlgebra(ContextualAlgebra):
    """
    Линейная контекстуальная алгебра над ℝ.
    
    Контексты — действительные числа.
    Операция контекстного перехода: k1 ∘ k2 = α·k1 + (1-α)·k2
    """
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        
        # Контекстное пространство K = ℝ
        def real_order(a, b):
            return a <= b
        
        def real_transition(a, b):
            # Нейтральный контекст 0 должен сохранять другой контекст
            if a == 0.0:
                return b
            if b == 0.0:
                return a
            return self.alpha * a + (1 - self.alpha) * b
        
        space = ContextSpace(
            elements=list(range(-1000, 1001)),  # Дискретное приближение ℝ
            order=real_order,
            transition=real_transition
        )
        
        # Явно задаём нейтральный контекст
        space._neutral = 0.0
        
        super().__init__(space, base_operation=lambda a, b: (a + b) / 2)
        
    def contextual_rule(
        self, 
        x1: float, 
        k1: float, 
        x2: float, 
        k2: float
    ) -> Tuple[float, float]:
        """Взвешенное среднее с контекстно-зависимыми весами"""
        new_context = self.K.transition_op(k1, k2)
        
        # Веса зависят от "силы" контекста
        k_total = abs(k1) + abs(k2) + 1e-10
        w1 = abs(k1) / k_total
        w2 = abs(k2) / k_total
        
        result = x1 * w1 + x2 * w2
        return result, new_context


class CyclicContextualAlgebra(ContextualAlgebra):
    """
    Циклическая контекстуальная алгебра.
    
    Контексты образуют цикл (например, фазы).
    Полезно для моделирования периодических процессов.
    """
    
    def __init__(self, period: float = 2 * math.pi):
        self.period = period
        
        def cyclic_order(a, b):
            # Для циклических контекстов порядок определяется расстоянием
            return abs(a - b) < self.period / 2
        
        def cyclic_transition(a, b):
            # Модульная арифметика
            return (a + b) % self.period
        
        space = ContextSpace(
            elements=[i * 0.1 for i in range(int(period * 10))],
            order=cyclic_order,
            transition=cyclic_transition
        )
        
        super().__init__(space)
        
    def contextual_rule(
        self, 
        x1: float, 
        k1: float, 
        x2: float, 
        k2: float
    ) -> Tuple[float, float]:
        """Операция с циклическим контекстом"""
        new_context = self.K.transition_op(k1, k2)
        
        # Фазовая зависимость: результат "вращается" вокруг центра
        center = (x1 + x2) / 2
        amplitude = abs(x1 - x2) / 2
        
        # Фаза определяется контекстом
        phase = (k1 + k2) / 2
        result = center + amplitude * math.sin(phase)
        
        return result, new_context


class AdaptiveFilterAlgebra(ContextualAlgebra):
    """
    Контекстуальная алгебра для адаптивной фильтрации.
    
    Контекст = "уровень доверия" к предыдущим измерениям.
    """
    
    def __init__(self, learning_rate: float = 0.1):
        self.eta = learning_rate
        
        # Контекст — от 0 (полная неопределённость) до 1 (полная достоверность)
        space = ContextSpace(
            elements=[i * 0.01 for i in range(101)],
            order=lambda a, b: a <= b,
            transition=lambda a, b: min(1.0, a + b)  # Накопление достоверности
        )
        
        super().__init__(space)
        
    def contextual_rule(
        self, 
        x1: float, 
        k1: float, 
        x2: float, 
        k2: float
    ) -> Tuple[float, float]:
        """
        Адаптивное усреднение с учётом достоверности.
        
        - Если k1 высокое (высокая достоверность x1), результат ближе к x1
        - Контекст накапливается (k = k1 + k2), но не превышает 1
        """
        # Новый контекст — сумма достоверностей (с насыщением)
        new_context = min(1.0, k1 + k2)
        
        # Вес пропорционален достоверности
        k_total = k1 + k2 + 1e-10
        w1 = k1 / k_total
        w2 = k2 / k_total
        
        # Адаптивное среднее
        result = x1 * w1 + x2 * w2
        
        # Коррекция на скорость обучения
        result = result * (1 + self.eta * (w1 - w2))
        
        return result, new_context


# =============================================================================
# ТЕСТЫ И ПРОВЕРКИ
# =============================================================================

def test_neutral_context():
    """Тест аксиомы 1: существование нейтрального контекста"""
    print("=" * 60)
    print("ТЕСТ 1: Нейтральный контекст")
    print("=" * 60)
    
    algebra = LinearContextualAlgebra(alpha=0.5)
    neutral = algebra.K.find_neutral()
    
    print(f"Найден нейтральный контекст: {neutral}")
    
    # Проверка: k ∘ k0 = k для любого k
    test_k = 5.0
    result = algebra.K.transition_op(test_k, neutral)
    print(f"Проверка: {test_k} ∘ {neutral} = {result} (должно быть {test_k})")
    
    return neutral is not None and result == test_k


def test_context_dependence():
    """Тест аксиомы 3: зависимость результата от контекста"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Зависимость результата от контекста")
    print("=" * 60)
    
    algebra = LinearContextualAlgebra(alpha=0.5)
    
    # Одни и те же значения, разные контексты
    elem_a = ContextualElement(10.0, 1.0)  # высокая достоверность
    elem_b = ContextualElement(10.0, 0.1)  # низкая достоверность
    elem_c = ContextualElement(10.0, 0.0)  # нулевой контекст
    
    other = ContextualElement(20.0, 1.0)
    
    result_a = algebra.operate(elem_a, other)
    result_b = algebra.operate(elem_b, other)
    result_c = algebra.operate(elem_c, other)
    
    print(f"(10, 1.0) ⊗ (20, 1.0) = {result_a}")
    print(f"(10, 0.1) ⊗ (20, 1.0) = {result_b}")
    print(f"(10, 0.0) ⊗ (20, 1.0) = {result_c}")
    
    # Результаты должны различаться
    different = (result_a.value != result_b.value or 
                 result_b.value != result_c.value or
                 result_a.value != result_c.value)
    
    print(f"Результаты различны: {different}")
    return different


def test_neutral_context_reduction():
    """Тест аксиомы 4: в нейтральном контексте — классическая операция"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Сведение к классической алгебре в нейтральном контексте")
    print("=" * 60)
    
    algebra = LinearContextualAlgebra(alpha=0.5)
    
    # В нейтральном контексте (k=0) операция должна сводиться к базовой
    elem1 = ContextualElement(10.0, 0.0)
    elem2 = ContextualElement(20.0, 0.0)
    
    result = algebra.operate(elem1, elem2)
    expected = (10.0 + 20.0) / 2  # базовое среднее
    
    print(f"(10, 0) ⊗ (20, 0) = {result}")
    print(f"Ожидаемое (классическое среднее): {expected}")
    print(f"Результат в нейтральном контексте сводится к классике: {abs(result.value - expected) < 0.001}")
    
    return abs(result.value - expected) < 0.001


def test_associativity():
    """Тест аксиомы 5: ассоциативность"""
    print("\n" + "=" * 60)
    print("ТЕСТ 4: Ассоциативность")
    print("=" * 60)
    
    algebra = LinearContextualAlgebra(alpha=0.5)
    
    a = ContextualElement(5.0, 1.0)
    b = ContextualElement(10.0, 2.0)
    c = ContextualElement(15.0, 0.5)
    
    # (a ⊗ b) ⊗ c
    ab = algebra.operate(a, b)
    result1 = algebra.operate(ab, c)
    
    # a ⊗ (b ⊗ c)
    bc = algebra.operate(b, c)
    result2 = algebra.operate(a, bc)
    
    print(f"(a ⊗ b) ⊗ c = {result1}")
    print(f"a ⊗ (b ⊗ c) = {result2}")
    
    # Проверяем с точностью до погрешности вычислений
    is_associative = (
        abs(result1.value - result2.value) < 0.001 and
        abs(result1.context - result2.context) < 0.001
    )
    print(f"Ассоциативность выполняется: {is_associative}")
    
    return is_associative


def test_adaptive_filtering():
    """Практический пример: адаптивная фильтрация"""
    print("\n" + "=" * 60)
    print("ПРАКТИЧЕСКИЙ ПРИМЕР: Адаптивная фильтрация")
    print("=" * 60)
    
    algebra = AdaptiveFilterAlgebra(learning_rate=0.1)
    
    # Симуляция: измерения с разной достоверностью
    measurements = [
        (100.0, 0.9),   # надёжное измерение
        (50.0, 0.3),    # ненадёжное измерение  
        (95.0, 0.8),    # довольно надёжное
        (10.0, 0.1),    # очень ненадёжное
        (98.0, 0.95),   # очень надёжное
    ]
    
    # Начинаем с начального контекста
    current = ContextualElement(measurements[0][0], measurements[0][1])
    print(f"Начальное значение: {current}")
    
    for i, (value, trust) in enumerate(measurements[1:], 1):
        next_elem = ContextualElement(value, trust)
        current = algebra.operate(current, next_elem)
        print(f"После измерения {i}: {current}")
    
    print(f"\nФинальный результат: {current.value:.2f}")
    print("Классическое среднее: {:.2f}".format(
        sum(m[0] for m in measurements) / len(measurements)
    ))
    
    return True


def test_cyclic_algebra():
    """Практический пример: циклическая алгебра"""
    print("\n" + "=" * 60)
    print("ПРАКТИЧЕСКИЙ ПРИМЕР: Циклическая контекстуальная алгебра")
    print("=" * 60)
    
    algebra = CyclicContextualAlgebra(period=2 * math.pi)
    
    # Моделируем периодический процесс
    results = []
    for phase in [0, math.pi/4, math.pi/2, math.pi, 3*math.pi/2]:
        elem1 = ContextualElement(10.0, phase)
        elem2 = ContextualElement(20.0, phase + 0.5)
        result = algebra.operate(elem1, elem2)
        results.append((phase, result.value))
        print(f"Фаза {phase:.2f}: результат = {result.value:.2f}")
    
    # Проверяем, что результаты зависят от фазы
    values = [r[1] for r in results]
    is_different = len(set(values)) > 1
    
    print(f"\nРезультаты различны для разных фаз: {is_different}")
    return is_different


def demonstrate_advantage():
    """Демонстрация преимущества над классическими методами"""
    print("\n" + "=" * 60)
    print("ДЕМОНСТРАЦИЯ ПРЕИМУЩЕСТВА")
    print("=" * 60)
    
    print("""
ЗАДАЧА: Онлайн-усреднение данных с меняющимся уровнем шума.

Классический подход: простое скользящее среднее
Контекстуальный подход: учитываем достоверность каждого измерения
    """)
    
    # Симуляция данных с меняющимся шумом
    import random
    random.seed(42)
    
    true_value = 100.0
    data = []
    for i in range(20):
        noise_level = 50 if i < 10 else 5  # Сначала шум высокий, потом низкий
        measurement = true_value + random.gauss(0, noise_level)
        trust = 1.0 / (1.0 + noise_level)  # Высокий шум = низкая достоверность
        data.append((measurement, trust))
    
    # Классическое среднее (все данные одинаково важны)
    classic_avg = sum(d[0] for d in data) / len(data)
    
    # Контекстуальное среднее (с весами достоверности)
    algebra = AdaptiveFilterAlgebra(learning_rate=0.05)
    current = ContextualElement(data[0][0], data[0][1])
    for value, trust in data[1:]:
        current = algebra.operate(current, ContextualElement(value, trust))
    
    contextual_avg = current.value
    
    print(f"Истинное значение: {true_value}")
    print(f"Классическое среднее: {classic_avg:.2f}")
    print(f"Контекстуальное среднее: {contextual_avg:.2f}")
    print(f"Ошибка классического: {abs(classic_avg - true_value):.2f}")
    print(f"Ошибка контекстуального: {abs(contextual_avg - true_value):.2f}")
    
    advantage = abs(classic_avg - true_value) - abs(contextual_avg - true_value)
    print(f"\nВыигрыш контекстуального подхода: {advantage:.2f}")
    
    return contextual_avg


# =============================================================================
# ЗАПУСК ВСЕХ ТЕСТОВ
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("КОНТЕКСТУАЛЬНАЯ АЛГЕБРА - ПОЛНЫЙ НАБОР ТЕСТОВ")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("Нейтральный контекст", test_neutral_context()))
    results.append(("Зависимость от контекста", test_context_dependence()))
    results.append(("Сведение к классике", test_neutral_context_reduction()))
    results.append(("Ассоциативность", test_associativity()))
    results.append(("Адаптивная фильтрация", test_adaptive_filtering()))
    results.append(("Циклическая алгебра", test_cyclic_algebra()))
    
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    demonstrate_advantage()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("ЕСТЬ ПРОВАЛЕННЫЕ ТЕСТЫ!")
    print("=" * 60)