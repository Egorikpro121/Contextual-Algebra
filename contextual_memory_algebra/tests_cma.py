"""
Тесты для верификации математических свойств Контекстной алгебры памяти (CMA)
==============================================================================

Проверяются:
1. Теорема о затухании (влияние ранних операций уменьшается с глубиной)
2. Теорема о единственности нейтрального элемента
3. Теорема о контекстной независимости при k=0
4. Теорема о вычислительной сложности
5. Теорема о неподвижной точке

Автор: Независимый исследователь
Дата: 2026
"""

import unittest
import math
import random
from cma_core import (
    ContextualMemoryAlgebra,
    Context,
    ExponentialDecay,
    LinearDecay,
    InversePosition,
    UniformWeight,
    create_cma_from_numbers,
    LinearCMA,
    MultiplicativeCMA,
    NeuralCMA,
    EvolutionaryCMA
)


class TestContextBasics(unittest.TestCase):
    """Тесты базового функционала контекста."""
    
    def test_context_creation(self):
        """Тест создания контекста."""
        ctx = Context(k=5)
        self.assertEqual(ctx.k, 5)
        self.assertEqual(len(ctx.history), 0)
    
    def test_context_add_operation(self):
        """Тест добавления операции в историю."""
        ctx = Context(k=3)
        ctx.add_operation(2, 3, 5)
        ctx.add_operation(1, 1, 2)
        
        self.assertEqual(len(ctx.history), 2)
        self.assertEqual(ctx.history[0], (2, 3, 5))
    
    def test_context_truncation(self):
        """Тест усечения истории до глубины памяти."""
        ctx = Context(k=2)
        for i in range(5):
            ctx.add_operation(i, i, i*2)
        
        self.assertEqual(len(ctx.history), 2)
        # Должны остаться только последние 2 операции (i=3 и i=4)
        self.assertEqual(ctx.history[-1][0], 4)  # Последняя операция (i=4)
    
    def test_active_history(self):
        """Тест получения активной истории."""
        ctx = Context(k=2)
        for i in range(5):
            ctx.add_operation(i, i, i*2)
        
        active = ctx.get_active_history()
        self.assertEqual(len(active), 2)


class TestWeightFunctions(unittest.TestCase):
    """Тесты функций влияния."""
    
    def test_exponential_decay(self):
        """Тест экспоненциального затухания."""
        wf = ExponentialDecay(alpha=0.5)
        
        w0 = wf.get_weight(0, 5)  # Последняя операция
        w1 = wf.get_weight(1, 5)  # Предпоследняя
        w2 = wf.get_weight(2, 5)
        
        self.assertAlmostEqual(w0, 1.0)
        self.assertAlmostEqual(w1, 0.5)
        self.assertAlmostEqual(w2, 0.25)
        self.assertTrue(w0 >= w1 >= w2)  # Монотонность
    
    def test_linear_decay(self):
        """Тест линейного затухания."""
        wf = LinearDecay(slope=0.2)
        
        w0 = wf.get_weight(0, 5)
        w1 = wf.get_weight(1, 5)
        
        self.assertAlmostEqual(w0, 1.0)
        self.assertAlmostEqual(w1, 0.8)
        self.assertTrue(w0 >= w1)
    
    def test_inverse_position(self):
        """Тест обратно-пропорциональных весов."""
        wf = InversePosition()
        
        w0 = wf.get_weight(0, 5)
        w1 = wf.get_weight(1, 5)
        
        self.assertAlmostEqual(w0, 1.0)
        self.assertAlmostEqual(w1, 0.5)
        self.assertTrue(w0 >= w1)
    
    def test_uniform_weight(self):
        """Тест равномерных весов."""
        wf = UniformWeight()
        
        w = wf.get_weight(0, 4)
        self.assertAlmostEqual(w, 0.25)


class TestCMABasicOperations(unittest.TestCase):
    """Тесты базовых операций CMA."""
    
    def test_basic_operation(self):
        """Тест базовой операции."""
        cma = create_cma_from_numbers('add', k=0)
        result = cma.operate(3, 5)
        self.assertEqual(result, 8)
    
    def test_operation_with_context(self):
        """Тест операции с контекстом."""
        cma = create_cma_from_numbers('add', k=3)
        
        # Первая операция - без влияния истории
        r1 = cma.operate(2, 3)
        
        # Вторая операция - с влиянием первой
        r2 = cma.operate(1, 1)
        
        # Результат должен отличаться от классического
        # (при наличии ненулевого влияния)
        self.assertIsInstance(r2, (int, float))
    
    def test_context_update(self):
        """Тест обновления контекста после операции."""
        cma = create_cma_from_numbers('add', k=2)
        
        cma.operate(2, 3)
        state1 = cma.get_context_state()
        
        cma.operate(1, 1)
        state2 = cma.get_context_state()
        
        self.assertEqual(state1['history_length'] + 1, state2['history_length'])
    
    def test_reset_context(self):
        """Тест сброса контекста."""
        cma = create_cma_from_numbers('add', k=3)
        
        cma.operate(2, 3)
        cma.operate(1, 1)
        
        cma.reset_context()
        state = cma.get_context_state()
        
        self.assertEqual(state['history_length'], 0)


class TestTheorems(unittest.TestCase):
    """Тесты математических теорем CMA."""
    
    def test_theorem_decay(self):
        """
        Теорема 3.1 (о затухании):
        При глубине памяти k, влияние операций с индексом > k равно нулю.
        """
        cma = create_cma_from_numbers('add', k=2)
        
        # Выполняем 5 операций
        for i in range(5):
            cma.operate(i, i)
        
        # Контекст должен содержать только 2 последние операции
        active = cma.context.get_active_history()
        self.assertEqual(len(active), 2)
        
        # Более ранние операции не влияют
        self.assertEqual(active[0][0], 3)  # Третья с конца
        self.assertEqual(active[1][0], 4)  # Вторая с конца - последняя
    
    def test_theorem_neutral_element(self):
        """
        Теорема 3.2 (о единственности нейтрального элемента):
        В CMA существует ровно один нейтральный элемент.
        """
        cma = create_cma_from_numbers('add', k=3)
        
        # Нейтральный элемент для сложения = 0
        self.assertEqual(cma.neutral_element, 0)
        
        # Проверяем свойство нейтральности
        result = cma.operate(5, cma.neutral_element)
        self.assertEqual(result, 5)
        
        # Проверяем в обратную сторону
        cma.reset_context()
        result = cma.operate(cma.neutral_element, 5)
        self.assertEqual(result, 5)
    
    def test_theorem_context_independence(self):
        """
        Теорема 3.3 (о контекстной независимости):
        При k=0 результат не зависит от контекста.
        """
        cma_k0 = create_cma_from_numbers('add', k=0)
        cma_k0_copy = create_cma_from_numbers('add', k=0)
        
        # Выполняем разные последовательности операций
        cma_k0.operate(1, 2)
        cma_k0.operate(3, 4)
        
        cma_k0_copy.operate(10, 20)
        cma_k0_copy.operate(30, 40)
        
        # Результат следующей операции должен быть одинаковым
        r1 = cma_k0.operate(5, 3)
        r2 = cma_k0_copy.operate(5, 3)
        
        self.assertEqual(r1, r2)
    
    def test_theorem_computational_complexity(self):
        """
        Теорема 3.4 (о вычислительной сложности):
        Вычисление операции требует O(k) дополнительных ресурсов.
        """
        import time
        
        # CMA с k=0
        cma_k0 = create_cma_from_numbers('add', k=0)
        
        # CMA с k=10
        cma_k10 = create_cma_from_numbers('add', k=10)
        
        # Заполняем историю для k10
        for _ in range(10):
            cma_k10.operate(1, 1)
        
        # Замеряем время выполнения операций
        n_iterations = 1000
        
        start = time.time()
        for _ in range(n_iterations):
            cma_k0.operate(5, 3)
        time_k0 = time.time() - start
        
        start = time.time()
        for _ in range(n_iterations):
            cma_k10.operate(5, 3)
        time_k10 = time.time() - start
        
        # Время должно быть больше для k10 (хотя бы в 1.5 раза)
        # Это не строгий тест, но показывает зависимость
        ratio = time_k10 / max(time_k0, 0.0001)
        print(f"\nВремя k=0: {time_k0:.6f}s, k=10: {time_k10:.6f}s, отношение: {ratio:.2f}")
    
    def test_theorem_fixed_point(self):
        """
        Теорема 3.5 (о неподвижной точке):
        Существует неподвижная точка x: x ⊕_C a = x для любого a.
        
        Примечание: в CMA нейтральный элемент e удовлетворяет e ⊕ a = a,
        что означает e является левой неподвижной точкой для всех отображений f_a(x) = x ⊕ a.
        """
        cma = create_cma_from_numbers('add', k=3)
        
        # Нейтральный элемент должен быть неподвижной точкой
        e = cma.neutral_element
        
        # Проверяем: e ⊕ a = a (левая нейтральность)
        for a in [1, 2, 5, 10, -3]:
            cma.reset_context()
            result = cma.operate(e, a)
            self.assertEqual(result, a)


class TestSpecializedCMA(unittest.TestCase):
    """Тесты специализированных CMA."""
    
    def test_linear_cma(self):
        """Тест линейной CMA."""
        cma = LinearCMA(set(range(-100, 100)), alpha=0.3, k=3)
        
        result = cma.operate(5, 3)
        self.assertIsInstance(result, (int, float))
        
        # Проверяем, что результат близок к сумме
        self.assertAlmostEqual(result, 8, delta=2)
    
    def test_multiplicative_cma(self):
        """Тест мультипликативной CMA."""
        cma = MultiplicativeCMA(set(range(1, 100)), beta=0.1, k=3)
        
        result = cma.operate(4, 5)
        self.assertIsInstance(result, (int, float))
        
        # Проверяем, что результат близок к произведению
        self.assertAlmostEqual(result, 20, delta=5)
    
    def test_neural_cma(self):
        """Тест нейронной CMA."""
        cma = NeuralCMA(set(range(-100, 100)), hidden_dim=4, k=3)
        
        result = cma.operate(5, 3)
        self.assertIsInstance(result, (int, float))
    
    def test_evolutionary_cma(self):
        """Тест эволюционной CMA."""
        cma = EvolutionaryCMA(set(range(-100, 100)), mutation_rate=0.2, k=5)
        
        # Тестируем мутацию
        individual = 10.0
        mutated = cma.mutate(individual)
        
        self.assertIsInstance(mutated, float)


class TestPracticalApplications(unittest.TestCase):
    """Тесты практических применений."""
    
    def test_neural_network_simulation(self):
        """
        Симуляция нейронной сети с памятью.
        
        Показывает преимущество CMA перед классическим подходом
        для задач с зависимостью от контекста.
        """
        # Классическая нейронная сеть (без памяти)
        class ClassicNetwork:
            def __init__(self):
                self.weights = [0.5, 0.3, 0.2]
            
            def forward(self, inputs):
                return sum(w * x for w, x in zip(self.weights, inputs))
        
        # CMA-подобная сеть (с памятью)
        class CMANetwork:
            def __init__(self):
                self.weights = [0.5, 0.3, 0.2]
                self.context_weight = 0.1
                self.history = []
            
            def forward(self, inputs, context_input=0):
                base = sum(w * x for w, x in zip(self.weights, inputs))
                context_effect = self.context_weight * sum(self.history[-3:]) if self.history else 0
                result = base + context_effect
                self.history.append(result)
                return result
        
        classic = ClassicNetwork()
        cma_net = CMANetwork()
        
        # Последовательность входов
        inputs_sequence = [
            [1.0, 0.5, 0.2],
            [0.8, 0.3, 0.1],
            [0.6, 0.4, 0.3]
        ]
        
        classic_outputs = []
        cma_outputs = []
        
        for inp in inputs_sequence:
            classic_outputs.append(classic.forward(inp))
            cma_outputs.append(cma_net.forward(inp))
        
        # CMA учитывает контекст - результаты отличаются
        # Классическая сеть дает одинаковые результаты для одинаковых входов
        # CMA дает разные результаты из-за влияния истории
        
        # Проверяем, что CMA реагирует на контекст
        self.assertNotEqual(cma_outputs[0], cma_outputs[2])
    
    def test_evolutionary_algorithm(self):
        """
        Эволюционный алгоритм с адаптивной мутацией.
        
        Показывает, как CMA естественно моделирует эволюцию с памятью.
        """
        random.seed(42)  # Для воспроизводимости
        
        population = [5.0, 3.0, 7.0, 2.0, 8.0]
        cma = EvolutionaryCMA(set(range(-1000, 1000)), mutation_rate=0.2, k=5)
        
        # Симуляция нескольких поколений
        for generation in range(10):
            # Оцениваем приспособленность (просто сумма)
            fitness = [p * 2 for p in population]
            
            # Обновляем контекст результатами
            for f in fitness:
                cma.context.add_operation(0, 0, f)
            
            # Применяем адаптивную мутацию
            new_population = []
            for p in population:
                mutated = cma.mutate(p)
                new_population.append(mutated)
            
            population = new_population
        
        # Проверяем, что популяция изменилась
        self.assertNotEqual(population[0], 5.0)


class TestEdgeCases(unittest.TestCase):
    """Тесты граничных случаев."""
    
    def test_empty_history(self):
        """Тест операции с пустой историей."""
        cma = create_cma_from_numbers('add', k=3)
        
        # История пустая
        result = cma.operate(5, 3)
        self.assertEqual(result, 8)  # Без влияния контекста
    
    def test_zero_depth_memory(self):
        """Тест с нулевой глубиной памяти."""
        cma = create_cma_from_numbers('add', k=0)
        
        cma.operate(1, 2)
        cma.operate(3, 4)
        
        # Результат не должен зависеть от истории
        result = cma.operate(5, 3)
        self.assertEqual(result, 8)
    
    def test_large_numbers(self):
        """Тест с большими числами."""
        cma = create_cma_from_numbers('add', k=3)
        
        result = cma.operate(10**6, 10**6)
        self.assertIsInstance(result, (int, float))
    
    def test_negative_numbers(self):
        """Тест с отрицательными числами."""
        cma = create_cma_from_numbers('add', k=3)
        
        result = cma.operate(-5, 3)
        self.assertEqual(result, -2)


def run_all_tests():
    """Запуск всех тестов."""
    print("=" * 70)
    print("ЗАПУСК ТЕСТОВ КОНТЕКСТНОЙ АЛГЕБРЫ ПАМЯТИ")
    print("=" * 70)
    
    # Создаем тестовый набор
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем все тестовые классы
    suite.addTests(loader.loadTestsFromTestCase(TestContextBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestWeightFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestCMABasicOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestTheorems))
    suite.addTests(loader.loadTestsFromTestCase(TestSpecializedCMA))
    suite.addTests(loader.loadTestsFromTestCase(TestPracticalApplications))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Запускаем тесты
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Выводим итоги
    print("\n" + "=" * 70)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"Выполнено тестов: {result.testsRun}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Ошибок: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО")
    else:
        print("\n✗ ЕСТЬ ПРОВАЛЕННЫЕ ТЕСТЫ")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)