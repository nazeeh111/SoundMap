"""Independent scalar oracles for the actual six-argument evaluator caller."""
import ast
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parents[1] / 'evaluate_multiple_source.py'


def functions():
    tree = ast.parse(SOURCE.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in {'create_result_row_single_frequency', 'evaluate_all_frequencies'}]
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    namespace.update(get_id=lambda group, num: 91,
                     get_true_coordinates=lambda group, num: dict(x_1=1, y_1=2, z_1=3, x_2=-2, y_2=1, z_2=0),
                     get_paths=lambda group, num: ('synthetic', []),
                     find_filepath=lambda path, pattern: None)
    return namespace


def row(freq, first, second):
    result = {'freq': freq}
    for i, point in enumerate([first, second, first, second], 1):
        result['assigned_source_' + str(i)] = 1 if i % 2 else 2
        result.update({axis + str(i): value for axis, value in zip('xyz', point)})
    return result


class EvaluatorContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = functions()

    def evaluate(self, rows):
        with tempfile.TemporaryDirectory(prefix='soundmap-oracle-') as tmp:
            path = Path(tmp) / 'predictions.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            original = path.read_bytes()
            try:
                return self.ns['evaluate_all_frequencies'](path, 7, 2, 'signal.h5', 2)
            finally:
                self.assertEqual(path.read_bytes(), original)

    def test_axis_mae_population_distance_dispersion_and_frequency_schema(self):
        # First-source errors: [-3,0,0], [0,4,0], and two zeros.
        # Source2 errors: constant [1,-2,2]. Scalar formulas below use math only.
        first = row(700, (-2, 2, 3), (-1, -1, 2))
        second = row(700, (1, 6, 3), (-1, -1, 2))
        for entry in (first, second):
            entry.update(x3=1, y3=2, z3=3)
        noisy, exact = self.evaluate([
            first, second, row(100, (1, 2, 3), (-2, 1, 0)),
        ])
        self.assertEqual((noisy['frequency'], exact['frequency']), (700, 100))
        expected_keys = {'id', 'group', 'num', 'frequency', 'mae_avg', 'std_avg'}
        expected_keys.update(f'{axis}{k}_mean' for axis in 'xyz' for k in (1, 2))
        expected_keys.update(f'{metric}_{k}' for metric in ('mae', 'std') for k in (1, 2))
        self.assertEqual(set(noisy), expected_keys)
        self.assertEqual((noisy['id'], noisy['group'], noisy['num']), (91, 7, 2))
        self.assertAlmostEqual(noisy['mae_1'], 7 / 12)
        self.assertAlmostEqual(noisy['mae_2'], 5 / 3)
        self.assertAlmostEqual(noisy['mae_avg'], 9 / 8)
        distances = [math.sqrt(6.0625), math.sqrt(9.5625), 1.25, 1.25]
        mean_distance = sum(distances) / 4
        dispersion = math.sqrt(sum((d - mean_distance) ** 2 for d in distances) / 4)
        self.assertAlmostEqual(noisy['std_1'], dispersion)
        self.assertEqual(noisy['std_2'], 0)
        self.assertAlmostEqual(noisy['std_avg'], dispersion / 2)
        self.assertEqual(tuple(noisy[f'{a}1_mean'] for a in 'xyz'), (.25, 3, 3))
        self.assertEqual(exact['mae_avg'], 0)

    def test_empty_assignment_nan_and_missing_file_errors_remain(self):
        entry = row(700, (1, 2, 3), (-2, 1, 0))
        entry.update({f'assigned_source_{i}': 0 for i in range(1, 5)})
        result, = self.evaluate([entry])
        self.assertTrue(math.isnan(result['mae_avg']))
        self.assertTrue(math.isnan(result['std_avg']))
        self.assertEqual(set(result), {'id', 'group', 'num', 'frequency', 'mae_avg', 'std_avg'})
        with tempfile.TemporaryDirectory(prefix='soundmap-missing-') as tmp:
            with self.assertRaises(FileNotFoundError):
                self.ns['evaluate_all_frequencies'](Path(tmp) / 'absent.csv', 7, 2, 'signal.h5', 2)

    def test_existing_temperature_provider_failure_is_not_swallowed(self):
        self.ns['find_filepath'] = lambda path, pattern: 'temperature.csv'

        def fail_temperature(path):
            raise ValueError('controlled temperature parse failure')

        self.ns['get_temperature'] = fail_temperature
        with self.assertRaisesRegex(ValueError, 'controlled temperature parse failure'):
            self.evaluate([row(700, (1, 2, 3), (-2, 1, 0))])


if __name__ == '__main__':
    unittest.main()
