"""Real pandas/NumPy metrics; AST loading avoids device imports and batch side effects."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / 'evaluate_multiple_source.py'


def evaluation_functions():
    tree = ast.parse(SOURCE.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {'create_result_row_single_frequency', 'evaluate_all_frequencies'}]
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


def prediction_row(freq, positions, assignments):
    # Existing prediction CSV columns plus the assigned_source_N evaluation fields.
    row = dict(id=77, group=7, num=1, block=1, freq=freq)
    for index, (position, assigned) in enumerate(zip(positions, assignments), 1):
        for axis, value in zip('xyz', position):
            row[f'{axis}{index}'] = value
        row[f's{index}'] = 42.0
        row[f'assigned_source_{index}'] = assigned
    return row


class MultiSourceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.ns = evaluation_functions()
        self.ns.update(
            get_id=Mock(return_value=77),
            get_true_coordinates=Mock(return_value=dict(x_1=0, y_1=0, z_1=1, x_2=2, y_2=1, z_2=3)),
            get_paths=Mock(return_value=('synthetic-measurement', [])),
            find_filepath=Mock(side_effect=lambda path, pattern: 'synthetic-temperature' if 'temperature' in pattern else 'synthetic-signal'),
            get_temperature=Mock(return_value=(22, .5, 343)),
        )

    def evaluate_csv(self, rows, num_sources=2, num=1):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.csv'
            pd.DataFrame(rows).to_csv(path, index=False)
            original = path.read_bytes()
            result = self.ns['evaluate_all_frequencies'](path, 7, num, 'signal_2s_1.h5', num_sources)
            self.assertEqual(path.read_bytes(), original, 'evaluation must not rewrite prediction CSV')
            return result

    def test_actual_caller_keeps_per_source_metrics_and_frequency_rows(self):
        # Source1 x errors [-.6,-.3,.3,.6]; source2 y errors twice as large.
        rows = [prediction_row(1000, [(-.6, 0, 1), (2, -.2, 3), (.3, 0, 1), (2, 1.6, 3)], [1, 2, 1, 2]),
                prediction_row(1000, [(-.3, 0, 1), (2, .4, 3), (.6, 0, 1), (2, 2.2, 3)], [1, 2, 1, 2]),
                prediction_row(2000, [(0, 0, 1), (2, 1, 3), (0, 0, 1), (2, 1, 3)], [1, 2, 1, 2])]
        results = self.evaluate_csv(rows)
        self.assertEqual([r['frequency'] for r in results], [1000, 2000])
        expected_keys = {'id', 'group', 'num', 'frequency', 'mae_avg', 'std_avg'}
        expected_keys.update(f'{axis}{k}_mean' for k in (1, 2) for axis in 'xyz')
        expected_keys.update(f'{metric}_{k}' for k in (1, 2) for metric in ('mae', 'std'))
        for row in results:
            self.assertEqual(set(row), expected_keys)
            self.assertEqual((row['id'], row['group'], row['num']), (77, 7, 1))
            for k, point in ((1, (0, 0, 1)), (2, (2, 1, 3))):
                for axis, value in zip('xyz', point):
                    self.assertAlmostEqual(row[f'{axis}{k}_mean'], value)
        noisy, exact = results
        for metric in ('mae', 'std'):
            self.assertAlmostEqual(noisy[f'{metric}_1'], .15)
            self.assertAlmostEqual(noisy[f'{metric}_2'], .30)
            self.assertAlmostEqual(noisy[f'{metric}_avg'], .225)
            self.assertAlmostEqual(exact[f'{metric}_avg'], 0)
        self.ns['get_temperature'].assert_called_once_with('synthetic-temperature')

    def test_actual_three_source_caller_retains_legacy_coordinates(self):
        points = [(-.47, -.01, 1.445), (-1.122, -.67, 1.465), (.345, -.61, 1.145)]
        result, = self.evaluate_csv([prediction_row(3150, [*points, points[2]], [1, 2, 3, 3])], num_sources=3)
        self.ns['get_true_coordinates'].assert_not_called()
        self.assertEqual(result['frequency'], 3150)
        for k, point in enumerate(points, 1):
            for axis, value in zip('xyz', point):
                self.assertAlmostEqual(result[f'{axis}{k}_mean'], value)
            self.assertEqual(result[f'mae_{k}'], 0)
            self.assertEqual(result[f'std_{k}'], 0)
        self.assertEqual(result['mae_avg'], 0)
        self.assertEqual(result['std_avg'], 0)

    def test_unassigned_source_is_omitted_from_existing_averages(self):
        frame = pd.DataFrame([prediction_row(1000, [(0, 0, 1)] * 4, [1] * 4)])
        row = self.ns['create_result_row_single_frequency'](frame, 7, 1, 1000, [(0, 0, 1), (2, 1, 3)], 2)
        self.assertNotIn('mae_2', row)
        self.assertNotIn('x2_mean', row)
        self.assertEqual(row['mae_avg'], 0)
        self.assertEqual(row['std_avg'], 0)

    def test_missing_assignment_column_still_raises(self):
        row = prediction_row(1000, [(0, 0, 1)] * 4, [1] * 4)
        del row['assigned_source_4']
        with self.assertRaises(KeyError):
            self.ns['create_result_row_single_frequency'](pd.DataFrame([row]), 7, 1, 1000, [(0, 0, 1)], 1)


if __name__ == '__main__':
    unittest.main()
