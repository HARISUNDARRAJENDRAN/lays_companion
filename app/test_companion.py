import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lays_companion import Companion


class LifecycleTests(unittest.TestCase):
    def make(self):
        return Companion(SimpleNamespace(command=None, speak=False), Mock())

    def run_worker(self, c):
        t = threading.Thread(target=c.worker)
        t.start()
        return t

    def test_newest_queued_command_wins(self):
        c = self.make()
        c.execute = Mock(return_value=0)
        c.submit('old')
        c.submit('new')
        c.commands.put(None)
        t = self.run_worker(c)
        t.join(2)
        self.assertFalse(t.is_alive())
        c.execute.assert_called_once_with('new')

    def test_cancellation_stays_set_until_previous_run_returns(self):
        c = self.make()
        started, release, second = threading.Event(), threading.Event(), threading.Event()
        def execute(goal):
            if goal == 'old':
                started.set()
                self.assertTrue(release.wait(2))
                self.assertTrue(c.cancel.is_set())
            else:
                self.assertFalse(c.cancel.is_set())
                second.set()
        c.execute = execute
        c.submit('old')
        t = self.run_worker(c)
        self.assertTrue(started.wait(2))
        generation = c.new_turn()
        c.submit('new', generation)
        self.assertTrue(c.cancel.is_set())
        release.set()
        self.assertTrue(second.wait(2))
        c.close()
        t.join(2)
        self.assertFalse(t.is_alive())

    def test_old_transcription_is_dropped(self):
        c = self.make()
        old = c.new_turn()
        c.new_turn()
        c.submit('stale transcript', old)
        self.assertTrue(c.commands.empty())

    def test_worker_survives_exception(self):
        c = self.make()
        seen = []
        def execute(goal):
            seen.append(goal)
            if goal == 'bad':
                c.commands.put((c.generation, 'good'))
                raise RuntimeError('test failure')
            c.commands.put(None)
        c.execute = execute
        c.submit('bad')
        t = self.run_worker(c)
        t.join(2)
        self.assertEqual(seen, ['bad', 'good'])
        self.assertFalse(t.is_alive())

    def test_one_shot_completion_on_error(self):
        c = self.make()
        c.a.command = 'bad'
        c.execute = Mock(side_effect=RuntimeError('test failure'))
        c.submit('bad')
        c.commands.put(None)
        t = self.run_worker(c)
        self.assertTrue(c.finished.wait(2))
        t.join(2)
        self.assertEqual(c.exit_code, 1)

    def test_voice_gpu_fallback(self):
        import hex_voice
        api = {'base': 'local-test'}
        with patch.object(hex_voice, '_ensure_hex', side_effect=[RuntimeError('unavailable'), api]) as ensure:
            self.assertEqual(hex_voice.ensure_hex('nvidia', 2), api)
            self.assertEqual(ensure.call_args_list[0].args, ('nvidia', 2))
            self.assertEqual(ensure.call_args_list[1].args, ('iris', 2))

    def test_explicit_iris_failure_is_reported(self):
        import hex_voice
        with patch.object(hex_voice, '_ensure_hex', side_effect=RuntimeError('unavailable')) as ensure:
            with self.assertRaises(RuntimeError):
                hex_voice.ensure_hex('iris', 2)
            ensure.assert_called_once_with('iris', 2)


if __name__ == '__main__':
    unittest.main()
