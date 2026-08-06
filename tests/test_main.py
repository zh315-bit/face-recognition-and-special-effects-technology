import subprocess
import sys
import unittest


class MainProgramTests(unittest.TestCase):
    def test_prints_docker_hello_world_message(self):
        completed = subprocess.run(
            [sys.executable, "main.py"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.stdout, "Hello, World from Docker!\n")


if __name__ == "__main__":
    unittest.main()
