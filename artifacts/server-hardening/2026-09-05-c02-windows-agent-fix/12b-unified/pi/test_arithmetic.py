import unittest
from arithmetic import add

class AdditionTest(unittest.TestCase):
    def test_signed_addition(self):
        self.assertEqual(add(7, 5), 12)
        self.assertEqual(add(-4, 3), -1)

if __name__ == "__main__":
    unittest.main()
