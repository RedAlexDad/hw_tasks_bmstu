import unittest
import os
import filecmp
from main import ResistorCircuitSimulator

class TestResistorCircuitSimulator(unittest.TestCase):
    def test_1(self):
        input_file = 'file/input_1.txt'
        answer_file = 'file/answer_1.txt'
        output_file = 'file/output_1.txt'

        self.simulator = ResistorCircuitSimulator(input_file, output_file)
        self.simulator.run()

        self.assertTrue(filecmp.cmp(output_file, answer_file, shallow=False))

        if os.path.exists(output_file):
            os.remove(output_file)

    def test_2(self):
        input_file = 'file/input_2.txt'
        answer_file = 'file/answer_2.txt'
        output_file = 'file/output_2.txt'

        self.simulator = ResistorCircuitSimulator(input_file, output_file)
        self.simulator.run()

        self.assertTrue(filecmp.cmp(output_file, answer_file, shallow=False))

        if os.path.exists(output_file):
            os.remove(output_file)

    # def test_3(self):
    #     input_file = 'file/input_3.txt'
    #     answer_file = 'file/answer_3.txt'
    #     output_file = 'file/output_3.txt'
    #
    #     self.simulator = ResistorCircuitSimulator(input_file, output_file)
    #     self.simulator.run()
    #
    #     self.assertTrue(filecmp.cmp(output_file, answer_file, shallow=False))
    #
    #     if os.path.exists(output_file):
    #         os.remove(output_file)
    #
    # def test_4(self):
    #     input_file = 'file/input_4.txt'
    #     answer_file = 'file/answer_4.txt'
    #     output_file = 'file/output_4.txt'
    #
    #     self.simulator = ResistorCircuitSimulator(input_file, output_file)
    #     self.simulator.run()
    #
    #     self.assertTrue(filecmp.cmp(output_file, answer_file, shallow=False))
    #
    #     if os.path.exists(output_file):
    #         os.remove(output_file)

    # def test_5(self):
    #     input_file = 'file/input_5.txt'
    #     answer_file = 'file/answer_5.txt'
    #     output_file = 'file/output_5.txt'
    #
    #     self.simulator = ResistorCircuitSimulator(input_file, output_file)
    #     self.simulator.run()
    #
    #     self.assertTrue(filecmp.cmp(output_file, answer_file, shallow=False))
    #
    #     if os.path.exists(output_file):
    #         os.remove(output_file)

if __name__ == '__main__':
    unittest.main()
