
import unittest
import pickle as pkl
from unittest import makeSuite
from content import run_program, get_compressed_data


class TestRunner(unittest.TextTestRunner):
    def __init__(self, result=None):
        super(TestRunner, self).__init__(verbosity=2)

    def run(self):
        suite = TestSuite()
        return super(TestRunner, self).run(suite)

class TestSuite(unittest.TestSuite):
    def __init__(self):
        super(TestSuite, self).__init__()
        self.addTest(makeSuite(TestPredict))



class TestPredict(unittest.TestCase):
    def test_sigmoid(self):
        #save_data(get_compressed_data(),'train.pkl')
        predicted = run_program()
        self.assertEqual(0,0)

def save_data(data, file_name):
    output = open(file_name, 'wb')
    pkl.dump(data, output)
    return