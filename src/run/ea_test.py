import json
import unittest

from run.ea import EAPopulation
from run.my_run import conv_args



class MyTestCase(unittest.TestCase):
    def test_init(self):
        global args, logger
        p = EAPopulation(args, logger)
        self.assertEqual(p.score, -1)
        self.assertEqual(p.reward, -1)
        self.assertEqual(p.states, -1)
        self.assertEqual(p.args, args)
        self.assertEqual(p.loger, logger)
        self.assertNotEqual(p.agent_param.agent, None)
        self.assertNotEqual(p.agent_param.optimiser_dict, None)

    def test_eval(self):
        global args, logger
        p = EAPopulation(args, logger)
        p.evaluate()
        print("score:",p.score)
        print("reward:",p.reward)
        print("states:",p.states)
        self.assertNotEqual(p.score, -1)
        self.assertNotEqual(p.reward, -1)
        self.assertNotEqual(p.states, -1)


    def test_train(self):
        global args, logger
        p = EAPopulation(args, logger)
        for i in range(10):
            p.train()
            p.evaluate()
            print("score:", p.score)
            print("reward:", p.reward)
            print("states:", p.states)


if __name__ == '__main__':
    p = EAPopulation(args, logger)
    for i in range(10):
        p.train()
        p.evaluate()
        print("score:", p.score)
        print("reward:", p.reward)
        print("states:", p.states)
