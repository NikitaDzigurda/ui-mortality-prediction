import unittest
import numpy as np
import pandas as pd
from analysis import statistics, splits, run

class AnalysisTests(unittest.TestCase):
    def test_effect_direction(self):
        result=statistics(pd.DataFrame({'x':[1,2,3,10,11,12]}),pd.Series([0,0,0,1,1,1]),[])
        self.assertEqual(result.iloc[0]['Эффект'],1)
        self.assertAlmostEqual(result.iloc[0]['p'],result.iloc[0]['q (FDR)'])

    def test_constant_and_unknown(self):
        result=statistics(pd.DataFrame({'x':[np.nan]*6,'z':[1]*6}),pd.Series([0,0,0,1,1,1]),[])
        self.assertTrue(result.p.isna().all())

    def test_patient_separation(self):
        groups=np.repeat(np.arange(30),2)
        y=np.repeat(np.arange(30)%2,2)
        for train,test in splits(np.zeros((60,1)),y,groups,3,2026):
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_rare_event_rejected(self):
        with self.assertRaises(ValueError):
            splits(np.zeros((10,1)),np.array([1]+[0]*9),np.arange(10),3,2026)

    def test_target_cannot_be_predictor(self):
        with self.assertRaises(ValueError):
            run(pd.DataFrame({'y':[0,1]*10}),['y'],[],'y',1,None,'LR',1,2,lambda _:None,lambda:False)

if __name__=='__main__':unittest.main()
