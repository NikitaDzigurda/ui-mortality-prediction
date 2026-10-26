import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import patch
import pandas as pd
from PySide6 import QtWidgets as W
from app import Window,summarize_shap


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=W.QApplication.instance() or W.QApplication([])

    def setUp(self):
        self.window=Window()

    def tearDown(self):
        self.window.close()

    def test_named_basins_preserve_filter_codes(self):
        frame=pd.DataFrame({'group_id':[1,2,3], 'vascular_group':[0.,1.,2.],
                            'hospital_death':[0,1,1], 'age_years':[50,60,70]})
        with patch('app.read_data',return_value=frame), patch.object(W.QFileDialog,'getOpenFileName',return_value=('prepared.csv','')):
            self.window.load()
        self.assertEqual(self.window.target.currentData(),'hospital_death')
        self.assertEqual(self.window.family.currentData(),'RF')
        self.assertEqual(self.window.basin.itemText(2),'АБШ / аневризма аорты')
        self.window.basin.setCurrentIndex(2)
        self.assertEqual(self.window.selected_frame().index.tolist(),[1])
        self.window.basin.setCurrentIndex(0)
        self.assertEqual(len(self.window.selected_frame()),3)

    def test_patient_filter_preserves_original_shap_position(self):
        w=self.window
        w.result={'target':'hospital_death','predictions':pd.DataFrame({
            'row_index':[4,18,37], 'outcome':[0,1,1], 'probability':[.1,.2,.3]})}
        w.explanation=object()
        with patch.object(w,'draw_patient'):
            w.update_patients()
            self.assertEqual([w.patient.itemData(i) for i in range(w.patient.count())],[1,2])
            w.patient.setCurrentIndex(1)
            w.deaths_only.setChecked(False)
            self.assertEqual(w.patient.count(),3)
            self.assertEqual(w.patient.currentData(),2)
            w.patient.setCurrentIndex(0)
            w.deaths_only.setChecked(True)
            self.assertEqual(w.patient.currentData(),1)
            w.result['predictions']['outcome']=0
            w.update_patients()
            self.assertEqual(w.patient.count(),0)
            self.assertFalse(w.patient.isEnabled())

    def test_global_shap_uses_absolute_values_and_all_records(self):
        w=self.window
        values=pd.DataFrame({'age_years':[1.,-1.,.5], 'creatinine_value':[-1.,1.,.25]})
        summary=summarize_shap(values)
        self.assertEqual(summary['Сумма |SHAP|'].tolist(),[2.5,2.25])
        self.assertAlmostEqual(summary['Средний |SHAP|'].iloc[0],2.5/3)
        w.result={'shap_summary':summary,'predictions':pd.DataFrame({'outcome':[0,1,1]})}
        self.assertTrue(w.deaths_only.isChecked())
        w.shap_mode.setCurrentIndex(w.shap_mode.findData('summary'))
        self.assertTrue(w.patient.isHidden())
        self.assertTrue(w.deaths_only.isHidden())
        widths=[bar.get_width() for bar in w.patient_canvas.figure.axes[0].patches]
        self.assertEqual(widths,[.75,2.5/3])
        w.shap_aggregation.setCurrentIndex(1)
        widths=[bar.get_width() for bar in w.patient_canvas.figure.axes[0].patches]
        self.assertEqual(widths,[2.25,2.5])


if __name__=='__main__':
    unittest.main()
