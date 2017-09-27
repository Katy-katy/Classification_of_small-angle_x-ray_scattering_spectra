import os.path
import glob
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

import paws.api

paw = paws.api.start()
paw.add_wf('classifier_test')
paw.activate_op('IO.CSV.CSVToXYData')
paw.activate_op('PROCESSING.SAXS.SpectrumProfiler')
paw.activate_op('IO.MODELS.SAXS.LoadSAXSClassifiers')
paw.activate_op('PROCESSING.SAXS.SpectrumClassifier')

paw.add_op('read_csv','IO.CSV.CSVToXYData')
paw.add_op('profile','PROCESSING.SAXS.SpectrumProfiler')
paw.add_op('load_classifiers','IO.MODELS.SAXS.LoadSAXSClassifiers')
paw.add_op('classify','PROCESSING.SAXS.SpectrumClassifier')

paw.set_input('profile','q','read_csv.outputs.x')
paw.set_input('profile','I','read_csv.outputs.y')
paw.set_input('classify','profiler_output','profile.outputs.features')
paw.set_input('classify','scalers','load_classifiers.outputs.scalers')
paw.set_input('classify','classifiers','load_classifiers.outputs.classifiers')

# data frame with test set
test = pd.read_csv('paws_test.csv')
names = list(test['name'])
files_to_test = []
for f in names:
    f = '2016_saxs_data/' + f + '.csv'
    files_to_test.append(f)

bad_data_labels = []
bad_data_pr = []
form_factor_scattering = []
form_pr = []
precursor_scattering = []
prec_pr = []
diffraction_peaks = []
peaks_pr = []

for fname in files_to_test:

    paw.set_input('read_csv','file_path',fname)
    paw.execute()
    f = paw.get_output('classify','population_flags')

    #print (f)

    bad_data_labels.append(f['bad_data'][0])
    bad_data_pr.append(f['bad_data'][1])
    form_factor_scattering.append(f['form_factor_scattering'][0])
    form_pr.append(f['form_factor_scattering'][1])
    precursor_scattering.append(f['precursor_scattering'][0])
    prec_pr.append(f['precursor_scattering'][1])
    diffraction_peaks.append(f['diffraction_peaks'][0])
    peaks_pr.append(f['diffraction_peaks'][1])

print(test.shape)
print(len(bad_data_labels))
test['paws_bad_data'] =  bad_data_labels
test['paws_bad_data_pr'] =  bad_data_pr # propability to have this bad_data label
test['paws_form'] =  form_factor_scattering
test['paws_form_pr'] =  form_pr
test['paws_precursor'] =  precursor_scattering
test['paws_precursor_pr'] =  prec_pr
test['paws_structure'] =  diffraction_peaks
test['paws_structure_pr'] =  peaks_pr

print("Bad Data accuracy: ", accuracy_score(test['bad_data'], test['paws_bad_data']))
print("form_factor_scattering accuracy: ", accuracy_score(test['form'], test['paws_form']))
print("precursor_scattering accuracy: ", accuracy_score(test['precursor'], test['paws_precursor']))
print("diffraction_peaks accuracy: ", accuracy_score(test['structure'], test['paws_structure']))

test.to_csv("paws_results.csv")
