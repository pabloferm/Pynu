import PhysicsTunes as PT

dict = {
    't12': 0.2,
    't13': 0.022,
    't23': 0.52,
    'dm21': 1e-5,
    'dm31': 0.0025,
    'dcp': 3.14,
    'Ordering': 'normal',
    'dm41': 1.4}
osc = PT.Oscillator('3Osc', 'Atmospheric', 4)
osc.SetOscillations(**dict)
