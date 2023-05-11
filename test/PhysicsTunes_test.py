import PhysicsTunes as PT

s = 'normalization'
(T, dT) = PT.Flux('Atmospheric')
getattr(dT, s)
print(getattr(dT, s)(1, 2))
print(getattr(T, s)(1, 2))
