import pylidc as pl


qu = pl.query(pl.Scan).filter(pl.Scan.slice_thickness <= 1)
print(qu.count())
# => 97

scan = qu.first()

print scan.patient_id, scan.pixel_spacing, scan.slice_thickness
ann = pl.query(pl.Annotation).first()
vol, seg = ann.uniform_cubic_resample(side_length = 100)
print(vol.shape, seg.shape)
# => (101, 101, 101) (101, 101, 101)

import matplotlib.pyplot as plt
plt.imshow( vol[:,50,:] * (seg[:,50,:]*0.8 + 0.2), cmap=plt.cm.gray)
plt.show()
