import pylidc as pl

qu = pl.query(pl.Scan)
print qu.count()


#print(scan.patient_id, scan.pixel_spacing, scan.slice_thickness)
count = 0
for scan in qu:
    #print scan.series_instance_uid,"STUDY",scan.study_instance_uid
    clusters = scan.annotations_with_matching_overlap()
    for cluster in clusters:
        count += len(cluster)
print count
"""
#for ann in anns:
    for ann in cluster:
        print "diameter",ann.estimate_diameter(),"features_vals",ann.feature_vals(),"centroid",ann.centroid(),
        "bbox",ann.bbox()
        ann.visualize_in_scan()
"""
