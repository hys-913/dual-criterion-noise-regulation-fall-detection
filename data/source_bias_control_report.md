# Source-Shortcut Diagnostic

## Why This Check Is Needed

The released split contains Leeds Millennium fall and normal images, while COCO2017 contributes normal images only. A model could therefore appear strong if it learns source-specific cues. This diagnostic quantifies that shortcut.

## Source-Only Rule

Rule: `leeds_millennium => fall`, `coco2017 => normal`.

On the held-out test split this rule obtains 89.3% raw accuracy, 100.0% sensitivity, 72.6% specificity, and 86.3% balanced accuracy.

The high raw accuracy shows that source shortcuts are a real confound. The low specificity shows the shortcut fails on Leeds normal images.

## Source/Label Counts

| split   | source_dataset   | label   |   count |
|:--------|:-----------------|:--------|--------:|
| test    | coco2017         | normal  |     151 |
| test    | leeds_millennium | fall    |     323 |
| test    | leeds_millennium | normal  |      57 |
| train   | coco2017         | normal  |    1206 |
| train   | leeds_millennium | fall    |    2620 |
| train   | leeds_millennium | normal  |     422 |
| val     | coco2017         | normal  |     312 |
| val     | leeds_millennium | fall    |     655 |
| val     | leeds_millennium | normal  |      95 |

## Source-Only Metrics

| split   | rule                                     |    n |   accuracy |     f1 |   sensitivity |   specificity |   balanced_accuracy |   tp |   fn |   fp |   tn |
|:--------|:-----------------------------------------|-----:|-----------:|-------:|--------------:|--------------:|--------------------:|-----:|-----:|-----:|-----:|
| train   | leeds_millennium=>fall; coco2017=>normal | 4248 |     0.9007 | 0.9255 |        1.0000 |        0.7408 |              0.8704 | 2620 |    0 |  422 | 1206 |
| val     | leeds_millennium=>fall; coco2017=>normal | 1062 |     0.9105 | 0.9324 |        1.0000 |        0.7666 |              0.8833 |  655 |    0 |   95 |  312 |
| test    | leeds_millennium=>fall; coco2017=>normal |  531 |     0.8927 | 0.9189 |        1.0000 |        0.7260 |              0.8630 |  323 |    0 |   57 |  151 |
| all     | leeds_millennium=>fall; coco2017=>normal | 5841 |     0.9017 | 0.9261 |        1.0000 |        0.7441 |              0.8720 | 3598 |    0 |  574 | 1669 |
