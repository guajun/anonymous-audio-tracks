# Matched teacher curriculum (fixed current-course test)

| Arm | Course | Actual IoU A/B | Teacher IoU A/B | Actual empty FP | Extra area | Count MAE | Gap relation correct | Confidence |
|---|---|---|---|---:|---:|---:|---:|---:|
| teacher-amplitude | C1 | 0.3152/0.0651 | 0.8751/0.5486 | 48.79% | 1.149 | 3.679 | 3.33% | 0.786 |
| teacher-amplitude | C2-low | 0.2479/0.0450 | 0.9446/0.7770 | 62.34% | 1.068 | 4.672 | 29.51% | 0.740 |
| teacher-amplitude | C2-high | 0.5665/0.1153 | 0.9435/0.5491 | 67.75% | 1.256 | 4.393 | 17.74% | 0.798 |
| teacher-amplitude | C3 | 0.5315/0.3846 | 0.9312/0.8855 | 93.75% | 0.899 | 6.182 | 41.94% | 0.742 |
| teacher-identity | C1 | 0.3777/0.5237 | 0.9004/0.6557 | 39.65% | 0.860 | 2.590 | 50.00% | 0.785 |
| teacher-identity | C2-low | 0.4625/0.3112 | 0.9102/0.7526 | 70.58% | 0.770 | 4.626 | 39.34% | 0.813 |
| teacher-identity | C2-high | 0.5122/0.1399 | 0.9297/0.6097 | 55.27% | 0.619 | 3.748 | 45.16% | 0.831 |
| teacher-identity | C3 | 0.9339/0.6797 | 0.9339/0.8771 | 97.44% | 0.322 | 6.229 | 93.55% | 0.815 |
