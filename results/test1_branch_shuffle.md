# Test 1 (MAGNet-branch exogenous shuffle), SIG v2, seeds [0, 42, 123, 2024, 3407]

| Backbone | Data | Term | cells | mean intact MAE | mean shuffled MAE | mean dMAE % | shuffle-worse |
|---|---|---|---|---|---|---|---|
| DAG | real | short | 12/12 | 0.3596 | 0.3620 | +0.81% | 9/12 |
| DAG | real | long | 12/12 | 0.5518 | 0.5554 | +1.07% | 5/12 |
| DAG | synthetic | short | 5/5 | 0.5803 | 0.6829 | +17.98% | 5/5 |
| DAG | synthetic | long | 5/5 | 0.6556 | 0.6556 | +0.00% | 3/5 |
| GCGNet | real | short | 12/12 | 0.3569 | 0.3608 | +1.28% | 9/12 |
| GCGNet | real | long | 12/12 | 0.5120 | 0.5150 | +0.89% | 7/12 |
| GCGNet | synthetic | short | 4/5 | 0.6125 | 0.6242 | +2.04% | 3/4 |
| GCGNet | synthetic | long | 3/5 | 0.6609 | 0.6611 | +0.03% | 2/3 |
| TimeXer | real | short | 12/12 | 0.3838 | 0.3918 | +2.26% | 9/12 |
| TimeXer | real | long | 12/12 | 0.5780 | 0.5796 | +0.42% | 4/12 |
| TimeXer | synthetic | short | 5/5 | 0.5540 | 0.6888 | +25.86% | 5/5 |
| TimeXer | synthetic | long | 5/5 | 0.6581 | 0.6584 | +0.04% | 4/5 |
| TimeKAN | real | short | 12/12 | 0.3686 | 0.3953 | +9.56% | 9/12 |
| TimeKAN | real | long | 12/12 | 0.5358 | 0.5439 | +1.53% | 6/12 |
| TimeKAN | synthetic | short | 5/5 | 0.5632 | 0.8297 | +49.38% | 5/5 |
| TimeKAN | synthetic | long | 5/5 | 0.6583 | 0.6923 | +5.17% | 5/5 |
| PatchTST | real | short | 12/12 | 0.3513 | 0.3695 | +6.54% | 9/12 |
| PatchTST | real | long | 12/12 | 0.5528 | 0.5546 | +0.20% | 3/12 |
| PatchTST | synthetic | short | 5/5 | 0.5455 | 0.7839 | +46.74% | 5/5 |
| PatchTST | synthetic | long | 5/5 | 0.6615 | 0.6615 | -0.00% | 3/5 |
| TFT | real | short | 12/12 | 0.3465 | 0.3788 | +12.65% | 9/12 |
| TFT | real | long | 12/12 | 0.5092 | 0.5207 | +3.31% | 7/12 |
| TFT | synthetic | short | 5/5 | 0.5509 | 0.8765 | +61.77% | 5/5 |
| TFT | synthetic | long | 5/5 | 0.6521 | 0.8017 | +23.14% | 5/5 |
| DLinear | real | short | 12/12 | 0.3464 | 0.3658 | +7.33% | 11/12 |
| DLinear | real | long | 12/12 | 0.5190 | 0.5227 | +0.74% | 6/12 |
| DLinear | synthetic | short | 5/5 | 0.5545 | 0.8079 | +47.74% | 5/5 |
| DLinear | synthetic | long | 5/5 | 0.6491 | 0.6617 | +1.84% | 5/5 |
