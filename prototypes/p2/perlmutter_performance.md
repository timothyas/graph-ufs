# Perlmutter I/O Performance

## P2 Comparison

| Batch Size |     |     |     |     |     |     |     |      |     |
|------------|-----|-----|-----|-----|-----|-----|-----|------|-----|
|            | 1   | 2   | 4   | 8   | 16  | 32  | 64  | 128  | XTS |
| 16         | 1.5 | 8.3 | .55 | .48 | .39 | .42 | .43 | .40  | .29 |
| 32         | 2.6 | 1.5 | 1.0 | .81 | .77 | .75 | .82 | .76  | .56 |
| 64         | 4.8 | 2.7 | 2.0 | 1.7 | 1.5 | 1.3 | 1.3 | 1.34 | 1.1 |


This is about 4-4.5 times faster than what we get on Azure
