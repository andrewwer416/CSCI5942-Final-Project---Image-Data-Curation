# Tests

The suite creates tiny synthetic COCO annotations in temporary directories. It covers canonical
manifest construction, deterministic selection and caption sampling, validation, experiment-matrix
shape, and safe ZIP extraction. It never downloads COCO or writes into the project data directory.
